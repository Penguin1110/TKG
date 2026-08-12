"""Discover alternating entity/time question seeds from Wikidata and Wikipedia.

The first supported pattern is deliberately narrow and auditable::

    team@cutoff -> coach@cutoff
    coach@later -> another team@later
    another team@still-later -> successor coach@still-later

Wikidata P286 start/end qualifiers nominate candidates.  Exact Wikipedia
revisions, visible text, and hyperlinks are the admission evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests

from tkg.experiment.model_cutoffs import get_model_cutoff
from tkg.experiment.attribute_tail_generation import (
    generate_attribute_tail_variants, select_diverse_variants,
)
from tkg.experiment.multihop_generation import MultiHopSeed, validate_chain
from tkg.experiment.question_generation import _fold, _plain_text
from tkg.experiment.relation_catalog import PERSON_RELATIONS
from tkg.experiment.results import assert_new_output_path
from tkg.wikipedia.backend import WikipediaError, WikipediaPageBackend
from tkg.wikipedia.pageviews import PopularitySnapshot, fetch_top_pageviews


WDQS_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "TKG-temporal-candidate-discovery/0.13"
DATE_OFFSETS = (0, 1, 3, 7)


def attribute_quota_capacity(
    max_total: int, *, max_per_family: int, max_per_property: int
) -> int:
    """Maximum selectable catalog tails under the configured hard quotas."""
    properties_by_family: dict[str, set[str]] = {}
    for relation in PERSON_RELATIONS:
        properties_by_family.setdefault(relation.family, set()).add(relation.property_id)
    catalog_capacity = sum(
        min(max_per_family, len(properties) * max_per_property)
        for properties in properties_by_family.values()
    )
    return min(max_total, catalog_capacity)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _date(value: datetime) -> str:
    return value.date().isoformat()


def _article_title(url: str) -> str:
    path = unquote(urlparse(url).path)
    marker = "/wiki/"
    if marker not in path:
        raise ValueError(f"not an English Wikipedia article URL: {url!r}")
    return path.split(marker, 1)[1].replace("_", " ")


def _binding(row: dict[str, Any], key: str) -> str:
    value = row.get(key, {})
    return str(value.get("value", "")).strip() if isinstance(value, dict) else ""


@dataclass(frozen=True)
class P286Candidate:
    anchor_title: str
    coach_title: str
    middle_title: str
    successor_title: str
    middle_start: str
    successor_start: str
    anchor_qid: str = ""
    coach_qid: str = ""
    middle_qid: str = ""
    successor_qid: str = ""

    @classmethod
    def from_binding(cls, row: dict[str, Any]) -> "P286Candidate":
        return cls(
            anchor_title=_article_title(_binding(row, "anchorArticle")),
            coach_title=_article_title(_binding(row, "coachArticle")),
            middle_title=_article_title(_binding(row, "middleArticle")),
            successor_title=_article_title(_binding(row, "successorArticle")),
            middle_start=_date(_parse_time(_binding(row, "middleStart"))),
            successor_start=_date(_parse_time(_binding(row, "successorStart"))),
            anchor_qid=_qid(_binding(row, "anchor")) if _binding(row, "anchor") else "",
            coach_qid=_qid(_binding(row, "coach")) if _binding(row, "coach") else "",
            middle_qid=_qid(_binding(row, "middle")) if _binding(row, "middle") else "",
            successor_qid=(
                _qid(_binding(row, "successor")) if _binding(row, "successor") else ""
            ),
        )


def p286_query(cutoff: str, limit: int) -> str:
    """Fast first-stage query for all post-cutoff P286 assignments."""
    cutoff_value = _parse_time(cutoff).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""SELECT ?team ?teamArticle ?coach ?coachArticle ?start ?end WHERE {{
  ?team p:P286 ?statement .
  ?statement ps:P286 ?coach ; pq:P580 ?start .
  OPTIONAL {{ ?statement pq:P582 ?end . }}
  FILTER(?start > \"{cutoff_value}\"^^xsd:dateTime)
  ?teamArticle schema:about ?team ; schema:isPartOf <https://en.wikipedia.org/> .
  ?coachArticle schema:about ?coach ; schema:isPartOf <https://en.wikipedia.org/> .
}} ORDER BY ?start LIMIT {int(limit)}"""


def _query_bindings(query: str) -> list[dict[str, Any]]:
    response = requests.get(
        WDQS_URL,
        params={"query": query, "format": "json"},
        headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT},
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    bindings = payload.get("results", {}).get("bindings", [])
    if not isinstance(bindings, list):
        raise ValueError("WDQS response has no results.bindings list")
    return bindings


def query_p286_bindings(cutoff: str, limit: int) -> list[dict[str, Any]]:
    return _query_bindings(p286_query(cutoff, limit))


def _qid(url: str) -> str:
    return url.rsplit("/", 1)[-1]


def _cutoff_anchor_query(cutoff: str, coach_qids: list[str]) -> str:
    cutoff_value = _parse_time(cutoff).strftime("%Y-%m-%dT%H:%M:%SZ")
    values = " ".join(f"wd:{qid}" for qid in coach_qids)
    return f"""SELECT ?anchor ?anchorArticle ?coach ?coachArticle WHERE {{
  VALUES ?coach {{ {values} }}
  VALUES ?cutoff {{ \"{cutoff_value}\"^^xsd:dateTime }}
  ?anchor p:P286 ?statement .
  ?statement ps:P286 ?coach .
  OPTIONAL {{ ?statement pq:P580 ?start . }}
  OPTIONAL {{ ?statement pq:P582 ?end . }}
  FILTER((!BOUND(?start) || ?start <= ?cutoff) &&
         (!BOUND(?end) || ?end >= ?cutoff))
  ?anchorArticle schema:about ?anchor ; schema:isPartOf <https://en.wikipedia.org/> .
  ?coachArticle schema:about ?coach ; schema:isPartOf <https://en.wikipedia.org/> .
}}"""


def discover_p286_candidates(cutoff: str, limit: int) -> list[P286Candidate]:
    """Join a bounded post-cutoff timeline with cutoff anchors locally."""
    rows = query_p286_bindings(cutoff, limit)
    assignments: list[dict[str, str]] = []
    for row in rows:
        try:
            assignments.append({
                "team_qid": _qid(_binding(row, "team")),
                "team_title": _article_title(_binding(row, "teamArticle")),
                "coach_qid": _qid(_binding(row, "coach")),
                "coach_title": _article_title(_binding(row, "coachArticle")),
                "start": _date(_parse_time(_binding(row, "start"))),
            })
        except ValueError:
            continue
    by_team: dict[str, list[dict[str, str]]] = {}
    for assignment in assignments:
        by_team.setdefault(assignment["team_qid"], []).append(assignment)
    transitions: list[tuple[dict[str, str], dict[str, str]]] = []
    for timeline in by_team.values():
        ordered = sorted(timeline, key=lambda item: item["start"])
        for current, successor in zip(ordered, ordered[1:]):
            if current["coach_qid"] != successor["coach_qid"]:
                transitions.append((current, successor))

    wanted_coaches = sorted({current["coach_qid"] for current, _ in transitions})
    anchors_by_coach: dict[str, list[dict[str, str]]] = {}
    for start in range(0, len(wanted_coaches), 100):
        chunk = wanted_coaches[start:start + 100]
        for row in _query_bindings(_cutoff_anchor_query(cutoff, chunk)):
            try:
                coach_qid = _qid(_binding(row, "coach"))
                anchors_by_coach.setdefault(coach_qid, []).append({
                    "qid": _qid(_binding(row, "anchor")),
                    "title": _article_title(_binding(row, "anchorArticle")),
                })
            except ValueError:
                continue

    candidates: list[P286Candidate] = []
    seen = set()
    for middle, successor in transitions:
        for anchor in anchors_by_coach.get(middle["coach_qid"], []):
            if anchor["qid"] == middle["team_qid"]:
                continue
            candidate = P286Candidate(
                anchor_title=anchor["title"],
                coach_title=middle["coach_title"],
                middle_title=middle["team_title"],
                successor_title=successor["coach_title"],
                middle_start=middle["start"],
                successor_start=successor["start"],
                anchor_qid=anchor["qid"],
                coach_qid=middle["coach_qid"],
                middle_qid=middle["team_qid"],
                successor_qid=successor["coach_qid"],
            )
            key = tuple(value.casefold() for value in candidate.__dict__.values())
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    return candidates


def candidate_popularity(
    candidate: P286Candidate, snapshot: PopularitySnapshot
) -> dict[str, Any]:
    titles = (
        candidate.anchor_title, candidate.coach_title,
        candidate.middle_title, candidate.successor_title,
    )
    by_entity = {title: snapshot.views(title) for title in titles}
    return {
        "month": snapshot.month,
        "score": sum(by_entity.values()),
        "max_entity_views": max(by_entity.values(), default=0),
        "entity_views": by_entity,
    }


def _is_infrastructure_failure(errors: list[str]) -> bool:
    markers = (
        "Wikimedia API 呼叫失敗", "HTTP 429", "HTTP 5", "timed out",
        "connection", "temporarily unavailable",
    )
    return any(marker.casefold() in error.casefold() for error in errors for marker in markers)


def _has_link(page, target_title: str) -> bool:
    return any(link.target.casefold() == target_title.casefold() for link in page.links)


def _target_alias(page, target_title: str) -> str | None:
    plain = _plain_text(page.content)
    candidates = [target_title]
    candidates.extend(
        link.anchor for link in page.links
        if link.target.casefold() == target_title.casefold()
    )
    present = [value for value in candidates if value and _fold(value) in _fold(plain)]
    return max(present, key=len) if present else None


def _evidence_window(page, target_title: str) -> tuple[str, str] | None:
    plain = _plain_text(page.content)
    alias = _target_alias(page, target_title)
    if alias is None:
        return None
    position = plain.casefold().find(alias.casefold())
    if position < 0:
        return None
    start = max(0, position - 90)
    end = min(len(plain), position + len(alias) + 150)
    return plain[start:end].strip(), alias


def _first_supported_date(
    backend,
    *,
    title: str,
    target_title: str,
    event_date: str,
) -> tuple[str, Any, str, str] | None:
    base = _parse_time(event_date)
    for offset in DATE_OFFSETS:
        as_of = _date(base + timedelta(days=offset))
        try:
            page = backend.fetch_page(title, as_of=as_of)
        except WikipediaError:
            continue
        evidence = _evidence_window(page, target_title)
        if _has_link(page, target_title) and evidence is not None:
            excerpt, alias = evidence
            return as_of, page, excerpt, alias
    return None


def verify_p286_candidate(
    candidate: P286Candidate,
    *,
    model_id: str,
    backend,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Return one generator-compatible seed after strict revision gates."""
    cutoff = get_model_cutoff(model_id).cutoff_date
    errors: list[str] = []
    try:
        anchor_old = backend.fetch_page(candidate.anchor_title, as_of=cutoff)
        coach_old = backend.fetch_page(candidate.coach_title, as_of=cutoff)
    except WikipediaError as exc:
        return None, [f"cutoff fetch failed: {exc}"]

    anchor_evidence = _evidence_window(anchor_old, candidate.coach_title)
    if not _has_link(anchor_old, candidate.coach_title) or anchor_evidence is None:
        errors.append("cutoff anchor does not visibly link the coach")
    if _has_link(coach_old, candidate.middle_title):
        errors.append("middle team hyperlink already exists on coach cutoff revision")
    if _fold(candidate.middle_title) in _fold(coach_old.content):
        errors.append("middle team name already appears on coach cutoff revision")

    middle_support = _first_supported_date(
        backend, title=candidate.coach_title, target_title=candidate.middle_title,
        event_date=candidate.middle_start,
    )
    if middle_support is None:
        errors.append("no supported coach-to-middle-team revision in +0/+1/+3/+7 window")
    successor_support = _first_supported_date(
        backend, title=candidate.middle_title, target_title=candidate.successor_title,
        event_date=candidate.successor_start,
    )
    if successor_support is None:
        errors.append("no supported middle-team-to-successor revision in +0/+1/+3/+7 window")
    if errors or anchor_evidence is None or middle_support is None or successor_support is None:
        return None, errors

    middle_as_of, _, middle_evidence, middle_alias = middle_support
    successor_as_of, _, successor_evidence, successor_alias = successor_support
    try:
        middle_old = backend.fetch_page(candidate.middle_title, as_of=middle_as_of)
    except WikipediaError as exc:
        return None, [f"middle-team baseline fetch failed: {exc}"]
    if _has_link(middle_old, candidate.successor_title):
        errors.append("successor hyperlink already exists before second temporal switch")
    if _fold(candidate.successor_title) in _fold(middle_old.content):
        errors.append("successor name already appears before second temporal switch")
    if _parse_time(successor_as_of) <= _parse_time(middle_as_of):
        errors.append("second supported revision is not later than the first")
    if errors:
        return None, errors

    anchor_excerpt, coach_alias = anchor_evidence
    seed = {
        "id": (
            "p286_" + "_".join(
                value.casefold().replace(" ", "_").replace(".", "")
                for value in (
                    candidate.anchor_title, candidate.coach_title,
                    candidate.middle_title, candidate.successor_title,
                )
            )
        )[:180],
        "model_id": model_id,
        "target_as_of": successor_as_of,
        "anchor_label": candidate.anchor_title,
        "answer_kind": "Who",
        "category": "sports",
        "old_answer_keywords": [coach_alias],
        "hops": [
            {
                "source_title": candidate.anchor_title,
                "target_title": candidate.coach_title,
                "relation": "head coach at registered knowledge cutoff",
                "relative_clause": (
                    "the person who was head coach of {source} at the tested model's "
                    "registered knowledge cutoff"
                ),
                "as_of": cutoff,
                "evidence": anchor_excerpt,
                "target_aliases": [coach_alias],
            },
            {
                "source_title": candidate.coach_title,
                "target_title": candidate.middle_title,
                "relation": "later appointed head coach of football team",
                "relative_clause": "the football team that later appointed {source} as head coach",
                "as_of": middle_as_of,
                "evidence": middle_evidence,
                "target_aliases": [middle_alias],
            },
            {
                "source_title": candidate.middle_title,
                "target_title": candidate.successor_title,
                "relation": "head coach at target snapshot",
                "relative_clause": "the head coach of {source}",
                "as_of": successor_as_of,
                "evidence": successor_evidence,
                "target_aliases": [successor_alias],
            },
        ],
    }
    try:
        parsed = MultiHopSeed.from_dict(seed)
    except ValueError as exc:
        return None, [f"generated seed contract failed: {exc}"]
    chain_errors, _ = validate_chain(parsed, backend)
    if chain_errors:
        return None, chain_errors
    return seed, []


def _load_bindings(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    bindings = payload.get("results", {}).get("bindings", [])
    if not isinstance(bindings, list):
        raise ValueError("bindings JSON must be a list or WDQS results object")
    return bindings


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Discover popularity-ranked temporal P286 spines and diverse Wikidata "
            "attribute-tail question seeds"
        )
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--query-limit", type=int, default=5000)
    parser.add_argument("--max-accepted", type=int, default=20)
    parser.add_argument(
        "--popular-prefilter", type=int, default=100,
        help="verify only the highest-pageview temporal spine candidates",
    )
    parser.add_argument(
        "--popularity-month",
        help="immutable Wikimedia top-pageviews month in YYYY-MM; default is last complete month",
    )
    parser.add_argument(
        "--no-popularity-ranking", action="store_true",
        help="disable official Wikimedia pageview ranking",
    )
    parser.add_argument(
        "--no-attribute-tails", action="store_true",
        help="emit the legacy P286 spine answers instead of diverse person attributes",
    )
    parser.add_argument(
        "--allow-easy-tails", action="store_true",
        help="allow high-prior citizenship, birthplace, and native-language answers",
    )
    parser.add_argument(
        "--tails-per-entity", type=int, default=2,
        help="maximum selected attribute-tail variants for one discovered person",
    )
    parser.add_argument(
        "--max-per-relation-family", type=int, default=4,
        help="hard batch cap for education/geography/career/etc.",
    )
    parser.add_argument(
        "--max-per-property", type=int, default=1,
        help="hard batch cap for one Wikidata property ID",
    )
    parser.add_argument(
        "--candidate-contains",
        help="optional case-insensitive substring filter over the four entity titles",
    )
    parser.add_argument("--bindings-json", help="reuse a saved WDQS response instead of querying")
    parser.add_argument("--cache-path", default="wikipedia_candidate_discovery.db")
    parser.add_argument("--lang", default="en")
    parser.add_argument(
        "--request-interval", type=float, default=0.75,
        help="minimum seconds between Wikimedia API calls",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", default="discovered_multihop_seeds.json")
    parser.add_argument("--packets-output", default="candidate_discovery_packets.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.query_limit <= 0 or args.max_accepted <= 0 or args.popular_prefilter <= 0:
        parser.error("--query-limit, --max-accepted, and --popular-prefilter must be > 0")
    if (
        args.tails_per_entity <= 0 or args.max_per_relation_family <= 0
        or args.max_per_property <= 0
    ):
        parser.error(
            "--tails-per-entity, --max-per-relation-family, and --max-per-property "
            "must be > 0"
        )
    if args.request_interval < 0:
        parser.error("--request-interval must be >= 0")
    for path in (args.output, args.packets_output):
        assert_new_output_path(path)
        if Path(path).exists() and not args.overwrite:
            parser.error(f"refusing to overwrite {path}; pass --overwrite explicitly")

    cutoff = get_model_cutoff(args.model_id).cutoff_date
    try:
        if args.bindings_json:
            candidates = []
            seen = set()
            for row in _load_bindings(args.bindings_json):
                try:
                    candidate = P286Candidate.from_binding(row)
                except (KeyError, ValueError) as exc:
                    print(f"[binding reject] {exc}")
                    continue
                key = (
                    candidate.anchor_title.casefold(), candidate.coach_title.casefold(),
                    candidate.middle_title.casefold(), candidate.successor_title.casefold(),
                )
                if key not in seen:
                    seen.add(key)
                    candidates.append(candidate)
        else:
            candidates = discover_p286_candidates(cutoff, args.query_limit)
    except (OSError, ValueError, requests.RequestException) as exc:
        print(f"[error] candidate discovery failed: {exc}", file=sys.stderr)
        return 2
    if args.candidate_contains:
        needle = args.candidate_contains.casefold()
        candidates = [
            candidate for candidate in candidates
            if any(needle in value.casefold() for value in (
                candidate.anchor_title, candidate.coach_title,
                candidate.middle_title, candidate.successor_title,
            ))
        ]

    popularity = None
    if not args.no_popularity_ranking:
        try:
            popularity = fetch_top_pageviews(
                lang=args.lang, month=args.popularity_month,
            )
        except (OSError, ValueError, requests.RequestException) as exc:
            print(f"[error] popularity snapshot unavailable: {exc}", file=sys.stderr)
            return 2
        candidates.sort(key=lambda candidate: (
            -candidate_popularity(candidate, popularity)["score"],
            -candidate_popularity(candidate, popularity)["max_entity_views"],
            candidate.anchor_title.casefold(),
        ))
        candidates = candidates[:args.popular_prefilter]

    backend = WikipediaPageBackend(
        cache_path=args.cache_path, lang=args.lang, offline_only=args.offline,
        min_request_interval=args.request_interval,
    )
    accepted: list[dict[str, Any]] = []
    tail_pool: list[dict[str, Any]] = []
    packets: list[dict[str, Any]] = []
    quota_target = attribute_quota_capacity(
        args.max_accepted,
        max_per_family=args.max_per_relation_family,
        max_per_property=args.max_per_property,
    )
    packet_stream = Path(args.packets_output).open("w", encoding="utf-8")
    try:
        for candidate in candidates:
            seed, errors = verify_p286_candidate(
                candidate, model_id=args.model_id, backend=backend
            )
            status = (
                "deterministic_pass" if seed else
                "infrastructure_error" if _is_infrastructure_failure(errors) else
                "deterministic_reject"
            )
            popularity_record = (
                candidate_popularity(candidate, popularity) if popularity else None
            )
            packet = {
                "status": status,
                "candidate": candidate.__dict__,
                "popularity": popularity_record,
                "errors": errors,
                "seed_id": seed.get("id") if seed else None,
            }
            print(
                f"[{status}] {candidate.anchor_title} -> {candidate.coach_title} -> "
                f"{candidate.middle_title} -> {candidate.successor_title} "
                f"views={popularity_record['score'] if popularity_record else 'n/a'}",
                flush=True,
            )
            if seed:
                if args.no_attribute_tails:
                    accepted.append(seed)
                else:
                    variants, tail_packets = generate_attribute_tail_variants(
                        seed, backend=backend, popularity=popularity,
                        relations=tuple(
                            spec for spec in PERSON_RELATIONS
                            if args.allow_easy_tails or spec.guessability != "easy"
                        ),
                    )
                    for variant in variants:
                        metadata = variant.setdefault("selection_metadata", {})
                        metadata["spine_popularity"] = popularity_record
                    packet["attribute_tails"] = tail_packets
                    tail_pool.extend(variants)
            packets.append(packet)
            packet_stream.write(json.dumps(packet, ensure_ascii=False) + "\n")
            packet_stream.flush()
            if args.no_attribute_tails and len(accepted) >= args.max_accepted:
                break
            if not args.no_attribute_tails:
                provisional = select_diverse_variants(
                    tail_pool, max_total=args.max_accepted,
                    max_per_family=args.max_per_relation_family,
                    max_per_source=args.tails_per_entity,
                    max_per_property=args.max_per_property,
                )
                if len(provisional) >= quota_target:
                    print(
                        f"[early stop] diversity quota filled after "
                        f"{len(packets)} candidates ({quota_target} selectable tails)",
                        flush=True,
                    )
                    break
    finally:
        packet_stream.close()
        backend.close()

    if not args.no_attribute_tails:
        accepted = select_diverse_variants(
            tail_pool, max_total=args.max_accepted,
            max_per_family=args.max_per_relation_family,
            max_per_source=args.tails_per_entity,
            max_per_property=args.max_per_property,
        )

    Path(args.output).write_text(
        json.dumps({"seeds": accepted}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    families: dict[str, int] = {}
    for seed in accepted:
        family = str(seed.get("selection_metadata", {}).get("relation_family", "spine"))
        families[family] = families.get(family, 0) + 1
    print(
        f"[done] accepted {len(accepted)}/{len(packets)} checked candidates; "
        f"families={families}",
        flush=True,
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
