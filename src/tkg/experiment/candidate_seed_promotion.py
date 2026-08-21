"""Promote provisional KG candidates into auditable Wikipedia v6 seed inputs.

This module deliberately does not make a candidate formal.  It adds exhaustive
event-order certificates and exact historical Wikipedia evidence; the existing
``tkg-generate-multihop`` validator and independent whole-chain judge remain the
authority that may turn a promoted seed into a case.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

from tkg.experiment.candidate_question_batch import WDQS_URL, _request_json
from tkg.experiment.event_order import (
    build_event_order_certificate,
    event_order_certificate_errors,
)
from tkg.experiment.results import assert_new_output_path
from tkg.experiment.renewable_question_engine import (
    EdgeCandidate, TemporalEdgeContrastJudge, WikipediaSupport,
)
from tkg.experiment.temporal_relation_registry import (
    TemporalRelationSpec, load_temporal_relation_registry,
)
from tkg.wikipedia.backend import WikipediaPageBackend, normalize_title


SCHEMA_VERSION = "candidate-to-v6-seed-promotion-v1"
MAX_ORDER_ROWS = 501

RELATION_EVIDENCE_TERMS: dict[str, tuple[str, ...]] = {
    "P169": ("chief executive", "ceo", "incumbent", "appointed"),
    "P6": ("head of government", "prime minister", "mayor", "premier", "incumbent"),
    "P35": ("head of state", "president", "monarch", "king", "queen", "incumbent"),
    "P1308": (
        "incumbent", "officeholder", "current", "succeeded", "since",
        "serving as", "president",
    ),
}


@dataclass(frozen=True)
class Relation:
    property_id: str
    label: str
    family: str
    relative_clause: str
    later_relative_clause: str
    inverse_relative_clause: str


TAILS: dict[str, tuple[str, str, str]] = {
    "P19": ("place of birth", "the place of birth of {source}", "What"),
    "P20": ("place of death", "the place of death of {source}", "What"),
    "P21": ("sex or gender", "the sex or gender of {source}", "What"),
    "P22": ("father", "the father of {source}", "Who"),
    "P25": ("mother", "the mother of {source}", "Who"),
    "P26": ("spouse", "the spouse of {source}", "Who"),
    "P27": ("country of citizenship", "the country of citizenship of {source}", "What"),
    "P39": ("position held", "a position held by {source}", "What"),
    "P40": ("child", "a child of {source}", "Who"),
    "P54": ("member of sports team", "a sports team represented by {source}", "What"),
    "P69": ("educated at", "the educational institution attended by {source}", "What"),
    "P101": ("field of work", "the field of work of {source}", "What"),
    "P102": ("political party", "the political party of {source}", "What"),
    "P103": ("native language", "the native language of {source}", "What"),
    "P106": ("occupation", "the occupation of {source}", "What"),
    "P108": ("employer", "the employer of {source}", "What"),
    "P140": ("religion or worldview", "the religion or worldview of {source}", "What"),
    "P166": ("award received", "an award received by {source}", "What"),
    "P463": ("member of", "an organization of which {source} is a member", "What"),
    "P551": ("residence", "the residence of {source}", "What"),
    "P734": ("family name", "the family name of {source}", "What"),
    "P735": ("given name", "the given name of {source}", "What"),
    "P800": ("notable work", "a notable work of {source}", "What"),
    "P937": ("work location", "the work location of {source}", "What"),
    "P1412": ("languages spoken", "a language spoken by {source}", "What"),
}

TAIL_FAMILIES = {
    "P19": "geography", "P20": "geography", "P21": "identity",
    "P22": "family", "P25": "family", "P26": "family",
    "P27": "citizenship", "P69": "education", "P101": "career",
    "P39": "career", "P40": "family", "P54": "sports",
    "P102": "affiliation", "P463": "affiliation", "P800": "works",
    "P103": "language", "P106": "career", "P108": "career",
    "P140": "religion", "P166": "award", "P551": "geography",
    "P734": "name", "P735": "name", "P937": "career",
    "P1412": "language",
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _binding(row: dict[str, Any], key: str) -> str:
    value = row.get(key, {})
    return str(value.get("value", "")) if isinstance(value, dict) else ""


def _qid(value: str) -> str:
    match = re.search(r"Q[1-9]\d*$", value)
    return match.group(0) if match else ""


def _day(value: str) -> str:
    return value[:10]


def event_order_query(
    *, property_id: str, source_qid: str, direction: str,
    boundary: str, coverage_end: str, limit: int = MAX_ORDER_ROWS,
) -> str:
    lower = boundary + "T23:59:59Z"
    upper = coverage_end + "T23:59:59Z"
    if direction == "inverse":
        body = (
            f"?target p:{property_id} ?statement .\n"
            f"  ?statement ps:{property_id} wd:{source_qid} ; pq:P580 ?start ."
        )
    elif direction == "forward":
        body = (
            f"wd:{source_qid} p:{property_id} ?statement .\n"
            f"  ?statement ps:{property_id} ?target ; pq:P580 ?start ."
        )
    else:
        raise ValueError(f"unsupported direction: {direction}")
    return f"""SELECT ?target ?start WHERE {{
  {body}
  FILTER(?start > \"{lower}\"^^xsd:dateTime &&
         ?start <= \"{upper}\"^^xsd:dateTime)
}} ORDER BY ?start ?target LIMIT {int(limit)}"""


def normalize_order_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    result = {
        (_day(_binding(row, "start")), _qid(_binding(row, "target")))
        for row in rows
    }
    return [
        {"event_date": event_date, "target_qid": target_qid}
        for event_date, target_qid in sorted(result)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", event_date) and target_qid
    ]


def fetch_order_certificate(
    *, property_id: str, source_qid: str, selected_qid: str,
    selected_date: str, direction: str, boundary: str, coverage_end: str,
    request_interval: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    query = event_order_query(
        property_id=property_id, source_qid=source_qid, direction=direction,
        boundary=boundary, coverage_end=coverage_end,
    )
    payload = _request_json(
        WDQS_URL, params={"query": query, "format": "json"}, timeout=90,
        retries=2, request_interval=request_interval,
    )
    raw = payload.get("results", {}).get("bindings", [])
    rows = normalize_order_rows(raw if isinstance(raw, list) else [])
    audit = {
        "query": query, "query_sha256": _sha256(query), "raw_row_count": len(raw),
        "normalized_rows": rows, "limit": MAX_ORDER_ROWS,
        "complete": len(raw) < MAX_ORDER_ROWS,
    }
    if not audit["complete"]:
        audit["errors"] = ["event-order query saturated its hard limit"]
        return None, audit
    certificate = build_event_order_certificate(
        boundary_event_date=boundary, selected_event_date=selected_date,
        selected_target_qid=selected_qid, candidate_events=rows,
        coverage_end=coverage_end, source="targeted_wdqs_event_order_query",
        source_query_sha256=str(audit["query_sha256"]), complete=True,
    )
    errors = event_order_certificate_errors(
        certificate, expected_boundary=boundary,
        expected_event_date=selected_date, expected_target_qid=selected_qid,
    )
    audit["errors"] = errors
    return (certificate if not errors else None), audit


def evidence_block(
    page: Any, target_title: str, *, preferred_terms: Iterable[str] = (),
) -> tuple[str | None, list[str]]:
    requested = normalize_title(target_title).casefold()
    matching = [link for link in page.links if link.target.casefold() == requested]
    aliases = sorted({link.anchor for link in matching if link.anchor.strip()})
    if not matching:
        return None, []
    terms = tuple(term.casefold() for term in preferred_terms if term)
    candidates: list[tuple[int, int, int, str, str]] = []
    for block in page.content.split("\n\n"):
        for alias in aliases:
            if alias.casefold() in block.casefold():
                marked = f"[{alias} -> {matching[0].target}]".casefold() in block.casefold()
                semantic_hits = sum(term in block.casefold() for term in terms)
                navigation_penalty = 20 if block.lstrip().casefold().startswith("vte") else 0
                length_penalty = min(len(block) // 1000, 10)
                semantic_utility = semantic_hits * 10 - navigation_penalty - length_penalty
                candidates.append(
                    (-semantic_utility, 0 if marked else 1, len(block), block, alias)
                )
    if not candidates:
        return None, aliases
    _, _, _, block, alias = min(candidates)
    return block, [alias]


def _visible_target_excerpt(
    page: Any, target_title: str, aliases: list[str] | None = None,
) -> str | None:
    block, _ = evidence_block(page, target_title)
    if block is not None:
        return block
    folded = page.content.casefold()
    for visible_name in dict.fromkeys([target_title, *(aliases or [])]):
        position = folded.find(visible_name.casefold())
        if position >= 0:
            return page.content[
                max(0, position - 160):position + len(visible_name) + 280
            ].strip()
    return None


def attach_prior_relation_contrasts(
    hops: list[dict[str, Any]], backend: WikipediaPageBackend,
    spec: TemporalRelationSpec, judge: TemporalEdgeContrastJudge,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Judge links already visible before a later relation became true."""
    audits = []
    for index in range(1, len(hops) - 1):
        hop = hops[index]
        structured = hop["structured_evidence"]
        prior = backend.fetch_page(hop["source_title"], as_of=hops[index - 1]["as_of"])
        before_evidence = _visible_target_excerpt(
            prior, hop["target_title"], list(hop["target_aliases"]),
        )
        if before_evidence is None:
            continue
        after = backend.fetch_page(hop["source_title"], as_of=hop["as_of"])
        target = backend.fetch_page(hop["target_title"], as_of=hop["as_of"])
        support = WikipediaSupport(
            as_of=hop["as_of"], evidence=hop["evidence"],
            alias=hop["target_aliases"][0], source_revision_id=after.revision_id,
            target_revision_id=target.revision_id, prior_target_visible=True,
            prior_evidence=before_evidence, prior_revision_id=prior.revision_id,
        )
        candidate = EdgeCandidate(
            spec=spec, direction=structured["direction"],
            next_qid=(
                structured["kg_subject_qid"]
                if structured["direction"] == "inverse"
                else structured["kg_object_qid"]
            ),
            next_title=hop["target_title"], event_date=structured["event_date"],
            qualifier_dates={"P580": structured["event_date"]},
            kg_subject_qid=structured["kg_subject_qid"],
            kg_object_qid=structured["kg_object_qid"],
            event_order_certificate=structured["event_order_certificate"],
        )
        result = judge.judge(
            spec, candidate, support, source_title=hop["source_title"],
            target_title=hop["target_title"], previous_as_of=hops[index - 1]["as_of"],
        )
        audit = {"hop_index": index, **result}
        audits.append(audit)
        if result.get("decision") != "pass":
            raise ValueError(
                f"hop {index} prior relation contrast rejected: {result.get('reason')}"
            )
        raw_response = str(result.get("raw_response", ""))
        structured["prior_relation_contrast"] = {
            "method": "independent_llm_semantic_contrast", "decision": "pass",
            "confidence": result["confidence"], "judge_model": judge.model,
            "property_id": spec.property_id, "direction": structured["direction"],
            "before_revision_id": prior.revision_id,
            "after_revision_id": after.revision_id,
            "before_evidence": before_evidence, "reason": result.get("reason"),
            "raw_response_sha256": _sha256(raw_response),
        }
    return hops, audits


def _observation_dates(cutoff: str, event_1: str, event_2: str, target: str) -> tuple[str, str]:
    # Observe the first bridge immediately before the second world event where
    # possible; observe the successor at the fixed target snapshot.
    before_second = (date.fromisoformat(event_2) - timedelta(days=1)).isoformat()
    first = max(event_1, before_second)
    if not cutoff < first < target:
        first = event_1
    return first, target


def _relations(path: Path) -> dict[str, Relation]:
    payload = json.loads(path.read_text())
    result = {}
    for row in payload.get("relations", []):
        if row.get("status") != "active":
            continue
        result[row["property_id"]] = Relation(
            property_id=row["property_id"], label=row["label"], family=row["family"],
            relative_clause=row["relative_clause"],
            later_relative_clause=row["later_relative_clause"],
            inverse_relative_clause=row["inverse_relative_clause"],
        )
    return result


def candidate_priority(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(candidate.get("relation_property")), str(candidate.get("tail_property")),
        str(candidate.get("public_anchor")), str(candidate.get("id")),
    )


def select_candidates(
    candidates: Iterable[dict[str, Any]], active: set[str], limit: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(candidates, key=candidate_priority):
        prop = str(row.get("relation_property"))
        if (
            row.get("topology_id") == f"same-relation-forward-{prop}"
            and prop in active and row.get("tail_property") in TAILS
        ):
            groups.setdefault(prop, []).append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(groups.values()):
        for prop in sorted(groups):
            if groups[prop] and len(selected) < limit:
                selected.append(groups[prop].pop(0))
    return selected


def _claim_qids(entity: dict[str, Any], property_id: str) -> list[str]:
    result = []
    for claim in entity.get("claims", {}).get(property_id, []):
        if not isinstance(claim, dict) or claim.get("rank") == "deprecated":
            continue
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
        qid = value.get("id") if isinstance(value, dict) else None
        if isinstance(qid, str) and re.fullmatch(r"Q[1-9]\d*", qid):
            result.append(qid)
    return list(dict.fromkeys(result))


def renew_tail_from_wikipedia(
    candidate: dict[str, Any], backend: WikipediaPageBackend,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Replace an unlinked KG tail with a claim that is linked in the revision."""
    renewed = copy.deepcopy(candidate)
    hop = renewed["private_chain"][-1]
    page = backend.fetch_page(hop["source_title"], as_of=renewed["target_as_of"])
    linked = {link.target.casefold(): link.target for link in page.links}
    if normalize_title(hop["target_title"]).casefold() in linked:
        return renewed, None
    subject_qid = str(hop["source_qid"])
    entity = backend.get_wikidata_entities([subject_qid])[subject_qid]
    targets: list[tuple[str, str]] = []
    for property_id in sorted(TAILS):
        targets.extend((property_id, qid) for qid in _claim_qids(entity, property_id))
    target_entities = backend.get_wikidata_entities(
        [qid for _, qid in targets], props="labels|sitelinks",
    )
    alternatives = []
    for property_id, qid in targets:
        target = target_entities.get(qid, {})
        title = target.get("sitelinks", {}).get("enwiki", {}).get("title")
        if not isinstance(title, str) or title.casefold() not in linked:
            continue
        alternatives.append((property_id, qid, linked[title.casefold()]))
    if not alternatives:
        return renewed, None
    property_id, target_qid, target_title = alternatives[0]
    old = {
        "property_id": hop["property_id"], "target_qid": hop["target_qid"],
        "target_title": hop["target_title"],
    }
    hop.update({
        "property_id": property_id, "target_qid": target_qid,
        "target_title": target_title,
    })
    renewed["tail_property"] = property_id
    renewed["tail_family"] = TAIL_FAMILIES[property_id]
    return renewed, {
        "method": "wikidata_claim_intersect_revision_hyperlink",
        "source_revision_id": page.revision_id, "old_tail": old,
        "new_tail": {"property_id": property_id, "target_qid": target_qid,
                     "target_title": target_title},
        "eligible_alternative_count": len(alternatives),
    }


def promote_candidate(
    candidate: dict[str, Any], relation: Relation, backend: WikipediaPageBackend,
    *, request_interval: float, contrast_spec: TemporalRelationSpec | None = None,
    contrast_judge: TemporalEdgeContrastJudge | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    chain = candidate["private_chain"]
    cutoff, target = candidate["knowledge_cutoff"], candidate["target_as_of"]
    event_1, event_2 = chain[1]["event_date"], chain[2]["event_date"]
    packet: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "candidate_id": candidate["id"],
        "status": "rejected", "errors": [], "event_order": [], "snapshots": [],
    }
    cert1, audit1 = fetch_order_certificate(
        property_id=relation.property_id, source_qid=chain[0]["target_qid"],
        selected_qid=chain[1]["target_qid"], selected_date=event_1,
        direction="inverse", boundary=cutoff, coverage_end=target,
        request_interval=request_interval,
    )
    cert2, audit2 = fetch_order_certificate(
        property_id=relation.property_id, source_qid=chain[1]["target_qid"],
        selected_qid=chain[2]["target_qid"], selected_date=event_2,
        direction="forward", boundary=event_1, coverage_end=target,
        request_interval=request_interval,
    )
    packet["event_order"] = [audit1, audit2]
    if cert1 is None or cert2 is None:
        packet["errors"].extend(audit1.get("errors", []) + audit2.get("errors", []))
        return None, packet

    observation_1, observation_2 = _observation_dates(cutoff, event_1, event_2, target)
    as_ofs = [cutoff, observation_1, observation_2, target]
    evidence: list[tuple[str, list[str]]] = []
    try:
        for hop, as_of in zip(chain, as_ofs):
            page = backend.fetch_page(hop["source_title"], as_of=as_of)
            property_id = str(hop["property_id"])
            tail_terms = TAILS.get(property_id, ("", "", ""))[0].split()
            block, aliases = evidence_block(
                page, hop["target_title"],
                preferred_terms=(*RELATION_EVIDENCE_TERMS.get(property_id, ()), *tail_terms),
            )
            packet["snapshots"].append({
                "source_title": page.title, "revision_id": page.revision_id,
                "revision_date": page.timestamp, "as_of": as_of,
                "target_title": hop["target_title"], "evidence_found": block is not None,
            })
            if block is None:
                packet["errors"].append(
                    f"no revision-present hyperlink evidence: {hop['source_title']} -> "
                    f"{hop['target_title']} at {as_of}"
                )
            else:
                evidence.append((block, aliases))
    except Exception as exc:
        packet["errors"].append(f"Wikipedia infrastructure error: {exc}")
    if packet["errors"] or len(evidence) != 4:
        return None, packet

    tail_label, tail_clause, answer_kind = TAILS[chain[3]["property_id"]]
    temporal_common = {
        "source": "wikidata_time_qualified_statement",
        "property_id": relation.property_id,
        "temporal_operator": "next_after_boundary",
    }
    hops = [
        {
            "source_title": chain[0]["source_title"], "target_title": chain[0]["target_title"],
            "relation": relation.label,
            "relative_clause": relation.relative_clause.replace(
                "{source}", "{source} at the tested model's registered knowledge cutoff"
            ),
            "as_of": cutoff, "evidence": evidence[0][0],
            "target_aliases": evidence[0][1], "incoming_time_policy": "advance_required",
            "property_id": relation.property_id, "relation_family": relation.family,
            "structured_evidence": {"source": "wikipedia_revision_hyperlink"},
        },
        {
            "source_title": chain[1]["source_title"], "target_title": chain[1]["target_title"],
            "relation": relation.label,
            "relative_clause": relation.inverse_relative_clause,
            "as_of": observation_1, "evidence": evidence[1][0],
            "target_aliases": evidence[1][1], "incoming_time_policy": "advance_required",
            "property_id": relation.property_id, "relation_family": relation.family,
            "structured_evidence": {
                **temporal_common, "direction": "inverse", "event_date": event_1,
                "kg_subject_qid": chain[1]["target_qid"],
                "kg_object_qid": chain[1]["source_qid"],
                "event_order_certificate": cert1,
            },
        },
        {
            "source_title": chain[2]["source_title"], "target_title": chain[2]["target_title"],
            "relation": relation.label, "relative_clause": relation.later_relative_clause,
            "as_of": observation_2, "evidence": evidence[2][0],
            "target_aliases": evidence[2][1], "incoming_time_policy": "advance_required",
            "property_id": relation.property_id, "relation_family": relation.family,
            "structured_evidence": {
                **temporal_common, "direction": "forward", "event_date": event_2,
                "kg_subject_qid": chain[2]["source_qid"],
                "kg_object_qid": chain[2]["target_qid"],
                "event_order_certificate": cert2,
            },
        },
        {
            "source_title": chain[3]["source_title"], "target_title": chain[3]["target_title"],
            "relation": tail_label, "relative_clause": tail_clause,
            "as_of": target, "evidence": evidence[3][0],
            "target_aliases": evidence[3][1], "incoming_time_policy": "same_snapshot",
            "property_id": chain[3]["property_id"], "relation_family": candidate["tail_family"],
            "structured_evidence": {
                "source": "wikidata_statement", "direction": "forward_attribute",
                "kg_subject_qid": chain[3]["source_qid"],
                "kg_object_qid": chain[3]["target_qid"],
            },
        },
    ]
    if contrast_judge is not None:
        if contrast_spec is None:
            raise ValueError("edge contrast judge requires a relation specification")
        try:
            hops, contrast_audits = attach_prior_relation_contrasts(
                hops, backend, contrast_spec, contrast_judge,
            )
            packet["prior_relation_contrasts"] = contrast_audits
        except ValueError as exc:
            packet["errors"].append(str(exc))
            return None, packet
    suffix = ""
    if candidate.get("_tail_renewal"):
        suffix = "_" + str(chain[3]["property_id"]).lower()
    seed = {
        "id": "promoted_" + candidate["id"].removeprefix("kgcand_") + suffix,
        "model_id": candidate["_model_id"], "cutoff_date": cutoff,
        "target_as_of": target, "anchor_label": candidate["public_anchor"],
        "answer_kind": answer_kind, "category": candidate["domain_family"],
        "old_answer_keywords": [], "hops": hops,
        "selection_metadata": {
            "source_candidate_id": candidate["id"],
            "promotion_schema": SCHEMA_VERSION,
            "private_chain_inference_visibility": "forbidden",
            "tail_renewal": candidate.get("_tail_renewal"),
        },
    }
    packet["status"] = "promoted_pending_v6_validation"
    return seed, packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--model-id", default="openai/gpt-4.1-mini")
    parser.add_argument(
        "--candidate-id", action="append", default=[],
        help="promote only the named provisional candidate; repeat as needed",
    )
    parser.add_argument(
        "--exclude-packets", action="append", default=[],
        help=(
            "JSONL promotion packet ledger whose candidate_ids must not be retried; "
            "repeat to combine prior batches"
        ),
    )
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument(
        "--selection-offset", type=int, default=0,
        help="skip this many candidates in deterministic promotion order",
    )
    parser.add_argument(
        "--renew-unlinked-tail", action="store_true",
        help="intersect final-person Wikidata claims with target-revision hyperlinks",
    )
    parser.add_argument("--edge-contrast-judge-model")
    parser.add_argument("--edge-contrast-cache-path")
    parser.add_argument("--edge-contrast-min-confidence", type=float, default=0.8)
    parser.add_argument("--request-interval", type=float, default=0.25)
    parser.add_argument("--api-call-budget", type=int, default=500)
    parser.add_argument("--api-max-retries", type=int, default=6)
    parser.add_argument("--api-backoff-base", type=float, default=2.0)
    parser.add_argument("--api-backoff-max", type=float, default=60.0)
    parser.add_argument("--cache-path", default="wikipedia_snapshot.db")
    parser.add_argument("--output", required=True)
    parser.add_argument("--packets-output", required=True)
    args = parser.parse_args()
    if args.api_call_budget <= 0 or args.api_max_retries <= 0:
        parser.error("--api-call-budget and --api-max-retries must be > 0")
    if args.selection_offset < 0:
        parser.error("--selection-offset must be non-negative")
    if args.api_backoff_base < 0 or args.api_backoff_max < 0:
        parser.error("API backoff values must be >= 0")
    output, packets_output = Path(args.output), Path(args.packets_output)
    assert_new_output_path(str(output))
    assert_new_output_path(str(packets_output))
    relations = _relations(Path(args.registry))
    registry = load_temporal_relation_registry(args.registry)
    contrast_specs = {spec.property_id: spec for spec in registry.relations}
    payload = json.loads(Path(args.candidates).read_text())
    candidates = list(payload.get("questions", []))
    excluded_ids: set[str] = set()
    for ledger_path in args.exclude_packets:
        for line in Path(ledger_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("candidate_id"):
                excluded_ids.add(str(row["candidate_id"]))
    candidates = [
        row for row in candidates if str(row.get("id")) not in excluded_ids
    ]
    if args.candidate_id:
        requested = set(args.candidate_id)
        available = {str(row.get("id")) for row in candidates}
        missing = sorted(requested - available)
        if missing:
            parser.error(f"--candidate-id not found: {', '.join(missing)}")
        candidates = [row for row in candidates if row.get("id") in requested]
    for row in candidates:
        row["_model_id"] = args.model_id
    selected = select_candidates(
        candidates, set(relations), args.selection_offset + args.max_candidates,
    )[args.selection_offset:]
    backend = WikipediaPageBackend(
        cache_path=args.cache_path, min_request_interval=args.request_interval,
        max_api_calls=args.api_call_budget,
        api_max_retries=args.api_max_retries,
        backoff_base_seconds=args.api_backoff_base,
        backoff_max_seconds=args.api_backoff_max,
    )
    contrast_judge = (
        TemporalEdgeContrastJudge(
            args.edge_contrast_judge_model,
            min_confidence=args.edge_contrast_min_confidence,
            cache_path=args.edge_contrast_cache_path,
        )
        if args.edge_contrast_judge_model else None
    )
    seeds, packets = [], []
    try:
        for index, candidate in enumerate(selected, 1):
            api_before = backend.request_stats()
            print(json.dumps({
                "stage": "promotion", "index": index, "total": len(selected),
                "candidate_id": candidate["id"], "property_id": candidate["relation_property"],
            }), flush=True)
            try:
                renewal = None
                if args.renew_unlinked_tail:
                    candidate, renewal = renew_tail_from_wikipedia(candidate, backend)
                    if renewal is not None:
                        candidate["_tail_renewal"] = renewal
                seed, packet = promote_candidate(
                    candidate, relations[candidate["relation_property"]], backend,
                    request_interval=args.request_interval,
                    contrast_spec=contrast_specs[candidate["relation_property"]],
                    contrast_judge=contrast_judge,
                )
                if renewal is not None:
                    packet["tail_renewal"] = renewal
            except Exception as exc:
                seed, packet = None, {
                    "schema_version": SCHEMA_VERSION, "candidate_id": candidate["id"],
                    "status": "infrastructure_error", "errors": [str(exc)],
                }
            packet["wikipedia_api_usage"] = {
                "calls_this_candidate": (
                    int(backend.request_stats()["api_calls_used"] or 0)
                    - int(api_before["api_calls_used"] or 0)
                ),
                **backend.request_stats(),
                "cache_path": args.cache_path,
            }
            packets.append(packet)
            if seed is not None:
                seeds.append(seed)
    finally:
        if contrast_judge is not None:
            contrast_judge.close()
        backend.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema_version": "multihop-seed-batch-v1", "seeds": seeds,
        "provenance": {"promotion_schema": SCHEMA_VERSION,
                       "selected_candidates": len(selected), "promoted": len(seeds),
                       "selection_offset": args.selection_offset,
                       "excluded_prior_candidate_count": len(excluded_ids),
                       "exclude_packet_ledgers": list(args.exclude_packets)},
    }, ensure_ascii=False, indent=2) + "\n")
    packets_output.write_text("".join(
        json.dumps(packet, ensure_ascii=False) + "\n" for packet in packets
    ))
    print(json.dumps({"selected": len(selected), "promoted": len(seeds),
                      "rejected": len(selected) - len(seeds), "output": str(output)}))


if __name__ == "__main__":
    main()
