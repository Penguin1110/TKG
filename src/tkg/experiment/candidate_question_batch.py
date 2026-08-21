"""Generate a large, explicitly provisional temporal-question candidate batch.

This is the cheap first stage of the benchmark pipeline. Wikidata qualified
statements propose fixed entity/event chains; it does not claim Wikipedia support,
prior-knowledge admission, or experimental validity. Those expensive gates run later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

from tkg.experiment.candidate_topology_registry import (
    CandidateTopologyRegistry, load_candidate_topology_registry,
)
from tkg.experiment.model_cutoffs import get_model_cutoff
from tkg.experiment.renewable_question_engine import load_relation_admissions
from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_relation_registry import (
    TemporalRelationRegistry, TemporalRelationSpec, load_temporal_relation_registry,
)


BATCH_SCHEMA = "kg-temporal-question-candidate-batch-v1"
QUESTION_SCHEMA = "kg-temporal-question-candidate-v1"
QUESTION_TEMPLATE_VERSION = "deterministic-topology-template-v3"
WDQS_URL = "https://query.wikidata.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "TKG-candidate-question-batch/0.1 (research)"
GENERIC_OFFICE_PATTERNS = (
    r"^member of(?: the)?\b", r"^member\b", r"^judge$", r"^president$",
    r"^chairperson$", r"^spokesperson$", r"^bishop$", r"^minister$",
    r"^senator$", r"^representative$", r"^councillor$", r"^deputy$",
)
_DEFAULT_TOPOLOGY_REGISTRY = load_candidate_topology_registry()
LEADERSHIP_PROPERTIES = _DEFAULT_TOPOLOGY_REGISTRY.leadership_relation_ids
TAIL_SPECS = tuple(
    (tail.property_id, tail.family, tail.question)
    for tail in _DEFAULT_TOPOLOGY_REGISTRY.tails
)


@dataclass(frozen=True)
class EventRow:
    subject_qid: str
    value_qid: str
    start: str


@dataclass(frozen=True)
class QueryArtifact:
    property_id: str
    lower_exclusive: str
    upper_inclusive: str
    query: str
    rows: tuple[EventRow, ...]
    limit_saturated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "property_id": self.property_id,
            "lower_exclusive": self.lower_exclusive,
            "upper_inclusive": self.upper_inclusive,
            "query": self.query,
            "query_sha256": hashlib.sha256(self.query.encode()).hexdigest(),
            "row_count": len(self.rows),
            "normalized_rows": [row.__dict__ for row in self.rows],
            "limit_saturated": self.limit_saturated,
        }


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _strict_date(value: str) -> str:
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError(f"date must use YYYY-MM-DD: {value!r}")
    return value


def _qid(value: str) -> str:
    result = value.rsplit("/", 1)[-1]
    if not re.fullmatch(r"Q[1-9]\d*", result):
        raise ValueError(f"invalid Wikidata entity URI: {value!r}")
    return result


def _literal_date(value: str) -> str | None:
    match = re.match(r"^\+?(\d{4})-(\d{2})-(\d{2})T", value)
    if not match or "00" in match.groups()[1:]:
        return None
    try:
        return date.fromisoformat("-".join(match.groups())).isoformat()
    except ValueError:
        return None


def _binding(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    return str(value.get("value", "")) if isinstance(value, dict) else ""


def _event_query(
    property_id: str, lower_exclusive: str, upper_inclusive: str, limit: int,
) -> str:
    return f"""SELECT ?subject ?value ?start WHERE {{
  ?subject p:{property_id} ?statement .
  ?statement ps:{property_id} ?value ; pq:P580 ?start .
  FILTER(?start > \"{lower_exclusive}T23:59:59Z\"^^xsd:dateTime &&
         ?start <= \"{upper_inclusive}T23:59:59Z\"^^xsd:dateTime)
}} LIMIT {int(limit)}"""


def _normalize_event_rows(rows: Iterable[dict[str, Any]]) -> list[EventRow]:
    normalized = []
    for row in rows:
        try:
            subject = _qid(_binding(row, "subject"))
            value = _qid(_binding(row, "value"))
        except ValueError:
            continue
        start = _literal_date(_binding(row, "start"))
        if start and subject != value:
            normalized.append(EventRow(subject, value, start))
    return sorted(
        set(normalized), key=lambda row: (row.start, row.subject_qid, row.value_qid),
    )


def _fetch_event_rows_adaptive(
    property_id: str, *, cutoff: str, until: str, limit: int,
    request_interval: float, window_days: int = 180,
) -> tuple[list[EventRow], list[QueryArtifact], bool]:
    """Fetch bounded windows; saturation stays explicit for the formal order gate."""
    artifacts: list[QueryArtifact] = []
    all_rows: list[EventRow] = []
    unresolved_saturation = False
    lower_date = date.fromisoformat(cutoff)
    final_date = date.fromisoformat(until)
    while lower_date < final_date:
        upper_date = min(lower_date + timedelta(days=window_days), final_date)
        lower, upper = lower_date.isoformat(), upper_date.isoformat()
        query = _event_query(property_id, lower, upper, limit)
        payload = _request_json(
            WDQS_URL, params={"query": query, "format": "json"},
            timeout=60, retries=2, request_interval=request_interval,
        )
        raw = payload.get("results", {}).get("bindings", [])
        if not isinstance(raw, list):
            raise ValueError("WDQS results.bindings is not a list")
        rows = _normalize_event_rows(row for row in raw if isinstance(row, dict))
        saturated = len(raw) >= limit
        artifacts.append(QueryArtifact(
            property_id, lower, upper, query, tuple(rows), saturated,
        ))
        all_rows.extend(rows)
        unresolved_saturation = unresolved_saturation or saturated
        print(json.dumps({
            "stage": "staged_event_window", "property_id": property_id,
            "lower_exclusive": lower, "upper_inclusive": upper,
            "row_count": len(rows), "limit_saturated": saturated,
        }), flush=True)
        lower_date = upper_date
    unique = sorted(
        set(all_rows), key=lambda row: (row.start, row.subject_qid, row.value_qid),
    )
    return unique, artifacts, unresolved_saturation


def _pair_novelty_filters(property_id: str) -> str:
    return f"""
  FILTER NOT EXISTS {{
    ?laterAnchor p:{property_id} ?prior1 .
    ?prior1 ps:{property_id} ?p0 ; pq:P580 ?priorStart1 .
    FILTER(?priorStart1 < ?start1)
  }}
  FILTER NOT EXISTS {{
    ?laterAnchor p:{property_id} ?prior2 .
    ?prior2 ps:{property_id} ?p2 ; pq:P580 ?priorStart2 .
    FILTER(?priorStart2 < ?start2)
  }}"""


def _query(
    spec: TemporalRelationSpec, cutoff: str, until: str, limit: int, *,
    require_pair_novelty: bool = False,
) -> str:
    prop = spec.property_id
    cutoff_start = cutoff + "T00:00:00Z"
    cutoff_end = cutoff + "T23:59:59Z"
    until_end = until + "T23:59:59Z"
    return f"""SELECT ?anchor ?p0 ?start0 ?end0 ?laterAnchor ?start1 ?p2 ?start2 WHERE {{
  ?anchor p:{prop} ?s0 .
  ?s0 ps:{prop} ?p0 ; pq:P580 ?start0 .
  OPTIONAL {{ ?s0 pq:P582 ?end0 . }}
  FILTER(?start0 <= \"{cutoff_end}\"^^xsd:dateTime &&
         (!BOUND(?end0) || ?end0 >= \"{cutoff_start}\"^^xsd:dateTime))

  ?laterAnchor p:{prop} ?s1 .
  ?s1 ps:{prop} ?p0 ; pq:P580 ?start1 .
  FILTER(?start1 > \"{cutoff_end}\"^^xsd:dateTime &&
         ?start1 <= \"{until_end}\"^^xsd:dateTime)

  ?laterAnchor p:{prop} ?s2 .
  ?s2 ps:{prop} ?p2 ; pq:P580 ?start2 .
  FILTER(?start2 > ?start1 && ?start2 <= \"{until_end}\"^^xsd:dateTime &&
         ?p2 != ?p0 && ?anchor != ?laterAnchor)
  {_pair_novelty_filters(prop) if require_pair_novelty else ""}
}} ORDER BY ?start1 ?start2 ?anchor ?laterAnchor LIMIT {int(limit)}"""


def _request_json(
    url: str, *, params: dict[str, Any], timeout: float = 120,
    retries: int = 3, request_interval: float = 0.0,
) -> dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    error: Exception | None = None
    for attempt in range(retries):
        if request_interval:
            time.sleep(request_interval)
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 429 or response.status_code >= 500:
                error = requests.HTTPError(
                    f"HTTP {response.status_code}: {response.text[:200]}",
                    response=response,
                )
                delay = min(30.0, float(response.headers.get("Retry-After", 2 ** attempt)))
                time.sleep(delay)
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("API response is not a JSON object")
            return payload
        except (requests.RequestException, ValueError) as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"API request failed after {retries} attempts: {error}")


def _wdqs_rows(
    spec: TemporalRelationSpec, *, cutoff: str, until: str, limit: int,
    request_interval: float, require_pair_novelty: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    query = _query(
        spec, cutoff, until, limit,
        require_pair_novelty=require_pair_novelty,
    )
    payload = _request_json(
        WDQS_URL, params={"query": query, "format": "json"},
        request_interval=request_interval,
    )
    rows = payload.get("results", {}).get("bindings", [])
    if not isinstance(rows, list):
        raise ValueError("WDQS results.bindings is not a list")
    return query, [row for row in rows if isinstance(row, dict)]


def _event_first_query(
    spec: TemporalRelationSpec, *, cutoff: str, until: str,
    events: list[dict[str, str]], limit: int,
    require_pair_novelty: bool = False,
) -> str:
    """Join a small, pre-profiled event batch to its anchor and successor."""
    prop = spec.property_id
    cutoff_start = cutoff + "T00:00:00Z"
    cutoff_end = cutoff + "T23:59:59Z"
    until_end = until + "T23:59:59Z"
    values = "\n    ".join(
        f'(wd:{row["source_qid"]} wd:{row["target_qid"]} '
        f'"{row["start"]}T00:00:00Z"^^xsd:dateTime)'
        for row in events
    )
    return f"""SELECT ?anchor ?p0 ?start0 ?end0 ?laterAnchor ?start1 ?p2 ?start2 WHERE {{
  VALUES (?laterAnchor ?p0 ?start1) {{
    {values}
  }}
  ?anchor p:{prop} ?s0 .
  ?s0 ps:{prop} ?p0 ; pq:P580 ?start0 .
  OPTIONAL {{ ?s0 pq:P582 ?end0 . }}
  FILTER(?start0 <= "{cutoff_end}"^^xsd:dateTime &&
         (!BOUND(?end0) || ?end0 >= "{cutoff_start}"^^xsd:dateTime))

  ?laterAnchor p:{prop} ?s2 .
  ?s2 ps:{prop} ?p2 ; pq:P580 ?start2 .
  FILTER(?start2 > ?start1 && ?start2 <= "{until_end}"^^xsd:dateTime &&
         ?p2 != ?p0 && ?anchor != ?laterAnchor)
  {_pair_novelty_filters(prop) if require_pair_novelty else ""}
}} ORDER BY ?start1 ?start2 ?anchor ?laterAnchor LIMIT {int(limit)}"""


def _profile_event_seeds(
    profile_paths: list[str], *, property_id: str, cutoff: str, until: str,
) -> list[dict[str, str]]:
    seeds: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in profile_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for profile in payload.get("profiles", []):
            if not isinstance(profile, dict):
                continue
            relation = profile.get("relation", {})
            if not isinstance(relation, dict) or relation.get("property_id") != property_id:
                continue
            for sample in profile.get("samples", []):
                if not isinstance(sample, dict):
                    continue
                source = str(sample.get("source_qid") or "")
                target = str(sample.get("target_qid") or "")
                start = str(sample.get("start") or "")[:10]
                if not (
                    re.fullmatch(r"Q[1-9]\d*", source)
                    and re.fullmatch(r"Q[1-9]\d*", target)
                    and re.fullmatch(r"\d{4}-\d{2}-\d{2}", start)
                    and cutoff < start <= until
                ):
                    continue
                seeds[(source, target, start)] = {
                    "source_qid": source, "target_qid": target, "start": start,
                }
    return sorted(seeds.values(), key=lambda row: (
        row["start"], row["source_qid"], row["target_qid"],
    ))


def _event_first_wdqs_rows(
    spec: TemporalRelationSpec, *, cutoff: str, until: str,
    events: list[dict[str, str]], batch_size: int, limit: int,
    request_interval: float, require_pair_novelty: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    artifacts = []
    for offset in range(0, len(events), batch_size):
        batch = events[offset:offset + batch_size]
        query = _event_first_query(
            spec, cutoff=cutoff, until=until, events=batch, limit=limit,
            require_pair_novelty=require_pair_novelty,
        )
        payload = _request_json(
            WDQS_URL, params={"query": query, "format": "json"},
            request_interval=request_interval,
        )
        raw = payload.get("results", {}).get("bindings", [])
        if not isinstance(raw, list):
            raise ValueError("WDQS results.bindings is not a list")
        valid = [row for row in raw if isinstance(row, dict)]
        rows.extend(valid)
        artifacts.append({
            "query": query,
            "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
            "seed_count": len(batch), "row_count": len(valid),
            "seed_offset": offset,
        })
    return rows, artifacts


def _normalize_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    result = []
    for row in rows:
        try:
            normalized = {
                "anchor_qid": _qid(_binding(row, "anchor")),
                "person_0_qid": _qid(_binding(row, "p0")),
                "later_anchor_qid": _qid(_binding(row, "laterAnchor")),
                "person_2_qid": _qid(_binding(row, "p2")),
                "start_0": str(_literal_date(_binding(row, "start0")) or ""),
                "end_0": str(_literal_date(_binding(row, "end0")) or ""),
                "start_1": str(_literal_date(_binding(row, "start1")) or ""),
                "start_2": str(_literal_date(_binding(row, "start2")) or ""),
            }
        except ValueError:
            continue
        if not all(normalized[key] for key in (
            "anchor_qid", "person_0_qid", "later_anchor_qid",
            "person_2_qid", "start_0", "start_1", "start_2",
        )):
            continue
        if len({
            normalized["anchor_qid"], normalized["person_0_qid"],
            normalized["later_anchor_qid"], normalized["person_2_qid"],
        }) != 4:
            continue
        result.append(normalized)
    unique = {
        tuple(sorted(row.items())): row for row in result
    }
    return sorted(unique.values(), key=lambda row: (
        row["start_1"], row["start_2"], row["anchor_qid"],
        row["later_anchor_qid"], row["person_2_qid"],
    ))


def _select_spines(rows: list[dict[str, str]], max_spines: int) -> list[dict[str, str]]:
    """Choose locally earliest unambiguous rows; exhaustive order stays pending."""
    by_person: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_person[row["person_0_qid"]].append(row)
    selected = []
    seen_anchors: set[str] = set()
    seen_paths: set[tuple[str, ...]] = set()
    for person_0, person_rows in sorted(by_person.items()):
        first_start = min(row["start_1"] for row in person_rows)
        first_rows = [row for row in person_rows if row["start_1"] == first_start]
        later_anchors = {row["later_anchor_qid"] for row in first_rows}
        if len(later_anchors) != 1:
            continue
        later_anchor = next(iter(later_anchors))
        successor_rows = [
            row for row in first_rows if row["later_anchor_qid"] == later_anchor
        ]
        first_successor = min(row["start_2"] for row in successor_rows)
        successor_rows = [
            row for row in successor_rows if row["start_2"] == first_successor
        ]
        successors = {row["person_2_qid"] for row in successor_rows}
        if len(successors) != 1:
            continue
        for row in successor_rows:
            if row["anchor_qid"] in seen_anchors:
                continue
            path = (
                row["anchor_qid"], person_0, later_anchor,
                row["person_2_qid"],
            )
            if path in seen_paths:
                continue
            selected.append(row)
            seen_anchors.add(row["anchor_qid"])
            seen_paths.add(path)
            break
        if len(selected) >= max_spines:
            break
    return selected


def _fetch_entities(
    qids: Iterable[str], *, request_interval: float,
) -> dict[str, dict[str, Any]]:
    ordered = list(dict.fromkeys(qids))
    result: dict[str, dict[str, Any]] = {}
    for index in range(0, len(ordered), 50):
        batch = ordered[index:index + 50]
        payload = _request_json(WIKIDATA_API, params={
            "action": "wbgetentities", "ids": "|".join(batch),
            "props": "claims|labels|sitelinks", "languages": "en", "format": "json",
        }, timeout=60, request_interval=request_interval)
        entities = payload.get("entities", {})
        if isinstance(entities, dict):
            result.update({
                str(qid): entity for qid, entity in entities.items()
                if isinstance(entity, dict)
            })
    return result


def _title(entity: dict[str, Any]) -> str | None:
    link = entity.get("sitelinks", {}).get("enwiki", {})
    value = link.get("title") if isinstance(link, dict) else None
    return " ".join(str(value).split()) if value else None


def _label(entity: dict[str, Any]) -> str | None:
    value = entity.get("labels", {}).get("en", {})
    text = value.get("value") if isinstance(value, dict) else None
    return " ".join(str(text).split()) if text else None


def _entity_name(entity: dict[str, Any]) -> str | None:
    return _title(entity) or _label(entity)


def _claim_targets(entity: dict[str, Any], property_id: str) -> list[str]:
    result = []
    for claim in entity.get("claims", {}).get(property_id, []):
        if not isinstance(claim, dict) or claim.get("rank") == "deprecated":
            continue
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
        qid = value.get("id") if isinstance(value, dict) else None
        if isinstance(qid, str) and re.fullmatch(r"Q[1-9]\d*", qid):
            result.append(qid)
    return list(dict.fromkeys(result))


def _snak_qid(snak: Any) -> str | None:
    if not isinstance(snak, dict):
        return None
    value = snak.get("datavalue", {}).get("value", {})
    qid = value.get("id") if isinstance(value, dict) else None
    return qid if isinstance(qid, str) and re.fullmatch(r"Q[1-9]\d*", qid) else None


def _snak_date(snak: Any) -> str | None:
    if not isinstance(snak, dict):
        return None
    value = snak.get("datavalue", {}).get("value", {})
    raw = value.get("time") if isinstance(value, dict) else None
    return _literal_date(str(raw)) if raw else None


def _claim_intervals(
    entity: dict[str, Any], property_id: str,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for claim in entity.get("claims", {}).get(property_id, []):
        if not isinstance(claim, dict) or claim.get("rank") == "deprecated":
            continue
        target = _snak_qid(claim.get("mainsnak"))
        qualifiers = claim.get("qualifiers", {})
        if not target or not isinstance(qualifiers, dict):
            continue
        starts = sorted(
            value for snak in qualifiers.get("P580", [])
            if (value := _snak_date(snak)) is not None
        )
        ends = sorted(
            value for snak in qualifiers.get("P582", [])
            if (value := _snak_date(snak)) is not None
        )
        if starts:
            result.append({
                "target_qid": target, "start": starts[0],
                "end": ends[-1] if ends else "",
            })
    return result


def _active_targets(
    entity: dict[str, Any], property_id: str, cutoff: str,
) -> list[dict[str, str]]:
    return [
        row for row in _claim_intervals(entity, property_id)
        if row["start"] <= cutoff and (not row["end"] or row["end"] >= cutoff)
    ]


def _is_specific_office(entity: dict[str, Any]) -> bool:
    """Conservative candidate gate; formal singleton-office audit runs later."""
    name = (_entity_name(entity) or "").strip()
    if not name or _title(entity) is None:
        return False
    lowered = name.casefold()
    if any(re.search(pattern, lowered) for pattern in GENERIC_OFFICE_PATTERNS):
        return False
    # P1308 is the strongest cheap indication that the item models one office
    # with enumerated officeholders, rather than a reusable job class such as
    # judge, chairperson, or member of parliament. It remains only a candidate
    # gate: overlap/succession is proved later by the exhaustive order audit.
    return bool(_claim_targets(entity, "P1308"))


def _p39_event_skeletons(
    events: list[EventRow], max_candidates: int,
) -> list[dict[str, str]]:
    by_person: dict[str, list[EventRow]] = defaultdict(list)
    by_position: dict[str, list[EventRow]] = defaultdict(list)
    for event in events:
        by_person[event.subject_qid].append(event)
        by_position[event.value_qid].append(event)
    result = []
    for person_0, person_events in sorted(by_person.items()):
        first_date = min(event.start for event in person_events)
        first_positions = {
            event.value_qid for event in person_events if event.start == first_date
        }
        if len(first_positions) != 1:
            continue
        later_position = next(iter(first_positions))
        later = [
            event for event in by_position[later_position]
            if event.start > first_date and event.subject_qid != person_0
        ]
        if not later:
            continue
        next_date = min(event.start for event in later)
        successors = {
            event.subject_qid for event in later if event.start == next_date
        }
        if len(successors) != 1:
            continue
        result.append({
            "person_0_qid": person_0, "later_anchor_qid": later_position,
            "person_2_qid": next(iter(successors)),
            "start_1": first_date, "start_2": next_date,
        })
        if len(result) >= max_candidates:
            break
    return result


def _p39_transition_skeletons(
    events: list[EventRow], max_candidates: int,
) -> list[dict[str, str]]:
    by_person: dict[str, list[EventRow]] = defaultdict(list)
    for event in events:
        by_person[event.subject_qid].append(event)
    result = []
    for person_0, person_events in sorted(by_person.items()):
        first_date = min(event.start for event in person_events)
        first_positions = {
            event.value_qid for event in person_events if event.start == first_date
        }
        if len(first_positions) != 1:
            continue
        result.append({
            "person_0_qid": person_0,
            "later_anchor_qid": next(iter(first_positions)),
            "start_1": first_date,
        })
        if len(result) >= max_candidates:
            break
    return result


def _next_officeholder(
    office: dict[str, Any], *, incumbent_qid: str, boundary: str, until: str,
) -> tuple[str, str] | None:
    later = [
        row for row in _claim_intervals(office, "P1308")
        if boundary < row["start"] <= until and row["target_qid"] != incumbent_qid
    ]
    if not later:
        return None
    next_date = min(row["start"] for row in later)
    people = {row["target_qid"] for row in later if row["start"] == next_date}
    if len(people) != 1:
        return None
    return next(iter(people)), next_date


def _discover_p39_spines(
    events: list[EventRow], *, cutoff: str, until: str, max_spines: int,
    request_interval: float, unresolved_saturation: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    skeletons = _p39_transition_skeletons(events, len(events))
    later_position_qids = {row["later_anchor_qid"] for row in skeletons}
    entities = _fetch_entities(
        later_position_qids, request_interval=request_interval,
    )
    skeletons = [
        row for row in skeletons
        if _is_specific_office(entities.get(row["later_anchor_qid"], {}))
    ][:max(max_spines * 8, max_spines)]
    joined_skeletons = []
    for row in skeletons:
        successor = _next_officeholder(
            entities.get(row["later_anchor_qid"], {}),
            incumbent_qid=row["person_0_qid"], boundary=row["start_1"], until=until,
        )
        if successor is None:
            continue
        person_2_qid, start_2 = successor
        joined_skeletons.append({
            **row, "person_2_qid": person_2_qid, "start_2": start_2,
        })
    skeletons = joined_skeletons
    qids = {
        value for row in skeletons for key, value in row.items() if key.endswith("_qid")
    }
    entities.update(_fetch_entities(
        (qid for qid in qids if qid not in entities),
        request_interval=request_interval,
    ))
    active_by_person: dict[str, list[dict[str, str]]] = {}
    anchor_qids: set[str] = set()
    for row in skeletons:
        active = _active_targets(entities.get(row["person_0_qid"], {}), "P39", cutoff)
        active_by_person[row["person_0_qid"]] = active
        anchor_qids.update(value["target_qid"] for value in active)
    entities.update(_fetch_entities(anchor_qids, request_interval=request_interval))
    selected: list[dict[str, Any]] = []
    seen_paths = set()
    for row in skeletons:
        later_entity = entities.get(row["later_anchor_qid"], {})
        if not _is_specific_office(later_entity):
            continue
        active = sorted(
            active_by_person.get(row["person_0_qid"], []),
            key=lambda value: (value["start"], value["target_qid"]), reverse=True,
        )
        for held in active:
            anchor_qid = held["target_qid"]
            if anchor_qid == row["later_anchor_qid"]:
                continue
            if not _is_specific_office(entities.get(anchor_qid, {})):
                continue
            path = (
                anchor_qid, row["person_0_qid"], row["later_anchor_qid"],
                row["person_2_qid"],
            )
            if path in seen_paths:
                continue
            selected.append({
                **row, "anchor_qid": anchor_qid,
                "start_0": held["start"], "end_0": held["end"],
                "topology_id": "p39-to-p1308-office-succession",
                "domain_family": "career", "edge_0_property": "P39",
                "edge_1_property": "P39", "edge_2_property": "P1308",
                "specific_office_gate": "heuristic_pass_formal_singleton_audit_pending",
                "query_truncated": unresolved_saturation,
            })
            seen_paths.add(path)
            break
        if len(selected) >= max_spines:
            break
    return selected, entities


def _earliest_leadership_after(
    entity: dict[str, Any], boundary: str, until: str,
    leadership_properties: tuple[str, ...] = LEADERSHIP_PROPERTIES,
) -> tuple[str, str, str] | None:
    events = []
    for property_id in leadership_properties:
        for row in _claim_intervals(entity, property_id):
            if boundary < row["start"] <= until:
                events.append((row["start"], property_id, row["target_qid"]))
    if not events:
        return None
    earliest = min(row[0] for row in events)
    values = {(prop, target) for start, prop, target in events if start == earliest}
    if len(values) != 1:
        return None
    property_id, target = next(iter(values))
    return earliest, property_id, target


def _discover_mixed_spines(
    transition_property: str, events: list[EventRow],
    entities: dict[str, dict[str, Any]], *, cutoff: str, max_spines: int,
    until: str, unresolved_saturation: bool,
    leadership_properties: tuple[str, ...] = LEADERSHIP_PROPERTIES,
) -> list[dict[str, Any]]:
    by_person: dict[str, list[EventRow]] = defaultdict(list)
    for event in events:
        by_person[event.subject_qid].append(event)
    result = []
    seen_paths = set()
    for person_0, person_events in sorted(by_person.items()):
        first_date = min(event.start for event in person_events)
        first_values = {
            event.value_qid for event in person_events if event.start == first_date
        }
        if len(first_values) != 1:
            continue
        later_anchor = next(iter(first_values))
        leader = _earliest_leadership_after(
            entities.get(later_anchor, {}), first_date, until,
            leadership_properties,
        )
        if leader is None:
            continue
        start_2, edge_2_property, person_2 = leader
        active_positions = sorted(
            _active_targets(entities.get(person_0, {}), "P39", cutoff),
            key=lambda row: (row["start"], row["target_qid"]), reverse=True,
        )
        for active in active_positions:
            anchor_qid = active["target_qid"]
            if not _is_specific_office(entities.get(anchor_qid, {})):
                continue
            path = (anchor_qid, person_0, later_anchor, person_2)
            if len(set(path)) != 4 or path in seen_paths:
                continue
            result.append({
                "anchor_qid": anchor_qid,
                "person_0_qid": person_0, "later_anchor_qid": later_anchor,
                "person_2_qid": person_2, "start_0": active["start"],
                "end_0": active["end"], "start_1": first_date,
                "start_2": start_2,
                "topology_id": f"mixed-{transition_property}-leadership",
                "domain_family": (
                    "politics" if transition_property == "P102" else "career"
                ),
                "edge_0_property": "P39",
                "edge_1_property": transition_property,
                "edge_2_property": edge_2_property,
                "query_truncated": unresolved_saturation,
            })
            seen_paths.add(path)
            break
        if len(result) >= max_spines:
            break
    return result


def _source_noun(spec: TemporalRelationSpec) -> str:
    return spec.source_kind.replace("organization", "organisation")


def _question_text(
    *, spec: TemporalRelationSpec, anchor: str, cutoff: str,
    start_1: str, start_2: str, tail_question: str,
) -> tuple[str, str]:
    noun = _source_noun(spec)
    label = spec.label
    benchmark = " ".join((
        f"Step 1: Who was the {label} of {anchor} at the registered knowledge cutoff?",
        f"Step 2: For which other {noun} did the person from Step 1 first begin serving as {label} after the cutoff?",
        f"Step 3: Who next began serving as {label} for the {noun} from Step 2 after the tenure identified in Step 2 began?",
        f"Step 4: {tail_question}",
    ))
    audit = " ".join((
        f"Step 1: Who was the {label} of {anchor} at the registered knowledge cutoff ({cutoff})?",
        f"Step 2: For which other {noun} did that person begin serving as {label} on {start_1}?",
        f"Step 3: Who began serving as {label} for that {noun} on {start_2}?",
        f"Step 4: {tail_question}",
    ))
    return benchmark, audit


def _staged_question_text(
    spine: dict[str, Any], specs: dict[str, TemporalRelationSpec],
    *, anchor: str, cutoff: str, start_1: str, start_2: str,
    tail_question: str,
) -> tuple[str, str]:
    topology_id = str(spine["topology_id"])
    if topology_id.endswith("office-succession"):
        benchmark = " ".join((
            f"Step 1: Who held the position {anchor} at the registered knowledge cutoff?",
            "Step 2: Which other position did the person from Step 1 first begin holding after the cutoff?",
            "Step 3: Who next began holding the position from Step 2 after the tenure identified in Step 2 began?",
            f"Step 4: {tail_question}",
        ))
        audit = " ".join((
            f"Step 1: Who held the position {anchor} at the registered knowledge cutoff ({cutoff})?",
            f"Step 2: Which other position did that person begin holding on {start_1}?",
            f"Step 3: Who began holding that position on {start_2}?",
            f"Step 4: {tail_question}",
        ))
        return benchmark, audit
    transition = str(spine["edge_1_property"])
    transition_step = {
        "P102": "Which political party did the person from Step 1 first join after the cutoff?",
        "P108": "Which organisation first became the employer of the person from Step 1 after the cutoff?",
        "P1416": "With which organisation did the person from Step 1 first become affiliated after the cutoff?",
    }.get(transition)
    if transition_step is None:
        raise ValueError(f"unsupported staged transition property: {transition}")
    initial = specs[str(spine["edge_0_property"])]
    final = specs[str(spine["edge_2_property"])]
    initial_step = (
        f"Who held the position {anchor} at the registered knowledge cutoff?"
        if spine["edge_0_property"] == "P39"
        else f"Who was the {initial.label} of {anchor} at the registered knowledge cutoff?"
    )
    benchmark = " ".join((
        f"Step 1: {initial_step}",
        f"Step 2: {transition_step}",
        f"Step 3: Who first began serving as {final.label} of the organisation from Step 2 after the event identified in Step 2?",
        f"Step 4: {tail_question}",
    ))
    audit = " ".join((
        f"Step 1: {initial_step[:-1]} ({cutoff})?",
        f"Step 2: The {transition} event began on {start_1}.",
        f"Step 3: The later {final.label} tenure began on {start_2}.",
        f"Step 4: {tail_question}",
    ))
    return benchmark, audit


def _questions_for_staged_spines(
    spines: list[dict[str, Any]], specs: dict[str, TemporalRelationSpec],
    entities: dict[str, dict[str, Any]], *, model_id: str, cutoff: str,
    until: str, tail_specs: tuple[tuple[str, str, str], ...] = TAIL_SPECS,
) -> tuple[list[dict[str, Any]], set[str]]:
    provisional: list[dict[str, Any]] = []
    missing_tail_qids: set[str] = set()
    for spine in spines:
        chain_qids = [
            spine["anchor_qid"], spine["person_0_qid"],
            spine["later_anchor_qid"], spine["person_2_qid"],
        ]
        names = [_entity_name(entities.get(qid, {})) for qid in chain_qids]
        if any(name is None for name in names):
            continue
        person_2 = entities.get(spine["person_2_qid"], {})
        for tail_property, tail_family, tail_question in tail_specs:
            targets = _claim_targets(person_2, tail_property)
            if len(targets) != 1:
                continue
            missing_tail_qids.add(targets[0])
            provisional.append({
                "spine": spine, "names": [str(name) for name in names],
                "tail_property": tail_property, "tail_family": tail_family,
                "tail_question": tail_question, "tail_qid": targets[0],
            })
    if any(qid not in entities for qid in missing_tail_qids):
        return [], missing_tail_qids
    candidates: list[dict[str, Any]] = []
    for item in provisional:
        spine = item["spine"]
        anchor, person_0, later_anchor, person_2 = item["names"]
        answer = _entity_name(entities[item["tail_qid"]])
        if answer is None:
            continue
        benchmark, audit = _staged_question_text(
            spine, specs, anchor=anchor, cutoff=cutoff,
            start_1=spine["start_1"], start_2=spine["start_2"],
            tail_question=item["tail_question"],
        )
        if answer.casefold() in benchmark.casefold():
            continue
        relation_properties = [
            spine["edge_0_property"], spine["edge_1_property"],
            spine["edge_2_property"],
        ]
        identity = {
            "model_id": model_id, "cutoff": cutoff, "until": until,
            "topology_id": spine["topology_id"],
            "relation_properties": relation_properties,
            "entity_qids": [*[
                spine["anchor_qid"], spine["person_0_qid"],
                spine["later_anchor_qid"], spine["person_2_qid"],
            ], item["tail_qid"]],
            "event_dates": [spine["start_1"], spine["start_2"]],
            "tail_property": item["tail_property"],
        }
        fingerprint = _canonical_sha256(identity)
        directions = (
            ["inverse_at_cutoff", "forward", "inverse"]
            if str(spine["topology_id"]).endswith("office-succession")
            else [
                "inverse_at_cutoff" if spine["edge_0_property"] == "P39"
                else "forward_at_cutoff",
                "forward", "forward",
            ]
        )
        candidates.append({
            "schema_version": QUESTION_SCHEMA,
            "question_template_version": QUESTION_TEMPLATE_VERSION,
            "id": f"kgcand_{fingerprint[:20]}",
            "status": "pending_wikipedia_validation",
            "benchmark_question": benchmark, "audit_question_with_dates": audit,
            "expected_answer": answer, "expected_answer_qid": item["tail_qid"],
            "expected_answer_aliases": [answer], "public_anchor": anchor,
            "topology_id": spine["topology_id"],
            "relation_property": spine["edge_1_property"],
            "relation_properties": relation_properties,
            "relation_label": specs[str(spine["edge_1_property"])].label,
            "relation_family": spine["domain_family"],
            "domain_family": spine["domain_family"],
            "tail_property": item["tail_property"],
            "tail_family": item["tail_family"], "hop_count": 4,
            "temporal_transition_count": 2, "knowledge_cutoff": cutoff,
            "target_as_of": until,
            "private_chain": [
                {
                    "source_qid": spine["anchor_qid"], "source_title": anchor,
                    "target_qid": spine["person_0_qid"], "target_title": person_0,
                    "property_id": spine["edge_0_property"],
                    "direction": directions[0], "event_start": spine["start_0"],
                    "event_end": spine["end_0"] or None,
                },
                {
                    "source_qid": spine["person_0_qid"], "source_title": person_0,
                    "target_qid": spine["later_anchor_qid"],
                    "target_title": later_anchor,
                    "property_id": spine["edge_1_property"],
                    "direction": directions[1], "event_date": spine["start_1"],
                },
                {
                    "source_qid": spine["later_anchor_qid"],
                    "source_title": later_anchor,
                    "target_qid": spine["person_2_qid"], "target_title": person_2,
                    "property_id": spine["edge_2_property"],
                    "direction": directions[2], "event_date": spine["start_2"],
                },
                {
                    "source_qid": spine["person_2_qid"], "source_title": person_2,
                    "target_qid": item["tail_qid"], "target_title": answer,
                    "property_id": item["tail_property"],
                    "direction": "forward_attribute",
                },
            ],
            "specific_office_gate": spine.get("specific_office_gate"),
            "order_certificate_status": (
                "pending_exhaustive_query_limit_saturated"
                if spine["query_truncated"] else "pending_exhaustive_event_order_audit"
            ),
            "question_fingerprint": fingerprint,
            "validation_requirements": [
                "exhaustive_event_order_certificate",
                "specific_singleton_office_audit",
                "exact_historical_wikipedia_revision_evidence",
                "revision_present_hyperlinks", "independent_whole_chain_judge",
                "hash_bound_human_review", "per_model_factorized_pk_admission",
            ],
        })
    return candidates, missing_tail_qids


def _select_diverse_candidates(
    candidates: list[dict[str, Any]], *, max_questions: int,
    max_per_anchor: int, max_sports_share: float, max_topology_share: float,
) -> list[dict[str, Any]]:
    max_sports = math.floor(max_questions * max_sports_share)
    max_topology = max(1, math.floor(max_questions * max_topology_share))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in sorted(candidates, key=lambda row: row["id"]):
        groups[(str(candidate["topology_id"]), str(candidate["tail_property"]))].append(candidate)
    selected: list[dict[str, Any]] = []
    anchor_counts: dict[str, int] = defaultdict(int)
    topology_counts: dict[str, int] = defaultdict(int)
    sports_count = 0
    keys = sorted(groups)
    while keys and len(selected) < max_questions:
        next_keys = []
        progress = False
        for key in keys:
            topology_id = key[0]
            group = groups[key]
            chosen = None
            while group:
                candidate = group.pop(0)
                anchor = str(candidate["public_anchor"]).casefold()
                is_sports = candidate.get("domain_family") == "sports"
                if anchor_counts[anchor] >= max_per_anchor:
                    continue
                if topology_counts[topology_id] >= max_topology:
                    continue
                if is_sports and sports_count >= max_sports:
                    continue
                chosen = candidate
                break
            if chosen is not None:
                selected.append(chosen)
                anchor = str(chosen["public_anchor"]).casefold()
                anchor_counts[anchor] += 1
                topology_counts[topology_id] += 1
                sports_count += int(chosen.get("domain_family") == "sports")
                progress = True
                if len(selected) >= max_questions:
                    break
            if group and topology_counts[topology_id] < max_topology:
                next_keys.append(key)
        if not progress:
            break
        keys = next_keys
    return selected


def _questions_for_spines(
    spines_by_property: dict[str, list[dict[str, str]]],
    specs: dict[str, TemporalRelationSpec], entities: dict[str, dict[str, Any]],
    *, model_id: str, cutoff: str, until: str, max_questions: int,
    max_per_anchor: int, query_truncated: dict[str, bool],
    tail_specs: tuple[tuple[str, str, str], ...] = TAIL_SPECS,
) -> tuple[list[dict[str, Any]], set[str]]:
    provisional: list[dict[str, Any]] = []
    missing_tail_qids: set[str] = set()
    for property_id, spines in spines_by_property.items():
        spec = specs[property_id]
        for spine in spines:
            chain_qids = [
                spine["anchor_qid"], spine["person_0_qid"],
                spine["later_anchor_qid"], spine["person_2_qid"],
            ]
            names = [_entity_name(entities.get(qid, {})) for qid in chain_qids]
            if any(name is None for name in names):
                continue
            known_names = [str(name) for name in names]
            person_2 = entities.get(spine["person_2_qid"], {})
            for tail_property, tail_family, tail_question in tail_specs:
                targets = _claim_targets(person_2, tail_property)
                if len(targets) != 1:
                    continue
                missing_tail_qids.add(targets[0])
                provisional.append({
                    "spine": spine, "spec": spec, "names": known_names,
                    "tail_property": tail_property, "tail_family": tail_family,
                    "tail_question": tail_question, "tail_qid": targets[0],
                    "query_truncated": query_truncated[property_id],
                })

    # Caller fetches tail entities and invokes this helper again with them present.
    if any(qid not in entities for qid in missing_tail_qids):
        return [], missing_tail_qids

    candidates: list[dict[str, Any]] = []
    for item in provisional:
        spine = item["spine"]
        spec = item["spec"]
        anchor, person_0, later_anchor, person_2 = item["names"]
        answer = _entity_name(entities[item["tail_qid"]])
        if answer is None:
            continue
        benchmark, audit = _question_text(
            spec=spec, anchor=str(anchor), cutoff=cutoff,
            start_1=spine["start_1"], start_2=spine["start_2"],
            tail_question=item["tail_question"],
        )
        if answer.casefold() in benchmark.casefold():
            # A tail such as citizenship=Japan is unusable when the public anchor
            # already names Japan; do not let an easy lexical leak enter the batch.
            continue
        identity = {
            "model_id": model_id, "cutoff": cutoff, "until": until,
            "property_id": spec.property_id,
            "entity_qids": [
                spine["anchor_qid"], spine["person_0_qid"],
                spine["later_anchor_qid"], spine["person_2_qid"], item["tail_qid"],
            ],
            "event_dates": [spine["start_1"], spine["start_2"]],
            "tail_property": item["tail_property"],
        }
        fingerprint = _canonical_sha256(identity)
        candidates.append({
            "schema_version": QUESTION_SCHEMA,
            "question_template_version": QUESTION_TEMPLATE_VERSION,
            "id": f"kgcand_{fingerprint[:20]}",
            "status": "pending_wikipedia_validation",
            "benchmark_question": benchmark,
            "audit_question_with_dates": audit,
            "expected_answer": answer,
            "expected_answer_qid": item["tail_qid"],
            "expected_answer_aliases": [answer],
            "public_anchor": anchor,
            "topology_id": f"same-relation-forward-{spec.property_id}",
            "relation_property": spec.property_id,
            "relation_properties": [spec.property_id] * 3,
            "relation_label": spec.label,
            "relation_family": spec.family,
            "domain_family": "sports" if spec.property_id == "P286" else spec.family,
            "tail_property": item["tail_property"],
            "tail_family": item["tail_family"],
            "hop_count": 4,
            "temporal_transition_count": 2,
            "knowledge_cutoff": cutoff,
            "target_as_of": until,
            "private_chain": [
                {
                    "source_qid": spine["anchor_qid"], "source_title": anchor,
                    "target_qid": spine["person_0_qid"], "target_title": person_0,
                    "property_id": spec.property_id, "direction": "forward_at_cutoff",
                    "event_start": spine["start_0"], "event_end": spine["end_0"] or None,
                },
                {
                    "source_qid": spine["person_0_qid"], "source_title": person_0,
                    "target_qid": spine["later_anchor_qid"], "target_title": later_anchor,
                    "property_id": spec.property_id, "direction": "inverse",
                    "event_date": spine["start_1"],
                },
                {
                    "source_qid": spine["later_anchor_qid"], "source_title": later_anchor,
                    "target_qid": spine["person_2_qid"], "target_title": person_2,
                    "property_id": spec.property_id, "direction": "forward",
                    "event_date": spine["start_2"],
                },
                {
                    "source_qid": spine["person_2_qid"], "source_title": person_2,
                    "target_qid": item["tail_qid"], "target_title": answer,
                    "property_id": item["tail_property"], "direction": "forward_attribute",
                },
            ],
            "order_certificate_status": (
                "pending_exhaustive_query_limit_saturated"
                if item["query_truncated"] else "pending_exhaustive_event_order_audit"
            ),
            "question_fingerprint": fingerprint,
            "validation_requirements": [
                "exhaustive_event_order_certificate",
                "exact_historical_wikipedia_revision_evidence",
                "revision_present_hyperlinks",
                "independent_whole_chain_judge",
                "hash_bound_human_review",
                "per_model_factorized_pk_admission",
            ],
        })

    # Greedy round-robin across relation and tail properties, with an anchor cap.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in sorted(candidates, key=lambda row: row["id"]):
        groups[(candidate["relation_property"], candidate["tail_property"])].append(candidate)
    selected: list[dict[str, Any]] = []
    anchor_counts: dict[str, int] = defaultdict(int)
    keys = sorted(groups)
    while keys and len(selected) < max_questions:
        next_keys = []
        for key in keys:
            group = groups[key]
            chosen = None
            while group:
                candidate = group.pop(0)
                anchor_key = str(candidate["public_anchor"]).casefold()
                if anchor_counts[anchor_key] < max_per_anchor:
                    chosen = candidate
                    break
            if chosen is not None:
                selected.append(chosen)
                anchor_counts[str(chosen["public_anchor"]).casefold()] += 1
                if len(selected) >= max_questions:
                    break
            if group:
                next_keys.append(key)
        keys = next_keys
    return selected, missing_tail_qids


def _write_json(path: str, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(destination)


def _write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(destination)


def _write_markdown(path: str, questions: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Provisional temporal multi-hop question catalog", "",
        f"Generated: {metadata['created_at']}", "",
        f"Count: {len(questions)}", "",
        "> These are KG candidates, not Wikipedia-validated benchmark cases.", "",
    ]
    for index, question in enumerate(questions, start=1):
        lines.extend([
            f"## {index}. {question['id']}", "",
            str(question["benchmark_question"]), "",
            f"- Expected answer: {question['expected_answer']}",
            f"- Topology: {question.get('topology_id', question['relation_property'])}",
            f"- Relations: {' → '.join(question.get('relation_properties', [question['relation_property']]))} / tail {question['tail_property']}",
            f"- Domain: {question.get('domain_family', question['relation_family'])}",
            f"- Status: {question['status']}", "",
        ])
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _refresh_question_texts(
    questions: list[dict[str, Any]], specs: dict[str, TemporalRelationSpec],
    tail_specs: tuple[tuple[str, str, str], ...] = TAIL_SPECS,
) -> list[dict[str, Any]]:
    tail_questions = {property_id: question for property_id, _, question in tail_specs}
    refreshed = []
    for raw in questions:
        relation_property = str(raw.get("relation_property", ""))
        tail_property = str(raw.get("tail_property", ""))
        spec = specs.get(relation_property)
        tail_question = tail_questions.get(tail_property)
        chain = raw.get("private_chain")
        if spec is None or tail_question is None or not isinstance(chain, list) or len(chain) < 3:
            raise ValueError(f"{raw.get('id')}: cannot refresh malformed candidate")
        topology_id = str(raw.get("topology_id", ""))
        if (
            topology_id.startswith("inverted-")
            or topology_id.startswith("mixed-")
            or topology_id.startswith("p39-to-")
        ):
            staged_spine = {
                "topology_id": topology_id,
                "edge_0_property": str(chain[0]["property_id"]),
                "edge_1_property": str(chain[1]["property_id"]),
                "edge_2_property": str(chain[2]["property_id"]),
            }
            benchmark, audit = _staged_question_text(
                staged_spine, specs, anchor=str(raw["public_anchor"]),
                cutoff=str(raw["knowledge_cutoff"]),
                start_1=str(chain[1]["event_date"]),
                start_2=str(chain[2]["event_date"]), tail_question=tail_question,
            )
        else:
            benchmark, audit = _question_text(
                spec=spec, anchor=str(raw["public_anchor"]),
                cutoff=str(raw["knowledge_cutoff"]),
                start_1=str(chain[1]["event_date"]),
                start_2=str(chain[2]["event_date"]), tail_question=tail_question,
            )
        aliases = raw.get("expected_answer_aliases", [])
        if any(str(alias).casefold() in benchmark.casefold() for alias in aliases):
            continue
        refreshed.append({
            **raw, "question_template_version": QUESTION_TEMPLATE_VERSION,
            "benchmark_question": benchmark, "audit_question_with_dates": audit,
        })
    return refreshed


def _batch_hash(metadata: dict[str, Any], questions: list[dict[str, Any]]) -> str:
    return _canonical_sha256({
        "metadata": {key: value for key, value in metadata.items()
                     if key not in {"raw_query_artifacts", "batch_sha256"}},
        "questions_sha256": _canonical_sha256([{
            "id": row["id"], "question": row["benchmark_question"],
            "answer_qid": row["expected_answer_qid"],
            "fingerprint": row["question_fingerprint"],
        } for row in questions]),
    })


def _property_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("property selection cannot be empty")
    invalid = [item for item in result if not re.fullmatch(r"P[1-9]\d*", item)]
    if invalid:
        raise ValueError(f"invalid property IDs: {sorted(set(invalid))!r}")
    return list(dict.fromkeys(result))


def _resolve_candidate_relations(
    registry: TemporalRelationRegistry,
    topology_registry: CandidateTopologyRegistry,
    admitted_property_ids: set[str],
    *,
    requested_direct: list[str] | None = None,
    requested_staged: list[str] | None = None,
) -> tuple[dict[str, TemporalRelationSpec], dict[str, TemporalRelationSpec], tuple[str, ...]]:
    """Resolve generator capabilities against the admitted relation set."""
    by_property = {spec.property_id: spec for spec in registry.relations}
    missing_admissions = admitted_property_ids - set(by_property)
    if missing_admissions:
        raise ValueError(
            f"admitted properties absent from registry: {sorted(missing_admissions)!r}"
        )
    direct_contract = topology_registry.direct_topology
    direct_capable = {
        property_id for property_id in admitted_property_ids
        if (
            by_property[property_id].time_mode in set(direct_contract.time_modes)
            and by_property[property_id].answer_kind == direct_contract.answer_kind
            and by_property[property_id].target_kind == direct_contract.target_kind
        )
    }
    if requested_direct is not None:
        unavailable = set(requested_direct) - direct_capable
        if unavailable:
            raise ValueError(
                "direct properties are not admitted or incompatible with the "
                f"person-transition topology: {sorted(unavailable)!r}"
            )
        direct_ids = requested_direct
    else:
        direct_ids = sorted(direct_capable)

    leadership_ids = tuple(
        property_id for property_id in topology_registry.leadership_relation_ids
        if property_id in admitted_property_ids
    )
    staged_capable: dict[str, TemporalRelationSpec] = {}
    for topology in topology_registry.staged_topologies:
        property_id = topology.event_property_id
        if not set(topology.required_relation_ids) <= admitted_property_ids:
            continue
        if topology.adapter == "mixed_leadership" and not leadership_ids:
            continue
        spec = by_property.get(property_id)
        if spec is not None and spec.time_mode in {"interval", "either"}:
            staged_capable[property_id] = spec
    if requested_staged is not None:
        unavailable = set(requested_staged) - set(staged_capable)
        if unavailable:
            raise ValueError(
                "staged properties lack admission, dependencies, or an adapter: "
                f"{sorted(unavailable)!r}"
            )
        staged_ids = requested_staged
    else:
        staged_ids = sorted(staged_capable)

    direct = {property_id: by_property[property_id] for property_id in direct_ids}
    staged = {property_id: staged_capable[property_id] for property_id in staged_ids}
    if not direct and not staged:
        raise ValueError("no admitted relation is supported by a candidate topology")
    return direct, staged, leadership_ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate at least N provisional multi-hop questions from Wikidata"
    )
    parser.add_argument("--model-id", default="openai/gpt-4.1-mini")
    parser.add_argument("--until", default=date.today().isoformat())
    parser.add_argument(
        "--properties",
        help="optional direct-relation subset; default is every admitted compatible relation",
    )
    parser.add_argument(
        "--staged-properties",
        help="optional staged-relation subset; default is every admitted configured adapter",
    )
    parser.add_argument("--registry")
    parser.add_argument(
        "--profile", action="append",
        help="immutable profiler artifact; repeat for multiple profiling windows",
    )
    parser.add_argument("--topology-registry")
    parser.add_argument(
        "--allow-bootstrap-candidates", action="store_true",
        help="explicit provisional fallback when no active semantic profile is available",
    )
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--max-spines-per-property", type=int, default=160)
    parser.add_argument("--max-per-anchor", type=int, default=3)
    parser.add_argument("--query-limit", type=int, default=600)
    parser.add_argument("--event-window-days", type=int, default=180)
    parser.add_argument(
        "--event-first-profile-samples", action="store_true",
        help="seed direct joins from post-cutoff events already sampled by profiles",
    )
    parser.add_argument(
        "--event-first-discover-events", action="store_true",
        help="add small-window post-cutoff event queries before targeted joins",
    )
    parser.add_argument("--event-first-batch-size", type=int, default=12)
    parser.add_argument("--event-first-max-seeds-per-property", type=int, default=200)
    parser.add_argument(
        "--require-pair-novelty", action="store_true",
        help=(
            "candidate-only prefilter requiring each post-cutoff source-target "
            "pair to have no earlier qualified tenure; default is unchanged"
        ),
    )
    parser.add_argument("--max-sports-share", type=float, default=0.2)
    parser.add_argument("--max-topology-share", type=float, default=0.35)
    parser.add_argument("--request-interval", type=float, default=0.25)
    parser.add_argument("--output", default="candidate_question_batch.json")
    parser.add_argument("--packets-output", default="candidate_question_batch_packets.jsonl")
    parser.add_argument("--markdown-output", default="candidate_question_catalog.md")
    parser.add_argument(
        "--reuse-batch",
        help="refresh deterministic wording from an existing batch without API calls",
    )
    parser.add_argument(
        "--merge-batches", nargs="+",
        help="merge existing candidate JSON batches and reapply diversity caps",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.event_first_discover_events and not args.event_first_profile_samples:
        parser.error(
            "--event-first-discover-events requires --event-first-profile-samples"
        )
    for value in (
        args.max_questions, args.max_spines_per_property, args.max_per_anchor,
        args.query_limit, args.event_window_days, args.event_first_batch_size,
        args.event_first_max_seeds_per_property,
    ):
        if value <= 0:
            parser.error("all count arguments must be > 0")
    if args.request_interval < 0:
        parser.error("--request-interval must be >= 0")
    for name, value in (
        ("--max-sports-share", args.max_sports_share),
        ("--max-topology-share", args.max_topology_share),
    ):
        if not 0 < value <= 1:
            parser.error(f"{name} must be in (0, 1]")
    until = _strict_date(args.until)
    cutoff = get_model_cutoff(args.model_id).cutoff_date
    if until <= cutoff:
        parser.error("--until must be after the model cutoff")
    outputs = (args.output, args.packets_output, args.markdown_output)
    for output in outputs:
        if not args.overwrite:
            assert_new_output_path(output)

    if args.merge_batches:
        merged_candidates: list[dict[str, Any]] = []
        sources = []
        for source in args.merge_batches:
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
            raw_questions = payload.get("questions") if isinstance(payload, dict) else None
            if not isinstance(raw_questions, list):
                parser.error(f"{source}: candidate batch needs a questions list")
            source_metadata = payload.get("_batch", {})
            merged_candidates.extend(
                dict(row) for row in raw_questions if isinstance(row, dict)
            )
            sources.append({
                "path": source,
                "batch_sha256": (
                    source_metadata.get("batch_sha256")
                    if isinstance(source_metadata, dict) else None
                ),
                "question_count": len(raw_questions),
            })
        deduplicated = {
            str(row.get("question_fingerprint", row.get("id"))): row
            for row in merged_candidates
        }
        questions = _select_diverse_candidates(
            list(deduplicated.values()), max_questions=args.max_questions,
            max_per_anchor=args.max_per_anchor,
            max_sports_share=args.max_sports_share,
            max_topology_share=args.max_topology_share,
        )
        merge_topology_counts: dict[str, int] = defaultdict(int)
        merge_domain_counts: dict[str, int] = defaultdict(int)
        merge_property_counts: dict[str, int] = defaultdict(int)
        merge_tail_counts: dict[str, int] = defaultdict(int)
        for question in questions:
            merge_topology_counts[str(question["topology_id"])] += 1
            merge_domain_counts[str(question["domain_family"])] += 1
            merge_property_counts[str(question["relation_property"])] += 1
            merge_tail_counts[str(question["tail_property"])] += 1
        metadata = {
            "schema_version": BATCH_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "candidate_only_not_experiment_ready",
            "model_id": args.model_id, "knowledge_cutoff": cutoff, "until": until,
            "requested_count": args.max_questions, "generated_count": len(questions),
            "candidate_pool_count": len(deduplicated), "source_batches": sources,
            "counts_by_relation_property": dict(sorted(merge_property_counts.items())),
            "counts_by_tail_property": dict(sorted(merge_tail_counts.items())),
            "counts_by_topology": dict(sorted(merge_topology_counts.items())),
            "counts_by_domain": dict(sorted(merge_domain_counts.items())),
            "diversity_caps": {
                "max_sports_share": args.max_sports_share,
                "max_topology_share": args.max_topology_share,
                "max_per_anchor": args.max_per_anchor,
            },
            "formal_admission_required": True,
            "question_template_version": QUESTION_TEMPLATE_VERSION,
        }
        metadata["batch_sha256"] = _batch_hash(metadata, questions)
        _write_json(args.output, {"_batch": metadata, "questions": questions})
        _write_jsonl(args.packets_output, [{
            "stage": "batch_merge", "status": "pass", **metadata,
        }])
        _write_markdown(args.markdown_output, questions, metadata)
        print(
            f"[done] merged={len(questions)} requested={args.max_questions} "
            f"output={args.output}", flush=True,
        )
        return 0 if len(questions) >= args.max_questions else 1

    try:
        registry = load_temporal_relation_registry(args.registry)
        topology_registry = load_candidate_topology_registry(args.topology_registry)
        requested = _property_list(args.properties)
        staged_requested = _property_list(args.staged_properties)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    by_property = {spec.property_id: spec for spec in registry.relations}
    tail_specs = tuple(
        (tail.property_id, tail.family, tail.question)
        for tail in topology_registry.tails
    )

    if args.reuse_batch:
        payload = json.loads(Path(args.reuse_batch).read_text(encoding="utf-8"))
        raw_questions = payload.get("questions") if isinstance(payload, dict) else None
        if not isinstance(raw_questions, list):
            parser.error("--reuse-batch must contain a questions list")
        try:
            questions = _refresh_question_texts(
                [dict(row) for row in raw_questions if isinstance(row, dict)],
                by_property, tail_specs,
            )[:args.max_questions]
        except (KeyError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        source_metadata = payload.get("_batch", {})
        metadata = dict(source_metadata) if isinstance(source_metadata, dict) else {}
        metadata.update({
            "schema_version": BATCH_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "candidate_only_not_experiment_ready",
            "generated_count": len(questions),
            "requested_count": args.max_questions,
            "question_template_version": QUESTION_TEMPLATE_VERSION,
            "reused_from_batch_sha256": source_metadata.get("batch_sha256"),
        })
        metadata["batch_sha256"] = _batch_hash(metadata, questions)
        _write_json(args.output, {"_batch": metadata, "questions": questions})
        _write_jsonl(args.packets_output, [{
            "stage": "manifest", **{
                key: value for key, value in metadata.items()
                if key != "raw_query_artifacts"
            },
        }, {
            "stage": "wording_refresh", "status": "pass",
            "source": args.reuse_batch, "question_count": len(questions),
            "template_version": QUESTION_TEMPLATE_VERSION,
        }])
        _write_markdown(args.markdown_output, questions, metadata)
        print(
            f"[done] refreshed={len(questions)} requested={args.max_questions} "
            f"output={args.output}", flush=True,
        )
        return 0 if len(questions) >= args.max_questions else 1

    if args.profile and args.allow_bootstrap_candidates:
        parser.error("--profile and --allow-bootstrap-candidates are mutually exclusive")
    admission_records: list[dict[str, Any]] = []
    profile_artifacts: list[dict[str, str]] = []
    if args.profile:
        try:
            admissions = load_relation_admissions(
                args.profile, registry, require_active=True,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        admitted_property_ids = {row.spec.property_id for row in admissions}
        selection_mode = "active_semantic_profile_intersection"
        profile_artifacts = [{
            "path": str(Path(path)),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        } for path in args.profile]
        admission_records = [{
            "property_id": row.spec.property_id,
            "recommendations": list(row.recommendations),
            "profile_sha256s": list(row.profile_sha256s),
            "semantic_support_rate": row.semantic_support_rate,
        } for row in admissions]
    elif args.allow_bootstrap_candidates:
        admitted_property_ids = {
            spec.property_id for spec in registry.relations
            if spec.status not in {"quarantined", "deprecated"}
        }
        selection_mode = "explicit_provisional_bootstrap"
    else:
        parser.error(
            "fresh generation requires --profile with active semantic admissions; "
            "use --allow-bootstrap-candidates only for explicitly provisional discovery"
        )
    try:
        specs, staged_specs, leadership_properties = _resolve_candidate_relations(
            registry, topology_registry, admitted_property_ids,
            requested_direct=requested, requested_staged=staged_requested,
        )
    except ValueError as exc:
        parser.error(str(exc))
    wording_specs = {
        property_id: by_property[property_id]
        for property_id in set(specs) | set(staged_specs) | set(leadership_properties)
        if property_id in by_property
    }

    packets: list[dict[str, Any]] = []
    raw_artifacts = []
    spines_by_property: dict[str, list[dict[str, str]]] = {}
    query_truncated: dict[str, bool] = {}
    for property_id, spec in specs.items():
        started = time.monotonic()
        try:
            event_first_artifacts: list[dict[str, Any]] = []
            discovery_artifacts: list[dict[str, Any]] = []
            discovery_unresolved = False
            if args.event_first_profile_samples:
                event_seeds = _profile_event_seeds(
                    args.profile or [], property_id=property_id,
                    cutoff=cutoff, until=until,
                )
                if args.event_first_discover_events:
                    discovered, artifacts, discovery_unresolved = (
                        _fetch_event_rows_adaptive(
                            property_id, cutoff=cutoff, until=until,
                            limit=args.query_limit,
                            request_interval=args.request_interval,
                            window_days=args.event_window_days,
                        )
                    )
                    discovered_seeds = [{
                        "source_qid": row.subject_qid,
                        "target_qid": row.value_qid,
                        "start": row.start,
                    } for row in discovered]
                    event_seeds = list({
                        (row["source_qid"], row["target_qid"], row["start"]): row
                        for row in [*event_seeds, *discovered_seeds]
                    }.values())
                    event_seeds.sort(key=lambda row: (
                        row["start"], row["source_qid"], row["target_qid"],
                    ))
                    discovery_artifacts = [artifact.to_dict() for artifact in artifacts]
                event_seeds = event_seeds[:args.event_first_max_seeds_per_property]
                raw_rows, event_first_artifacts = _event_first_wdqs_rows(
                    spec, cutoff=cutoff, until=until, events=event_seeds,
                    batch_size=args.event_first_batch_size,
                    limit=args.query_limit, request_interval=args.request_interval,
                    require_pair_novelty=args.require_pair_novelty,
                ) if event_seeds else ([], [])
                query = "event-first-batched-profile-samples"
            else:
                event_seeds = []
                query, raw_rows = _wdqs_rows(
                    spec, cutoff=cutoff, until=until, limit=args.query_limit,
                    request_interval=args.request_interval,
                    require_pair_novelty=args.require_pair_novelty,
                )
            normalized = _normalize_rows(raw_rows)
            spines = _select_spines(normalized, args.max_spines_per_property)
            saturated = (
                any(row["row_count"] >= args.query_limit for row in event_first_artifacts)
                if args.event_first_profile_samples
                else len(raw_rows) >= args.query_limit
            )
            spines_by_property[property_id] = spines
            query_truncated[property_id] = saturated
            raw_artifacts.append({
                "property_id": property_id, "query": query,
                "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "row_count": len(raw_rows), "normalized_rows": normalized,
                "limit_saturated": saturated,
                "event_first": args.event_first_profile_samples,
                "event_seed_count": len(event_seeds),
                "event_first_batches": event_first_artifacts,
                "event_discovery_batches": discovery_artifacts,
                "event_discovery_unresolved_saturation": discovery_unresolved,
            })
            packet = {
                "stage": "kg_spine_query", "property_id": property_id,
                "status": "pass" if spines else "empty",
                "row_count": len(raw_rows), "normalized_count": len(normalized),
                "selected_spines": len(spines), "limit_saturated": saturated,
                "event_first": args.event_first_profile_samples,
                "event_seed_count": len(event_seeds),
                "query_batch_count": len(event_first_artifacts),
                "event_discovery_batch_count": len(discovery_artifacts),
                "event_discovery_unresolved_saturation": discovery_unresolved,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
        except Exception as exc:
            spines_by_property[property_id] = []
            query_truncated[property_id] = False
            packet = {
                "stage": "kg_spine_query", "property_id": property_id,
                "status": "infrastructure_error", "error": str(exc),
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
        packets.append(packet)
        print(json.dumps(packet, ensure_ascii=False), flush=True)

    staged_spines: list[dict[str, Any]] = []
    staged_entities: dict[str, dict[str, Any]] = {}
    event_cache: dict[str, tuple[list[EventRow], bool]] = {}
    for property_id in staged_specs:
        started = time.monotonic()
        try:
            events, artifacts, unresolved = _fetch_event_rows_adaptive(
                property_id, cutoff=cutoff, until=until, limit=args.query_limit,
                request_interval=args.request_interval,
                window_days=args.event_window_days,
            )
            event_cache[property_id] = (events, unresolved)
            raw_artifacts.extend({
                "topology_stage": "post_cutoff_events", **artifact.to_dict(),
            } for artifact in artifacts)
            packet = {
                "stage": "staged_event_query", "property_id": property_id,
                "status": "pass" if events else "empty", "event_count": len(events),
                "window_count": len(artifacts),
                "unresolved_limit_saturation": unresolved,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
        except Exception as exc:
            event_cache[property_id] = ([], False)
            packet = {
                "stage": "staged_event_query", "property_id": property_id,
                "status": "infrastructure_error", "error": str(exc),
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
        packets.append(packet)
        print(json.dumps(packet, ensure_ascii=False), flush=True)

    staged_adapters = {
        row.event_property_id: row.adapter
        for row in topology_registry.staged_topologies
    }
    office_properties = [
        property_id for property_id in staged_specs
        if staged_adapters.get(property_id) == "p39_office_succession"
    ]
    if len(office_properties) > 1:
        parser.error("only one p39_office_succession adapter may be selected")
    if office_properties and event_cache.get(office_properties[0], ([], False))[0]:
        office_property = office_properties[0]
        started = time.monotonic()
        try:
            events, unresolved = event_cache[office_property]
            p39_spines, p39_entities = _discover_p39_spines(
                events, cutoff=cutoff, until=until,
                max_spines=args.max_spines_per_property,
                request_interval=args.request_interval,
                unresolved_saturation=unresolved,
            )
            staged_spines.extend(p39_spines)
            staged_entities.update(p39_entities)
            packet = {
                "stage": "topology_join", "topology_id": "p39-to-p1308-office-succession",
                "status": "pass" if p39_spines else "empty",
                "selected_spines": len(p39_spines),
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
        except Exception as exc:
            packet = {
                "stage": "topology_join", "topology_id": "p39-to-p1308-office-succession",
                "status": "infrastructure_error", "error": str(exc),
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
        packets.append(packet)
        print(json.dumps(packet, ensure_ascii=False), flush=True)

    mixed_properties = [
        property_id for property_id in staged_specs
        if staged_adapters.get(property_id) == "mixed_leadership"
        and event_cache.get(property_id, ([], False))[0]
    ]
    if mixed_properties:
        bounded_events = {
            property_id: event_cache[property_id][0][
                :max(args.max_spines_per_property * 4, args.max_spines_per_property)
            ]
            for property_id in mixed_properties
        }
        mixed_qids = {
            qid for property_id in mixed_properties
            for event in bounded_events[property_id]
            for qid in (event.subject_qid, event.value_qid)
        }
        staged_entities.update(_fetch_entities(
            mixed_qids, request_interval=args.request_interval,
        ))
        mixed_anchor_qids = {
            active["target_qid"] for property_id in mixed_properties
            for event in bounded_events[property_id]
            for active in _active_targets(
                staged_entities.get(event.subject_qid, {}), "P39", cutoff,
            )
        }
        staged_entities.update(_fetch_entities(
            (qid for qid in mixed_anchor_qids if qid not in staged_entities),
            request_interval=args.request_interval,
        ))
        for property_id in mixed_properties:
            started = time.monotonic()
            _, unresolved = event_cache[property_id]
            relevant = bounded_events[property_id]
            spines = _discover_mixed_spines(
                property_id, relevant, staged_entities, cutoff=cutoff,
                max_spines=args.max_spines_per_property, until=until,
                unresolved_saturation=unresolved,
                leadership_properties=leadership_properties,
            )
            staged_spines.extend(spines)
            packet = {
                "stage": "topology_join",
                "topology_id": f"mixed-{property_id}-leadership",
                "status": "pass" if spines else "empty",
                "selected_spines": len(spines), "relevant_events": len(relevant),
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
            packets.append(packet)
            print(json.dumps(packet, ensure_ascii=False), flush=True)

    spine_qids = {
        value for spines in spines_by_property.values() for spine in spines
        for key, value in spine.items() if key.endswith("_qid")
    }
    spine_qids.update({
        value for spine in staged_spines for key, value in spine.items()
        if key.endswith("_qid")
    })
    entities = staged_entities
    entities.update(_fetch_entities(
        (qid for qid in spine_qids if qid not in entities),
        request_interval=args.request_interval,
    ))
    candidate_limit = max(args.max_questions * 10, args.max_questions)
    _, direct_tail_qids = _questions_for_spines(
        spines_by_property, specs, entities, model_id=args.model_id,
        cutoff=cutoff, until=until, max_questions=candidate_limit,
        max_per_anchor=args.max_per_anchor, query_truncated=query_truncated,
        tail_specs=tail_specs,
    )
    _, staged_tail_qids = _questions_for_staged_spines(
        staged_spines, wording_specs, entities, model_id=args.model_id,
        cutoff=cutoff, until=until, tail_specs=tail_specs,
    )
    entities.update(_fetch_entities(
        direct_tail_qids | staged_tail_qids, request_interval=args.request_interval,
    ))
    direct_questions, _ = _questions_for_spines(
        spines_by_property, specs, entities, model_id=args.model_id,
        cutoff=cutoff, until=until, max_questions=candidate_limit,
        max_per_anchor=args.max_per_anchor, query_truncated=query_truncated,
        tail_specs=tail_specs,
    )
    staged_questions, _ = _questions_for_staged_spines(
        staged_spines, wording_specs, entities, model_id=args.model_id,
        cutoff=cutoff, until=until, tail_specs=tail_specs,
    )
    questions = _select_diverse_candidates(
        direct_questions + staged_questions, max_questions=args.max_questions,
        max_per_anchor=args.max_per_anchor,
        max_sports_share=args.max_sports_share,
        max_topology_share=args.max_topology_share,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    counts_by_property: dict[str, int] = defaultdict(int)
    counts_by_tail: dict[str, int] = defaultdict(int)
    counts_by_topology: dict[str, int] = defaultdict(int)
    counts_by_domain: dict[str, int] = defaultdict(int)
    for question in questions:
        counts_by_property[question["relation_property"]] += 1
        counts_by_tail[question["tail_property"]] += 1
        counts_by_topology[question["topology_id"]] += 1
        counts_by_domain[question["domain_family"]] += 1
    metadata = {
        "schema_version": BATCH_SCHEMA, "created_at": created_at,
        "status": "candidate_only_not_experiment_ready",
        "model_id": args.model_id, "knowledge_cutoff": cutoff, "until": until,
        "registry_version": registry.registry_version,
        "registry_sha256": _canonical_sha256(registry.to_dict()),
        "topology_registry_version": topology_registry.registry_version,
        "topology_registry_sha256": _canonical_sha256(topology_registry.to_dict()),
        "relation_selection_mode": selection_mode,
        "profile_artifacts": profile_artifacts,
        "selected_direct_properties": sorted(specs),
        "selected_staged_properties": sorted(staged_specs),
        "selected_leadership_properties": list(leadership_properties),
        "relation_admissions": admission_records,
        "requested_count": args.max_questions, "generated_count": len(questions),
        "counts_by_relation_property": dict(sorted(counts_by_property.items())),
        "counts_by_tail_property": dict(sorted(counts_by_tail.items())),
        "counts_by_topology": dict(sorted(counts_by_topology.items())),
        "counts_by_domain": dict(sorted(counts_by_domain.items())),
        "candidate_pool_count": len(direct_questions) + len(staged_questions),
        "diversity_caps": {
            "max_sports_share": args.max_sports_share,
            "max_topology_share": args.max_topology_share,
            "max_per_anchor": args.max_per_anchor,
        },
        "raw_query_artifacts": raw_artifacts,
        "formal_admission_required": True,
        "question_template_version": QUESTION_TEMPLATE_VERSION,
    }
    metadata["batch_sha256"] = _batch_hash(metadata, questions)
    _write_json(args.output, {"_batch": metadata, "questions": questions})
    _write_jsonl(args.packets_output, [
        {"stage": "manifest", **{key: value for key, value in metadata.items()
                                  if key != "raw_query_artifacts"}},
        *packets,
    ])
    _write_markdown(args.markdown_output, questions, metadata)
    print(
        f"[done] generated={len(questions)} requested={args.max_questions} "
        f"output={args.output}", flush=True,
    )
    return 0 if len(questions) >= args.max_questions else 1


if __name__ == "__main__":
    raise SystemExit(main())
