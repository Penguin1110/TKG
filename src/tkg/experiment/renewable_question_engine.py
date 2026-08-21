"""Profile-driven renewable temporal multi-hop question generation.

This module closes the loop between relation profiling and formal question
admission.  Wikidata proposes time-qualified edges, exact historical Wikipedia
revisions prove that those edges were visible, a bounded beam search composes
world-event-ordered entity transitions, and the existing deterministic/LLM
gates admit final cases. A content-addressed SQLite ledger prevents regeneration of
the same semantic path across scheduled runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import unquote, urlparse

import requests

from tkg.api.openrouter import call_model, model_call_context
from tkg.experiment.attribute_tail_generation import generate_attribute_tail_variants
from tkg.experiment.event_order import build_event_order_certificate
from tkg.experiment.model_cutoffs import get_model_cutoff
from tkg.experiment.multihop_generation import (
    MultiHopQuestionJudge,
    MultiHopQuestionWriter,
    MultiHopSeed,
    QuestionWriterGateError,
    build_case,
    compose_relative_question,
    has_infrastructure_error,
    validate_chain,
    validate_shortest_arena,
)
from tkg.experiment.question_generation import _fold, _plain_text, call_json_model
from tkg.experiment.question_ledger import QuestionLedger, question_fingerprint
from tkg.experiment.relation_catalog import PERSON_RELATIONS
from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_relation_profiler import (
    PROFILE_SCHEMA, WDQS_URL, semantic_judge_schema_errors,
)
from tkg.experiment.temporal_relation_registry import (
    TemporalRelationRegistry,
    TemporalRelationSpec,
    load_temporal_relation_registry,
)
from tkg.wikipedia.backend import WikipediaError, WikipediaPageBackend
from tkg.wikipedia.pageviews import PopularitySnapshot, fetch_top_pageviews


ENGINE_SCHEMA = "renewable-temporal-question-engine-v3"
ADMITTED_RECOMMENDATIONS = {
    "semantic_validated_candidate_human_review_required",
}
DATE_OFFSETS = (0, 1, 3, 7)
USER_AGENT = "TKG-renewable-question-engine/0.1 (research)"
EDGE_CONTRAST_CHECKS = (
    "after_relation_supported", "before_relation_absent", "direction_correct",
)


def _strict_date(value: str) -> date:
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError(f"date must use strict YYYY-MM-DD form: {value!r}")
    return parsed


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _uri_id(value: str, prefix: str) -> str:
    result = value.rsplit("/", 1)[-1]
    if not re.fullmatch(rf"{re.escape(prefix)}[1-9]\d*", result):
        raise ValueError(f"invalid {prefix} entity URI: {value!r}")
    return result


def _article_title(value: str) -> str:
    path = unquote(urlparse(value).path)
    if "/wiki/" not in path:
        raise ValueError(f"not an English Wikipedia article URL: {value!r}")
    return " ".join(path.split("/wiki/", 1)[1].replace("_", " ").split())


def _binding(row: dict[str, Any], key: str) -> str:
    value = row.get(key, {})
    return str(value.get("value", "")).strip() if isinstance(value, dict) else ""


@dataclass(frozen=True)
class RelationAdmission:
    spec: TemporalRelationSpec
    recommendations: tuple[str, ...]
    profile_sha256s: tuple[str, ...]
    metrics: tuple[dict[str, Any], ...]
    source_samples: tuple[dict[str, Any], ...]

    @property
    def semantic_support_rate(self) -> float:
        values = [
            float(row["semantic_support_rate"])
            for row in self.metrics
            if row.get("semantic_support_rate") is not None
        ]
        return min(values) if values else 0.0


def load_relation_admissions(
    profile_paths: list[str | Path],
    registry: TemporalRelationRegistry,
    *,
    require_active: bool = False,
) -> tuple[RelationAdmission, ...]:
    """Load only semantically validated, registry-compatible relation profiles."""
    if not profile_paths:
        raise ValueError("at least one temporal relation profile is required")
    by_property = {spec.property_id: spec for spec in registry.relations}
    collected: dict[str, dict[str, Any]] = {}
    for path_value in sorted(str(Path(path)) for path in profile_paths):
        path = Path(path_value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != PROFILE_SCHEMA:
            raise ValueError(f"{path}: not a {PROFILE_SCHEMA} artifact")
        profile_registry = payload.get("registry", {})
        if profile_registry.get("schema_version") != registry.schema_version:
            raise ValueError(f"{path}: registry schema does not match generator registry")
        profile_hash = _file_sha256(path)
        raw_profiles = payload.get("profiles")
        if not isinstance(raw_profiles, list):
            raise ValueError(f"{path}: profiles must be a list")
        for raw in raw_profiles:
            if not isinstance(raw, dict):
                continue
            relation = raw.get("relation", {})
            property_id = str(relation.get("property_id", ""))
            spec = by_property.get(property_id)
            if spec is None or spec.status in {"quarantined", "deprecated"}:
                continue
            if require_active and spec.status != "active":
                continue
            for field in ("label", "family", "time_mode"):
                if str(relation.get(field, "")) != str(getattr(spec, field)):
                    raise ValueError(
                        f"{path}: {property_id} {field} disagrees with registry"
                    )
            recommendation = str(raw.get("recommendation", ""))
            metrics = raw.get("metrics", {})
            if recommendation not in ADMITTED_RECOMMENDATIONS:
                continue
            if not isinstance(metrics, dict) or int(metrics.get("semantic_checked", 0)) <= 0:
                raise ValueError(
                    f"{path}: {property_id} claims semantic admission without checks"
                )
            samples = raw.get("samples", [])
            if not isinstance(samples, list):
                raise ValueError(f"{path}: {property_id} samples must be a list")
            sample_quality: dict[tuple[str, str], int] = {}
            checks = raw.get("wikipedia_checks", [])
            semantic_judgments: list[dict[str, Any]] = []
            if isinstance(checks, list):
                for check in checks:
                    if not isinstance(check, dict):
                        continue
                    judgment = check.get("semantic_judge")
                    if (
                        isinstance(judgment, dict)
                        and judgment.get("decision") in {"pass", "reject"}
                    ):
                        semantic_judgments.append(judgment)
            valid_passes = sum(
                judgment.get("decision") == "pass"
                and not semantic_judge_schema_errors(judgment)
                for judgment in semantic_judgments
            )
            if (
                len(semantic_judgments) != int(metrics.get("semantic_checked", 0))
                or valid_passes != int(metrics.get("semantic_supported", 0))
                or valid_passes <= 0
            ):
                raise ValueError(
                    f"{path}: {property_id} semantic metrics do not match "
                    "schema-valid Wikipedia judgments"
                )
            if isinstance(checks, list):
                for check in checks:
                    if not isinstance(check, dict):
                        continue
                    key = (
                        str(check.get("source_qid", "")),
                        str(check.get("source_title", "")),
                    )
                    quality = 0
                    if check.get("status") == "link_supported":
                        quality = 1
                    semantic = check.get("semantic_judge")
                    if (
                        isinstance(semantic, dict)
                        and semantic.get("decision") == "pass"
                        and not semantic_judge_schema_errors(semantic)
                    ):
                        quality = 2
                    sample_quality[key] = max(sample_quality.get(key, 0), quality)
            bucket = collected.setdefault(property_id, {
                "recommendations": [], "hashes": [], "metrics": [], "samples": [],
            })
            bucket["recommendations"].append(recommendation)
            bucket["hashes"].append(profile_hash)
            bucket["metrics"].append(dict(metrics))
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                copied = dict(sample)
                key = (
                    str(copied.get("source_qid", "")),
                    str(copied.get("source_title", "")),
                )
                copied["_profile_evidence_quality"] = sample_quality.get(key, 0)
                bucket["samples"].append(copied)
    admissions = []
    for property_id in sorted(collected):
        bucket = collected[property_id]
        samples_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for sample in bucket["samples"]:
            key = (str(sample.get("source_qid", "")), str(sample.get("source_title", "")))
            if not re.fullmatch(r"Q[1-9]\d*", key[0]) or not key[1]:
                continue
            previous = samples_by_key.get(key)
            if (
                previous is None
                or int(sample.get("_profile_evidence_quality", 0))
                > int(previous.get("_profile_evidence_quality", 0))
            ):
                samples_by_key[key] = sample
        samples = sorted(
            samples_by_key.values(),
            key=lambda row: (
                -int(row.get("_profile_evidence_quality", 0)),
                str(row.get("source_title", "")).casefold(),
            ),
        )
        admissions.append(RelationAdmission(
            spec=by_property[property_id],
            recommendations=tuple(sorted(set(bucket["recommendations"]))),
            profile_sha256s=tuple(sorted(set(bucket["hashes"]))),
            metrics=tuple(bucket["metrics"]),
            source_samples=tuple(samples),
        ))
    if not admissions:
        qualifier = "active " if require_active else ""
        raise ValueError(
            f"profiles contain no {qualifier}semantically validated relation"
        )
    return tuple(admissions)


def _snak_date(snak: dict[str, Any]) -> str | None:
    value = snak.get("datavalue", {}).get("value", {})
    raw = value.get("time") if isinstance(value, dict) else None
    if not isinstance(raw, str):
        return None
    match = re.match(r"^\+?(\d{4})-(\d{2})-(\d{2})T", raw)
    if not match or "00" in match.groups()[1:]:
        return None
    try:
        return date.fromisoformat("-".join(match.groups())).isoformat()
    except ValueError:
        return None


def _qualifier_date(claim: dict[str, Any], property_id: str) -> str | None:
    raw_values = claim.get("qualifiers", {}).get(property_id, [])
    values = sorted({
        parsed for snak in raw_values if isinstance(snak, dict)
        if (parsed := _snak_date(snak)) is not None
    })
    return values[-1] if len(values) == 1 else None


def _claim_target_qid(claim: dict[str, Any]) -> str | None:
    value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
    qid = value.get("id") if isinstance(value, dict) else None
    return str(qid) if isinstance(qid, str) and re.fullmatch(r"Q[1-9]\d*", qid) else None


@dataclass(frozen=True)
class ClaimRecord:
    target_qid: str
    rank: str
    start: str | None
    end: str | None
    point: str | None

    @property
    def qualifier_dates(self) -> dict[str, str]:
        return {
            key: value for key, value in (
                ("P580", self.start), ("P582", self.end), ("P585", self.point),
            ) if value is not None
        }


def _claim_records(claims: Any) -> list[ClaimRecord]:
    if not isinstance(claims, list):
        return []
    records = []
    for claim in claims:
        if not isinstance(claim, dict) or claim.get("rank") == "deprecated":
            continue
        target = _claim_target_qid(claim)
        if target is None:
            continue
        records.append(ClaimRecord(
            target_qid=target,
            rank=str(claim.get("rank", "normal")),
            start=_qualifier_date(claim, "P580"),
            end=_qualifier_date(claim, "P582"),
            point=_qualifier_date(claim, "P585"),
        ))
    return records


def select_claim_at(
    claims: Any, spec: TemporalRelationSpec, as_of: str,
) -> ClaimRecord | None:
    """Apply a declared cardinality policy; ambiguous values fail closed."""
    selected_date = _strict_date(as_of)
    records = _claim_records(claims)
    if spec.selection_policy == "latest_point":
        eligible = [
            record for record in records
            if record.point is not None and _strict_date(record.point) <= selected_date
        ]
        if not eligible:
            return None
        latest = max(str(record.point) for record in eligible)
        eligible = [record for record in eligible if record.point == latest]
    else:
        eligible = [
            record for record in records
            if (record.start is None or _strict_date(record.start) <= selected_date)
            and (record.end is None or _strict_date(record.end) >= selected_date)
        ]
        if spec.selection_policy == "latest_start" and eligible:
            latest = max(record.start or "0001-01-01" for record in eligible)
            eligible = [record for record in eligible if (record.start or "0001-01-01") == latest]
    preferred = [record for record in eligible if record.rank == "preferred"]
    eligible = preferred or eligible
    targets = {record.target_qid for record in eligible}
    if len(targets) != 1:
        return None
    target = next(iter(targets))
    return next(record for record in eligible if record.target_qid == target)


@dataclass(frozen=True)
class EdgeCandidate:
    spec: TemporalRelationSpec
    direction: str
    next_qid: str
    next_title: str
    event_date: str
    qualifier_dates: dict[str, str]
    kg_subject_qid: str
    kg_object_qid: str
    event_order_certificate: dict[str, Any] | None = None


@dataclass(frozen=True)
class CutoffAnchorCandidate:
    source_qid: str
    source_title: str
    target_qid: str
    target_title: str
    property_id: str
    profile_evidence_quality: int


def inverse_cutoff_anchor_query(
    target_qids: Iterable[str],
    spec: TemporalRelationSpec,
    *,
    cutoff: str,
    limit: int,
) -> str:
    values = " ".join(f"wd:{qid}" for qid in dict.fromkeys(target_qids))
    if not values:
        raise ValueError("inverse cutoff anchor query needs target QIDs")
    cutoff_start = cutoff + "T00:00:00Z"
    cutoff_end = cutoff + "T23:59:59Z"
    prop = spec.property_id
    return f"""SELECT DISTINCT ?source ?sourceArticle ?target ?targetArticle ?start ?end WHERE {{
  VALUES ?target {{ {values} }}
  ?source p:{prop} ?statement .
  ?statement ps:{prop} ?target .
  OPTIONAL {{ ?statement pq:P580 ?start . }}
  OPTIONAL {{ ?statement pq:P582 ?end . }}
  FILTER((!BOUND(?start) || ?start <= \"{cutoff_end}\"^^xsd:dateTime) &&
         (!BOUND(?end) || ?end >= \"{cutoff_start}\"^^xsd:dateTime))
  ?sourceArticle schema:about ?source ; schema:isPartOf <https://en.wikipedia.org/> .
  ?targetArticle schema:about ?target ; schema:isPartOf <https://en.wikipedia.org/> .
}} ORDER BY ?target ?source LIMIT {int(limit)}"""


def _wdqs_rows(query: str, request_get: Callable) -> list[dict[str, Any]]:
    response = None
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            candidate_response = request_get(
                WDQS_URL,
                params={"query": query, "format": "json"},
                headers={
                    "Accept": "application/sparql-results+json",
                    "User-Agent": USER_AGENT,
                },
                timeout=90,
            )
            candidate_response.raise_for_status()
            response = candidate_response
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    if response is None:
        raise RuntimeError(f"WDQS query failed: {last_error}")
    payload = response.json()
    rows = payload.get("results", {}).get("bindings", [])
    if not isinstance(rows, list):
        raise ValueError("WDQS response has no results.bindings list")
    return rows


def query_cutoff_anchor_candidates(
    admission: RelationAdmission,
    *,
    cutoff: str,
    limit: int,
    request_get: Callable = requests.get,
) -> list[CutoffAnchorCandidate]:
    spec = admission.spec
    if not spec.inverse_relative_clause:
        return []
    targets: dict[str, tuple[str, int]] = {}
    for sample in admission.source_samples:
        qid = str(sample.get("target_qid", ""))
        title = str(sample.get("target_title", "")).strip()
        if not re.fullmatch(r"Q[1-9]\d*", qid) or not title:
            continue
        quality = int(sample.get("_profile_evidence_quality", 0))
        previous = targets.get(qid)
        if previous is None or quality > previous[1]:
            targets[qid] = (title, quality)
    if not targets:
        return []
    query = inverse_cutoff_anchor_query(
        targets, spec, cutoff=cutoff, limit=limit
    )
    rows = _wdqs_rows(query, request_get)
    result = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        try:
            source_qid = _uri_id(_binding(row, "source"), "Q")
            source_title = _article_title(_binding(row, "sourceArticle"))
            target_qid = _uri_id(_binding(row, "target"), "Q")
            target_title = _article_title(_binding(row, "targetArticle"))
        except ValueError:
            continue
        if target_qid not in targets:
            continue
        key = (source_qid, target_qid, spec.property_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(CutoffAnchorCandidate(
            source_qid=source_qid, source_title=source_title,
            target_qid=target_qid, target_title=target_title,
            property_id=spec.property_id,
            profile_evidence_quality=targets[target_qid][1],
        ))
    return result


def _next_forward_record(
    claims: Any,
    spec: TemporalRelationSpec,
    *,
    after: str,
    until: str,
) -> ClaimRecord | None:
    previous = select_claim_at(claims, spec, after)
    previous_qid = previous.target_qid if previous is not None else None
    by_event: dict[str, list[ClaimRecord]] = {}
    for record in _claim_records(claims):
        event = record.point if spec.time_mode == "point" else record.start
        if spec.time_mode == "either":
            event = record.start or record.point
        if event is None or not (after < event <= until):
            continue
        by_event.setdefault(event, []).append(record)
    for event in sorted(by_event):
        targets = {record.target_qid for record in by_event[event]}
        if len(targets) != 1:
            return None
        target = next(iter(targets))
        if target == previous_qid:
            continue
        return next(record for record in by_event[event] if record.target_qid == target)
    return None


def _forward_event_order_certificate(
    claims: Any, spec: TemporalRelationSpec, *, after: str, until: str,
    selected: ClaimRecord,
) -> dict[str, Any]:
    previous = select_claim_at(claims, spec, after)
    previous_qid = previous.target_qid if previous is not None else None
    candidates = []
    for record in _claim_records(claims):
        event = record.point if spec.time_mode == "point" else record.start
        if spec.time_mode == "either":
            event = record.start or record.point
        if (
            event is not None and after < event <= until
            and record.target_qid != previous_qid
        ):
            candidates.append({
                "event_date": event, "target_qid": record.target_qid,
            })
    selected_event = selected.point if spec.time_mode == "point" else selected.start
    if spec.time_mode == "either":
        selected_event = selected.start or selected.point
    assert selected_event is not None
    return build_event_order_certificate(
        boundary_event_date=after, selected_event_date=selected_event,
        selected_target_qid=selected.target_qid, candidate_events=candidates,
        coverage_end=until, source="wikidata_entity_claims",
    )


def inverse_transition_query(
    current_qid: str,
    specs: Iterable[TemporalRelationSpec],
    *,
    after: str,
    until: str,
    limit: int,
) -> str:
    rows = "\n    ".join(
        f"(wd:{spec.property_id} p:{spec.property_id} ps:{spec.property_id})"
        for spec in specs if spec.inverse_relative_clause
    )
    if not rows:
        raise ValueError("inverse transition query needs inverse-enabled relations")
    after_value = after + "T23:59:59Z"
    until_value = until + "T23:59:59Z"
    return f"""SELECT DISTINCT ?property ?source ?sourceArticle ?start ?point WHERE {{
  VALUES (?property ?claim ?statementProperty) {{
    {rows}
  }}
  ?source ?claim ?statement .
  ?statement ?statementProperty wd:{current_qid} .
  OPTIONAL {{ ?statement pq:P580 ?start . }}
  OPTIONAL {{ ?statement pq:P585 ?point . }}
  FILTER((BOUND(?start) && ?start > \"{after_value}\"^^xsd:dateTime &&
          ?start <= \"{until_value}\"^^xsd:dateTime) ||
         (BOUND(?point) && ?point > \"{after_value}\"^^xsd:dateTime &&
          ?point <= \"{until_value}\"^^xsd:dateTime))
  ?sourceArticle schema:about ?source ; schema:isPartOf <https://en.wikipedia.org/> .
}} ORDER BY ?start ?point ?property ?source LIMIT {int(limit)}"""


def query_inverse_candidates(
    current_qid: str,
    specs: tuple[TemporalRelationSpec, ...],
    *,
    after: str,
    until: str,
    limit: int,
    request_get: Callable = requests.get,
) -> list[EdgeCandidate]:
    inverse_specs = tuple(spec for spec in specs if spec.inverse_relative_clause)
    if not inverse_specs:
        return []
    query = inverse_transition_query(
        current_qid, inverse_specs, after=after, until=until, limit=limit
    )
    rows = _wdqs_rows(query, request_get)
    if len(rows) >= limit:
        # A LIMIT-saturated response cannot prove that the earliest candidate set
        # is exhaustive, so it cannot support a "next" edge.
        return []
    by_property = {spec.property_id: spec for spec in inverse_specs}
    grouped: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for row in rows:
        try:
            property_id = _uri_id(_binding(row, "property"), "P")
            source_qid = _uri_id(_binding(row, "source"), "Q")
            source_title = _article_title(_binding(row, "sourceArticle"))
        except ValueError:
            continue
        spec = by_property.get(property_id)
        if spec is None:
            continue
        start = _wikidata_literal_date(_binding(row, "start"))
        point = _wikidata_literal_date(_binding(row, "point"))
        event = point if spec.time_mode == "point" else start
        if spec.time_mode == "either":
            event = start or point
        if event is None or not (after < event <= until):
            continue
        grouped.setdefault((property_id, event), []).append((source_qid, source_title))
    candidates = []
    for property_id, spec in sorted(by_property.items()):
        events = sorted(event for prop, event in grouped if prop == property_id)
        if not events:
            continue
        event = events[0]
        sources = sorted(set(grouped[(property_id, event)]))
        if len({qid for qid, _ in sources}) != 1:
            continue
        source_qid, source_title = sources[0]
        qualifier = "P585" if spec.time_mode == "point" else "P580"
        candidate_events = [
            {"event_date": candidate_event, "target_qid": source_qid_value}
            for (prop, candidate_event), values in grouped.items()
            if prop == property_id
            for source_qid_value, _ in sorted(set(values))
        ]
        candidates.append(EdgeCandidate(
            spec=spec, direction="inverse", next_qid=source_qid,
            next_title=source_title, event_date=event,
            qualifier_dates={qualifier: event}, kg_subject_qid=source_qid,
            kg_object_qid=current_qid,
            event_order_certificate=build_event_order_certificate(
                boundary_event_date=after, selected_event_date=event,
                selected_target_qid=source_qid,
                candidate_events=candidate_events, coverage_end=until,
                source="wikidata_sparql_inverse_transition_query",
                source_query_sha256=hashlib.sha256(query.encode()).hexdigest(),
            ),
        ))
    return candidates


def _wikidata_literal_date(value: str) -> str | None:
    if not value:
        return None
    normalized = value.lstrip("+")
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})T", normalized)
    if not match or "00" in match.groups()[1:]:
        return None
    try:
        return date.fromisoformat("-".join(match.groups())).isoformat()
    except ValueError:
        return None


def _target_title(entity: dict[str, Any], lang: str) -> str | None:
    link = entity.get("sitelinks", {}).get(f"{lang}wiki", {})
    value = link.get("title") if isinstance(link, dict) else None
    return " ".join(str(value).split()) if value else None


def _fetch_entities(backend: Any, qids: Iterable[str]) -> dict[str, dict[str, Any]]:
    unique = list(dict.fromkeys(qids))
    result: dict[str, dict[str, Any]] = {}
    for index in range(0, len(unique), 50):
        result.update(backend.get_wikidata_entities(unique[index:index + 50]))
    return result


def _has_link(page: Any, target_title: str) -> bool:
    return any(link.target.casefold() == target_title.casefold() for link in page.links)


def _evidence_window(page: Any, target_title: str) -> tuple[str, str] | None:
    links = [
        link for link in page.links
        if link.target.casefold() == target_title.casefold()
    ]
    if not links:
        return None
    selected = max(links, key=lambda link: len(link.anchor))
    alias = selected.anchor
    token = f"[{selected.anchor} -> {selected.target}]"
    blocks = page.content.split("\n\n")
    for index, block in enumerate(blocks):
        if token.casefold() not in block.casefold():
            continue
        excerpt = _plain_text("\n\n".join(
            blocks[max(0, index - 2):min(len(blocks), index + 3)]
        ))
        if _fold(alias) in _fold(excerpt):
            return excerpt, alias
    plain = _plain_text(page.content)
    position = plain.casefold().find(alias.casefold())
    if position < 0:
        return None
    return (
        plain[max(0, position - 120):position + len(alias) + 220].strip(),
        alias,
    )


@dataclass(frozen=True)
class WikipediaSupport:
    as_of: str
    evidence: str
    alias: str
    source_revision_id: int
    target_revision_id: int
    prior_target_visible: bool = False
    prior_evidence: str | None = None
    prior_revision_id: int | None = None


def verify_initial_edge(
    backend: Any, *, source_title: str, target_title: str, cutoff: str,
) -> WikipediaSupport | None:
    source = backend.fetch_page(source_title, as_of=cutoff)
    evidence = _evidence_window(source, target_title)
    if evidence is None or not _has_link(source, target_title):
        return None
    excerpt, alias = evidence
    target = backend.fetch_page(target_title, as_of=cutoff)
    return WikipediaSupport(
        as_of=cutoff, evidence=excerpt, alias=alias,
        source_revision_id=source.revision_id,
        target_revision_id=target.revision_id,
    )


def verify_later_edge(
    backend: Any,
    *,
    source_title: str,
    target_title: str,
    previous_as_of: str,
    event_date: str,
    until: str,
) -> WikipediaSupport | None:
    prior = backend.fetch_page(source_title, as_of=previous_as_of)
    prior_link_evidence = _evidence_window(prior, target_title)
    prior_target_visible = (
        _has_link(prior, target_title)
        or _fold(target_title) in _fold(prior.content)
    )
    event = _strict_date(event_date)
    upper = _strict_date(until)
    for offset in DATE_OFFSETS:
        candidate_date = event + timedelta(days=offset)
        if candidate_date > upper:
            continue
        as_of = candidate_date.isoformat()
        if as_of <= previous_as_of:
            continue
        source = backend.fetch_page(source_title, as_of=as_of)
        evidence = _evidence_window(source, target_title)
        if (
            source.revision_id == prior.revision_id
            or evidence is None
            or not _has_link(source, target_title)
        ):
            continue
        excerpt, alias = evidence
        prior_target_visible = (
            prior_target_visible or _fold(alias) in _fold(prior.content)
        )
        prior_evidence = None
        if prior_target_visible:
            if prior_link_evidence is not None:
                prior_evidence = prior_link_evidence[0]
            else:
                folded = prior.content.casefold()
                positions = [
                    folded.find(value.casefold())
                    for value in (target_title, alias) if value
                ]
                positions = [position for position in positions if position >= 0]
                if positions:
                    position = min(positions)
                    prior_evidence = prior.content[
                        max(0, position - 120):position + max(len(target_title), len(alias)) + 220
                    ].strip()
        target = backend.fetch_page(target_title, as_of=as_of)
        return WikipediaSupport(
            as_of=as_of, evidence=excerpt, alias=alias,
            source_revision_id=source.revision_id,
            target_revision_id=target.revision_id,
            prior_target_visible=prior_target_visible,
            prior_evidence=prior_evidence,
            prior_revision_id=prior.revision_id,
        )
    return None


class TemporalEdgeContrastJudge:
    """Cached semantic before/after judge for targets visible before a role change."""

    def __init__(
        self,
        model: str,
        *,
        min_confidence: float = 0.8,
        cache_path: str | None = None,
        call_model_fn: Callable[..., str] = call_model,
    ):
        self.model = model
        self.min_confidence = min_confidence
        self.call_model_fn = call_model_fn
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if cache_path:
            self._conn = sqlite3.connect(cache_path, check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS edge_contrast_judge_cache ("
                "cache_key TEXT PRIMARY KEY, model TEXT NOT NULL, "
                "result_json TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            with self._lock:
                self._conn.close()
                self._conn = None

    def _load(self, key: str) -> dict[str, Any] | None:
        if self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT result_json FROM edge_contrast_judge_cache WHERE cache_key=?",
                (key,),
            ).fetchone()
        return dict(json.loads(row[0])) if row else None

    def _store(self, key: str, result: dict[str, Any]) -> None:
        if self._conn is None:
            return
        stored = {field: value for field, value in result.items() if field != "cache_hit"}
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO edge_contrast_judge_cache("
                "cache_key,model,result_json,created_at) VALUES(?,?,?,?)",
                (
                    key, self.model,
                    json.dumps(stored, ensure_ascii=False, sort_keys=True),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()

    def judge(
        self,
        spec: TemporalRelationSpec,
        candidate: EdgeCandidate,
        support: WikipediaSupport,
        *,
        source_title: str,
        target_title: str,
        previous_as_of: str,
    ) -> dict[str, Any]:
        before_evidence = support.prior_evidence or ""
        relation_clause = (
            spec.inverse_relative_clause
            if candidate.direction == "inverse"
            else spec.later_relative_clause or spec.relative_clause
        )
        visible = {
            "property_id": spec.property_id, "relation_label": spec.label,
            "direction": candidate.direction, "relation_clause": relation_clause,
            "semantic_guidance": spec.semantic_guidance,
            "source_title": source_title, "target_title": target_title,
            "before_as_of": previous_as_of, "after_as_of": support.as_of,
            "before_revision_id": support.prior_revision_id,
            "after_revision_id": support.source_revision_id,
            "before_evidence": before_evidence, "after_evidence": support.evidence,
        }
        key = hashlib.sha256(json.dumps({
            "prompt_version": 2, "model": self.model,
            "min_confidence": self.min_confidence, "input": visible,
        }, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cached = self._load(key)
        if cached is not None:
            return {**cached, "cache_hit": True}
        prompt = f"""Audit whether one temporal relation became true between two Wikipedia revisions.

Input:
{json.dumps(visible, ensure_ascii=False, indent=2)}

The target entity is visible in both excerpts, possibly in a different historical role.
Pass only when the AFTER excerpt explicitly expresses the named relation and direction,
while the BEFORE excerpt does not express that same relation between this source and target.
Mere co-occurrence, a prior unrelated role, or a historical mention is not the relation.
Use only these excerpts and the relation-specific guidance. Do not use outside knowledge.

Return one JSON object with decision (pass/reject), confidence (0..1), checks
(after_relation_supported, before_relation_absent, direction_correct), and reason.
"""
        raw, response = call_json_model(self.model, prompt, self.call_model_fn)
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        decision = str(raw.get("decision", "reject"))
        checks = raw.get("checks")
        schema_errors = []
        if not isinstance(checks, dict):
            schema_errors.append("checks_not_object")
        else:
            for check_name in EDGE_CONTRAST_CHECKS:
                if checks.get(check_name) is not True:
                    schema_errors.append(f"check_not_true:{check_name}")
        if not str(raw.get("reason", "")).strip():
            schema_errors.append("missing_reason")
        if decision != "pass" or confidence < self.min_confidence or schema_errors:
            decision = "reject"
        result = {
            **raw, "decision": decision, "confidence": confidence,
            "schema_gate_errors": schema_errors,
            "raw_response": response, "cache_hit": False,
        }
        self._store(key, result)
        return result


@dataclass(frozen=True)
class BeamState:
    hops: tuple[dict[str, Any], ...]
    entity_qids: tuple[str, ...]
    current_qid: str
    current_title: str
    current_as_of: str
    current_event_date: str
    property_steps: tuple[dict[str, str], ...]
    families: frozenset[str]
    score: float
    anchor_views: int
    last_answer_kind: str


class RenewableQuestionEngine:
    """Deterministic bounded beam search over admitted temporal relations."""

    def __init__(
        self,
        *,
        model_id: str,
        until: str,
        registry: TemporalRelationRegistry,
        admissions: tuple[RelationAdmission, ...],
        backend: Any,
        popularity: PopularitySnapshot | None = None,
        inverse_lookup_fn: Callable[..., list[EdgeCandidate]] = query_inverse_candidates,
        cutoff_anchor_lookup_fn: Callable[..., list[CutoffAnchorCandidate]] = (
            query_cutoff_anchor_candidates
        ),
        edge_contrast_judge: TemporalEdgeContrastJudge | None = None,
        min_hops: int = 4,
        max_hops: int = 5,
        min_families: int = 2,
        beam_width: int = 32,
        max_anchors: int = 100,
        max_expansions_per_state: int = 8,
        inverse_limit: int = 100,
        allow_repeated_properties: bool = False,
        max_property_occurrences: int = 3,
        terminal_attribute_tails: bool = True,
        min_temporal_hops: int = 3,
        tail_state_cap: int = 12,
        allow_easy_tails: bool = False,
    ):
        cutoff = get_model_cutoff(model_id)
        if _strict_date(until) <= _strict_date(cutoff.cutoff_date):
            raise ValueError("until must be after the tested model cutoff")
        if not 2 <= min_hops <= max_hops <= 6:
            raise ValueError("hop bounds must satisfy 2 <= min_hops <= max_hops <= 6")
        if not 1 <= min_families <= max_hops:
            raise ValueError("min_families must be between 1 and max_hops")
        if min_families > min_hops:
            raise ValueError("min_families cannot exceed min_hops")
        if min_hops - 1 < 2:
            raise ValueError("renewable questions require at least two temporal switches")
        if not 3 <= min_temporal_hops <= max_hops:
            raise ValueError("min_temporal_hops must be between 3 and max_hops")
        if max_property_occurrences > max_hops:
            raise ValueError("max_property_occurrences cannot exceed max_hops")
        for value, label in (
            (beam_width, "beam_width"), (max_anchors, "max_anchors"),
            (max_expansions_per_state, "max_expansions_per_state"),
            (inverse_limit, "inverse_limit"),
            (max_property_occurrences, "max_property_occurrences"),
            (tail_state_cap, "tail_state_cap"),
        ):
            if value <= 0:
                raise ValueError(f"{label} must be > 0")
        self.model_id = cutoff.model_id
        self.cutoff = cutoff.cutoff_date
        self.until = until
        self.registry = registry
        self.admissions = admissions
        self.admission_by_property = {
            admission.spec.property_id: admission for admission in admissions
        }
        self.specs = tuple(admission.spec for admission in admissions)
        self.backend = backend
        self.popularity = popularity
        self.inverse_lookup_fn = inverse_lookup_fn
        self.cutoff_anchor_lookup_fn = cutoff_anchor_lookup_fn
        self.edge_contrast_judge = edge_contrast_judge
        self.min_hops = min_hops
        self.max_hops = max_hops
        self.min_families = min_families
        self.beam_width = beam_width
        self.max_anchors = max_anchors
        self.max_expansions_per_state = max_expansions_per_state
        self.inverse_limit = inverse_limit
        self.allow_repeated_properties = allow_repeated_properties
        self.max_property_occurrences = (
            max_hops if allow_repeated_properties else max_property_occurrences
        )
        self.terminal_attribute_tails = terminal_attribute_tails
        self.min_temporal_hops = min_temporal_hops
        self.tail_state_cap = tail_state_cap
        self.allow_easy_tails = allow_easy_tails
        self.packets: list[dict[str, Any]] = []
        self._entity_cache: dict[str, dict[str, Any]] = {}
        self._inverse_cache: dict[tuple[str, str], list[EdgeCandidate]] = {}

    def _entities(self, qids: Iterable[str]) -> dict[str, dict[str, Any]]:
        unique = list(dict.fromkeys(qids))
        missing = [qid for qid in unique if qid not in self._entity_cache]
        if missing:
            self._entity_cache.update(_fetch_entities(self.backend, missing))
        return {
            qid: self._entity_cache[qid]
            for qid in unique if qid in self._entity_cache
        }

    def _initial_states(self) -> list[BeamState]:
        anchors_by_property: dict[
            str,
            list[tuple[
                int, int, str, str, RelationAdmission,
                str | None, str | None, str,
            ]],
        ] = {}
        seen: set[tuple[str, str, str | None]] = set()
        for admission in self.admissions:
            for sample in admission.source_samples:
                qid = str(sample.get("source_qid", ""))
                title = str(sample.get("source_title", "")).strip()
                key: tuple[str, str, str | None] = (
                    qid, admission.spec.property_id, None,
                )
                if not title or key in seen:
                    continue
                seen.add(key)
                views = self.popularity.views(title) if self.popularity else 0
                quality = int(sample.get("_profile_evidence_quality", 0))
                anchors_by_property.setdefault(admission.spec.property_id, []).append(
                    (quality, views, title, qid, admission, None, None, "profile_source")
                )
            try:
                inverse_anchors = self.cutoff_anchor_lookup_fn(
                    admission, cutoff=self.cutoff,
                    limit=max(self.max_anchors * 2, 50),
                )
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                self.packets.append({
                    "stage": "anchor_discovery", "status": "infrastructure_error",
                    "property_id": admission.spec.property_id,
                    "direction": "inverse_from_profile_target", "error": str(exc),
                })
                inverse_anchors = []
            print(
                f"[anchor discovery] {admission.spec.property_id} "
                f"profile_sources={len(admission.source_samples)} "
                f"inverse_cutoff={len(inverse_anchors)}",
                flush=True,
            )
            for candidate in inverse_anchors:
                key = (
                    candidate.source_qid, admission.spec.property_id,
                    candidate.target_qid,
                )
                if key in seen:
                    continue
                seen.add(key)
                views = (
                    self.popularity.views(candidate.source_title)
                    if self.popularity else 0
                )
                anchors_by_property.setdefault(admission.spec.property_id, []).append((
                    candidate.profile_evidence_quality, views,
                    candidate.source_title, candidate.source_qid, admission,
                    candidate.target_qid, candidate.target_title,
                    "inverse_profile_target",
                ))
        for values in anchors_by_property.values():
            values.sort(key=lambda row: (-row[0], -row[1], row[2].casefold(), row[3]))
        anchors: list[tuple[
            int, int, str, str, RelationAdmission,
            str | None, str | None, str,
        ]] = []
        property_ids = sorted(anchors_by_property)
        offset = 0
        while len(anchors) < self.max_anchors:
            added = False
            for property_id in property_ids:
                values = anchors_by_property[property_id]
                if offset < len(values):
                    anchors.append(values[offset])
                    added = True
                    if len(anchors) >= self.max_anchors:
                        break
            if not added:
                break
            offset += 1
        source_entities = self._entities(row[3] for row in anchors)
        selections: list[tuple[
            int, int, str, str, RelationAdmission, ClaimRecord,
            str | None, str,
        ]] = []
        for (
            quality, views, title, qid, admission,
            expected_target_qid, expected_target_title, origin,
        ) in anchors:
            entity = source_entities.get(qid, {})
            claims = entity.get("claims", {}).get(admission.spec.property_id, [])
            selected = select_claim_at(claims, admission.spec, self.cutoff)
            if selected is None:
                self.packets.append({
                    "stage": "anchor", "status": "reject_ambiguous_or_missing_cutoff_value",
                    "source_qid": qid, "source_title": title,
                    "property_id": admission.spec.property_id,
                    "anchor_origin": origin,
                })
                continue
            if (
                expected_target_qid is not None
                and selected.target_qid != expected_target_qid
            ):
                self.packets.append({
                    "stage": "anchor", "status": "reject_cutoff_target_mismatch",
                    "source_qid": qid, "source_title": title,
                    "property_id": admission.spec.property_id,
                    "anchor_origin": origin,
                    "expected_target_qid": expected_target_qid,
                    "selected_target_qid": selected.target_qid,
                })
                continue
            selections.append((
                quality, views, title, qid, admission, selected,
                expected_target_title, origin,
            ))
        targets = self._entities(row[5].target_qid for row in selections)
        states = []
        lang = str(getattr(self.backend, "lang", "en"))
        for (
            quality, views, title, qid, admission, selected,
            expected_target_title, origin,
        ) in selections:
            target_title = _target_title(targets.get(selected.target_qid, {}), lang)
            packet = {
                "stage": "anchor", "source_qid": qid, "source_title": title,
                "property_id": admission.spec.property_id,
                "target_qid": selected.target_qid, "target_title": target_title,
                "expected_target_title": expected_target_title,
                "anchor_origin": origin,
            }
            if not target_title or selected.target_qid == qid:
                self.packets.append({**packet, "status": "reject_missing_sitelink_or_cycle"})
                continue
            try:
                support = verify_initial_edge(
                    self.backend, source_title=title, target_title=target_title,
                    cutoff=self.cutoff,
                )
            except WikipediaError as exc:
                self.packets.append({**packet, "status": "infrastructure_error", "error": str(exc)})
                continue
            if support is None:
                self.packets.append({**packet, "status": "reject_no_cutoff_wikipedia_support"})
                continue
            spec = admission.spec
            clause = (
                f"{spec.relative_clause} at the tested model's registered knowledge cutoff"
            )
            structured = {
                "source": "wikidata_time_qualified_statement",
                "property_id": spec.property_id, "direction": "forward",
                "kg_subject_qid": qid, "kg_object_qid": selected.target_qid,
                "qualifier_dates": selected.qualifier_dates,
                "selection_policy": spec.selection_policy,
                "relation_profile_sha256s": list(admission.profile_sha256s),
                "relation_profile_recommendations": list(admission.recommendations),
            }
            hop = {
                "source_title": title, "target_title": target_title,
                "relation": f"{spec.label} at registered knowledge cutoff",
                "relative_clause": clause, "as_of": self.cutoff,
                "evidence": support.evidence, "target_aliases": [support.alias],
                "incoming_time_policy": "advance_required",
                "property_id": spec.property_id, "relation_family": spec.family,
                "structured_evidence": structured,
            }
            score = math.log1p(max(0, views)) + 4.0 * admission.semantic_support_rate
            states.append(BeamState(
                hops=(hop,), entity_qids=(qid, selected.target_qid),
                current_qid=selected.target_qid, current_title=target_title,
                current_as_of=self.cutoff,
                current_event_date=self.cutoff,
                property_steps=({"property_id": spec.property_id, "direction": "forward"},),
                families=frozenset({spec.family}), score=score,
                anchor_views=views, last_answer_kind=spec.answer_kind,
            ))
            self.packets.append({
                **packet, "status": "pass", "supported_as_of": support.as_of,
                "profile_evidence_quality": quality,
                "source_revision_id": support.source_revision_id,
                "target_revision_id": support.target_revision_id,
            })
        states.sort(key=self._state_sort_key)
        return states[:self.beam_width]

    @staticmethod
    def _state_sort_key(state: BeamState) -> tuple[Any, ...]:
        return (
            -len(state.families), -state.score, state.current_event_date,
            state.current_as_of,
            state.current_title.casefold(), state.entity_qids,
        )

    def _forward_candidates(self, state: BeamState) -> list[EdgeCandidate]:
        entity = self._entities([state.current_qid]).get(state.current_qid, {})
        claims_by_property = entity.get("claims", {})
        pending: list[tuple[TemporalRelationSpec, ClaimRecord]] = []
        for spec in self.specs:
            record = _next_forward_record(
                claims_by_property.get(spec.property_id, []), spec,
                after=state.current_event_date, until=self.until,
            )
            if record is not None:
                pending.append((spec, record))
        entities = self._entities(record.target_qid for _, record in pending)
        lang = str(getattr(self.backend, "lang", "en"))
        result = []
        for spec, record in pending:
            title = _target_title(entities.get(record.target_qid, {}), lang)
            if title is None:
                continue
            event = record.point if spec.time_mode == "point" else record.start
            if spec.time_mode == "either":
                event = record.start or record.point
            if event is None:
                continue
            result.append(EdgeCandidate(
                spec=spec, direction="forward", next_qid=record.target_qid,
                next_title=title, event_date=event,
                qualifier_dates=record.qualifier_dates,
                kg_subject_qid=state.current_qid,
                kg_object_qid=record.target_qid,
                event_order_certificate=_forward_event_order_certificate(
                    claims_by_property.get(spec.property_id, []), spec,
                    after=state.current_event_date, until=self.until,
                    selected=record,
                ),
            ))
        return result

    def _inverse_candidates(self, state: BeamState) -> list[EdgeCandidate]:
        key = (state.current_qid, state.current_event_date)
        if key not in self._inverse_cache:
            self._inverse_cache[key] = self.inverse_lookup_fn(
                state.current_qid, self.specs, after=state.current_event_date,
                until=self.until, limit=self.inverse_limit,
            )
        return list(self._inverse_cache[key])

    def _candidate_sort_key(
        self, state: BeamState, candidate: EdgeCandidate
    ) -> tuple[Any, ...]:
        return (
            candidate.spec.family in state.families,
            candidate.event_date,
            candidate.spec.property_id,
            candidate.direction,
            candidate.next_title.casefold(),
        )

    def _expand(self, state: BeamState) -> list[BeamState]:
        try:
            candidates = self._forward_candidates(state) + self._inverse_candidates(state)
        except (WikipediaError, requests.RequestException, ValueError) as exc:
            self.packets.append({
                "stage": "expand", "status": "infrastructure_error",
                "source_qid": state.current_qid, "source_title": state.current_title,
                "as_of": state.current_as_of, "error": str(exc),
            })
            return []
        unique: dict[tuple[str, str, str, str], EdgeCandidate] = {}
        for candidate in candidates:
            key = (
                candidate.spec.property_id, candidate.direction,
                candidate.next_qid, candidate.event_date,
            )
            unique[key] = candidate
        candidates = sorted(
            unique.values(), key=lambda candidate: self._candidate_sort_key(state, candidate)
        )[:self.max_expansions_per_state]
        next_states = []
        property_counts: dict[str, int] = {}
        for step in state.property_steps:
            property_id = step["property_id"]
            property_counts[property_id] = property_counts.get(property_id, 0) + 1
        for candidate in candidates:
            packet = {
                "stage": "expand", "source_qid": state.current_qid,
                "source_title": state.current_title, "prior_as_of": state.current_as_of,
                "prior_event_date": state.current_event_date,
                "property_id": candidate.spec.property_id,
                "direction": candidate.direction, "target_qid": candidate.next_qid,
                "target_title": candidate.next_title, "event_date": candidate.event_date,
            }
            if (
                property_counts.get(candidate.spec.property_id, 0)
                >= self.max_property_occurrences
            ):
                self.packets.append({
                    **packet, "status": "reject_property_occurrence_cap",
                    "max_property_occurrences": self.max_property_occurrences,
                })
                continue
            if candidate.next_qid in state.entity_qids:
                self.packets.append({**packet, "status": "reject_entity_cycle"})
                continue
            try:
                support = verify_later_edge(
                    self.backend, source_title=state.current_title,
                    target_title=candidate.next_title,
                    previous_as_of=state.current_as_of,
                    event_date=candidate.event_date, until=self.until,
                )
            except WikipediaError as exc:
                self.packets.append({**packet, "status": "infrastructure_error", "error": str(exc)})
                continue
            if support is None:
                self.packets.append({**packet, "status": "reject_no_novel_wikipedia_support"})
                continue
            spec = candidate.spec
            contrast_result = None
            if support.prior_target_visible:
                if self.edge_contrast_judge is None or not support.prior_evidence:
                    self.packets.append({
                        **packet,
                        "status": "reject_prior_target_requires_semantic_contrast",
                    })
                    continue
                try:
                    contrast_result = self.edge_contrast_judge.judge(
                        spec, candidate, support,
                        source_title=state.current_title,
                        target_title=candidate.next_title,
                        previous_as_of=state.current_as_of,
                    )
                except Exception as exc:
                    self.packets.append({
                        **packet, "status": "infrastructure_error",
                        "error": f"edge contrast judge: {exc}",
                    })
                    continue
                if contrast_result.get("decision") != "pass":
                    self.packets.append({
                        **packet, "status": "reject_prior_relation_present_or_ambiguous",
                        "edge_contrast_judge": contrast_result,
                    })
                    continue
            if candidate.direction == "inverse":
                clause = spec.inverse_relative_clause
                answer_kind = spec.inverse_answer_kind
            else:
                clause = spec.later_relative_clause or (
                    f"the next {spec.label} of {{source}} after that"
                )
                answer_kind = spec.answer_kind
            admission = self.admission_by_property[spec.property_id]
            structured_evidence: dict[str, Any] = {
                "source": "wikidata_time_qualified_statement",
                "property_id": spec.property_id,
                "direction": candidate.direction,
                "kg_subject_qid": candidate.kg_subject_qid,
                "kg_object_qid": candidate.kg_object_qid,
                "event_date": candidate.event_date,
                "temporal_operator": "next_after_boundary",
                "qualifier_dates": candidate.qualifier_dates,
                "selection_policy": spec.selection_policy,
                "relation_profile_sha256s": list(admission.profile_sha256s),
                "relation_profile_recommendations": list(admission.recommendations),
                "event_order_certificate": candidate.event_order_certificate,
            }
            if contrast_result is not None:
                raw_response = str(contrast_result.get("raw_response", ""))
                assert self.edge_contrast_judge is not None
                structured_evidence["prior_relation_contrast"] = {
                    "method": "independent_llm_semantic_contrast",
                    "decision": contrast_result["decision"],
                    "confidence": contrast_result["confidence"],
                    "judge_model": self.edge_contrast_judge.model,
                    "property_id": spec.property_id,
                    "direction": candidate.direction,
                    "before_revision_id": support.prior_revision_id,
                    "after_revision_id": support.source_revision_id,
                    "before_evidence": support.prior_evidence,
                    "reason": contrast_result.get("reason"),
                    "raw_response_sha256": hashlib.sha256(raw_response.encode()).hexdigest(),
                }
            hop = {
                "source_title": state.current_title,
                "target_title": candidate.next_title,
                "relation": f"next {spec.label} after prior entity snapshot",
                "relative_clause": clause, "as_of": support.as_of,
                "evidence": support.evidence, "target_aliases": [support.alias],
                "incoming_time_policy": "advance_required",
                "property_id": spec.property_id, "relation_family": spec.family,
                "structured_evidence": structured_evidence,
            }
            novelty = 1.0 if spec.family not in state.families else 0.0
            score = state.score + 3.0 * novelty + admission.semantic_support_rate
            next_states.append(BeamState(
                hops=(*state.hops, hop),
                entity_qids=(*state.entity_qids, candidate.next_qid),
                current_qid=candidate.next_qid,
                current_title=candidate.next_title,
                current_as_of=support.as_of,
                current_event_date=candidate.event_date,
                property_steps=(*state.property_steps, {
                    "property_id": spec.property_id, "direction": candidate.direction,
                }),
                families=state.families | {spec.family}, score=score,
                anchor_views=state.anchor_views, last_answer_kind=answer_kind,
            ))
            self.packets.append({
                **packet, "status": "pass", "supported_as_of": support.as_of,
                "source_revision_id": support.source_revision_id,
                "target_revision_id": support.target_revision_id,
                "prior_target_visible": support.prior_target_visible,
                "edge_contrast_judge": contrast_result,
            })
        return next_states

    def _seed_from_state(self, state: BeamState) -> dict[str, Any]:
        identity = {
            "model_id": self.model_id,
            "cutoff_date": self.cutoff,
            "target_as_of": state.current_as_of,
            "entity_qids": list(state.entity_qids),
            "property_steps": list(state.property_steps),
            "snapshot_dates": [str(hop["as_of"]) for hop in state.hops],
            "answer_qid": state.current_qid,
        }
        fingerprint = question_fingerprint(
            model_id=self.model_id,
            cutoff_date=self.cutoff,
            target_as_of=state.current_as_of,
            entity_qids=list(state.entity_qids),
            property_steps=list(state.property_steps),
            snapshot_dates=[str(hop["as_of"]) for hop in state.hops],
            answer_qid=state.current_qid,
        )
        seed_id = f"renew_{fingerprint[:20]}"
        profile_hashes = sorted({
            profile_hash
            for hop in state.hops
            for profile_hash in hop["structured_evidence"]["relation_profile_sha256s"]
        })
        return {
            "id": seed_id, "model_id": self.model_id,
            "cutoff_date": self.cutoff, "target_as_of": state.current_as_of,
            "anchor_label": state.hops[0]["source_title"],
            "answer_kind": state.last_answer_kind,
            "category": "mixed_temporal",
            # A prefix entity is not a stale answer to the composed question.
            # No stale final answer is emitted without a dedicated old-value gate.
            "old_answer_keywords": [],
            "hops": list(state.hops),
            "selection_metadata": {
                "engine_schema_version": ENGINE_SCHEMA,
                "question_fingerprint": fingerprint,
                "semantic_identity": identity,
                "beam_score": state.score,
                "anchor_pageviews": state.anchor_views,
                "relation_families": sorted(state.families),
                "relation_profile_sha256s": profile_hashes,
                "registry_version": self.registry.registry_version,
            },
        }

    def _tail_seeds_from_state(self, state: BeamState) -> list[dict[str, Any]]:
        base_seed = self._seed_from_state(state)
        variants, tail_packets = generate_attribute_tail_variants(
            base_seed, backend=self.backend, popularity=self.popularity,
            relations=tuple(
                spec for spec in PERSON_RELATIONS
                if self.allow_easy_tails or spec.guessability != "easy"
            ),
        )
        for packet in tail_packets:
            self.packets.append({
                "stage": "terminal_attribute_tail",
                "base_seed_id": base_seed["id"], **packet,
            })
        result = []
        profile_hashes = sorted({
            profile_hash
            for hop in state.hops
            for profile_hash in hop["structured_evidence"]["relation_profile_sha256s"]
        })
        for variant in variants:
            tail = variant["hops"][-1]
            structured = tail.get("structured_evidence", {})
            answer_qid = str(structured.get("target_qid", ""))
            if not re.fullmatch(r"Q[1-9]\d*", answer_qid):
                continue
            property_steps = [
                *state.property_steps,
                {"property_id": str(tail["property_id"]), "direction": "same_snapshot"},
            ]
            entity_qids = [*state.entity_qids, answer_qid]
            snapshot_dates = [str(hop["as_of"]) for hop in variant["hops"]]
            identity = {
                "model_id": self.model_id, "cutoff_date": self.cutoff,
                "target_as_of": state.current_as_of,
                "entity_qids": entity_qids,
                "property_steps": property_steps,
                "snapshot_dates": snapshot_dates, "answer_qid": answer_qid,
            }
            fingerprint = question_fingerprint(
                model_id=self.model_id, cutoff_date=self.cutoff,
                target_as_of=state.current_as_of,
                entity_qids=entity_qids, property_steps=property_steps,
                snapshot_dates=snapshot_dates, answer_qid=answer_qid,
            )
            tail_family = str(tail.get("relation_family", "other"))
            families = sorted(state.families | {tail_family})
            if len(families) < self.min_families:
                continue
            tail_metadata = dict(variant.get("selection_metadata", {}))
            variant["id"] = f"renew_{fingerprint[:20]}"
            variant["category"] = "mixed_temporal"
            variant["selection_metadata"] = {
                "engine_schema_version": ENGINE_SCHEMA,
                "question_fingerprint": fingerprint,
                "semantic_identity": identity,
                "beam_score": state.score + 1.0,
                "anchor_pageviews": state.anchor_views,
                "relation_families": families,
                "relation_profile_sha256s": profile_hashes,
                "registry_version": self.registry.registry_version,
                "terminal_attribute_tail": tail_metadata,
                "temporal_hop_count": len(state.hops),
            }
            result.append(variant)
        return result

    def discover(self) -> list[dict[str, Any]]:
        beam = self._initial_states()
        completed: dict[str, dict[str, Any]] = {}
        print(f"[beam depth=1] states={len(beam)} completed=0", flush=True)
        for depth in range(2, self.max_hops + 1):
            expanded = []
            for state in beam:
                expanded.extend(self._expand(state))
            dedup: dict[tuple[Any, ...], BeamState] = {}
            for state in expanded:
                key = (state.entity_qids, tuple(
                    (step["property_id"], step["direction"])
                    for step in state.property_steps
                ), state.current_as_of)
                previous = dedup.get(key)
                if previous is None or state.score > previous.score:
                    dedup[key] = state
            beam = sorted(dedup.values(), key=self._state_sort_key)[:self.beam_width]
            if depth >= self.min_hops and not self.terminal_attribute_tails:
                for state in beam:
                    if len(state.families) < self.min_families:
                        continue
                    seed = self._seed_from_state(state)
                    fingerprint = seed["selection_metadata"]["question_fingerprint"]
                    completed[fingerprint] = seed
            if (
                self.terminal_attribute_tails
                and depth >= self.min_temporal_hops
                and self.min_hops <= depth + 1 <= self.max_hops
            ):
                for state in beam[:self.tail_state_cap]:
                    for seed in self._tail_seeds_from_state(state):
                        fingerprint = seed["selection_metadata"]["question_fingerprint"]
                        completed[fingerprint] = seed
            print(
                f"[beam depth={depth}] states={len(beam)} completed={len(completed)}",
                flush=True,
            )
            if not beam:
                break
        return sorted(
            completed.values(),
            key=lambda seed: (
                -len(seed["selection_metadata"]["relation_families"]),
                -float(seed["selection_metadata"]["beam_score"]),
                seed["id"],
            ),
        )


def select_diverse_seeds(
    seeds: list[dict[str, Any]],
    *,
    max_total: int,
    max_per_anchor: int = 2,
    max_per_property_sequence: int = 1,
    max_per_family: int | None = None,
    max_per_property: int | None = None,
) -> list[dict[str, Any]]:
    """Greedy coverage sampler with hard anchor, relation, and domain caps."""
    if min(max_total, max_per_anchor, max_per_property_sequence) <= 0:
        raise ValueError("diversity limits must be > 0")
    remaining = list(seeds)
    selected: list[dict[str, Any]] = []
    anchor_counts: dict[str, int] = {}
    signature_counts: dict[tuple[tuple[str, str], ...], int] = {}
    family_counts: dict[str, int] = {}
    property_counts: dict[str, int] = {}
    while remaining and len(selected) < max_total:
        ranked = []
        for seed in remaining:
            anchor = str(seed["anchor_label"]).casefold()
            identity = seed["selection_metadata"]["semantic_identity"]
            signature = tuple(
                (str(step["property_id"]), str(step["direction"]))
                for step in identity["property_steps"]
            )
            if anchor_counts.get(anchor, 0) >= max_per_anchor:
                continue
            if signature_counts.get(signature, 0) >= max_per_property_sequence:
                continue
            families = seed["selection_metadata"]["relation_families"]
            unique_families = {str(value) for value in families}
            unique_properties = {property_id for property_id, _ in signature}
            if max_per_family is not None and any(
                family_counts.get(value, 0) >= max_per_family
                for value in unique_families
            ):
                continue
            if max_per_property is not None and any(
                property_counts.get(value, 0) >= max_per_property
                for value in unique_properties
            ):
                continue
            new_families = sum(family_counts.get(str(value), 0) == 0 for value in families)
            new_properties = sum(
                property_counts.get(property_id, 0) == 0
                for property_id, _ in signature
            )
            score = float(seed["selection_metadata"]["beam_score"])
            ranked.append((new_families, new_properties, score, seed["id"], seed))
        if not ranked:
            break
        chosen = max(ranked, key=lambda row: row[:4])[-1]
        selected.append(chosen)
        remaining.remove(chosen)
        anchor = str(chosen["anchor_label"]).casefold()
        identity = chosen["selection_metadata"]["semantic_identity"]
        signature = tuple(
            (str(step["property_id"]), str(step["direction"]))
            for step in identity["property_steps"]
        )
        anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
        for family in set(chosen["selection_metadata"]["relation_families"]):
            family_counts[str(family)] = family_counts.get(str(family), 0) + 1
        for property_id in {value for value, _ in signature}:
            property_counts[property_id] = property_counts.get(property_id, 0) + 1
    return selected


def _ledger_fields(seed: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    metadata = seed.get("selection_metadata", {})
    fingerprint = str(metadata.get("question_fingerprint", ""))
    identity = metadata.get("semantic_identity", {})
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint) or not isinstance(identity, dict):
        raise ValueError(f"{seed.get('id')}: missing renewable fingerprint metadata")
    return fingerprint, identity


def admit_seeds(
    seeds: list[dict[str, Any]],
    *,
    backend: Any,
    judge: MultiHopQuestionJudge,
    ledger: QuestionLedger,
    judge_workers: int,
    backlink_branch_cap: int,
    writer: MultiHopQuestionWriter | None = None,
    arena_node_cap: int = 500,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run deterministic route gates, then the independent whole-chain judge."""
    packets: list[dict[str, Any]] = []
    jobs: list[tuple[int, MultiHopSeed, dict[str, Any]]] = []
    for seed_value in seeds:
        fingerprint, identity = _ledger_fields(seed_value)
        seed = MultiHopSeed.from_dict(seed_value)
        errors, chain = validate_chain(seed, backend)
        shortest = None
        question = compose_relative_question(seed)
        question_generation = None
        if not errors:
            try:
                shortest = validate_shortest_arena(
                    seed, chain, backend, branch_cap=backlink_branch_cap,
                    node_cap=arena_node_cap,
                )
            except (WikipediaError, ValueError) as exc:
                errors.append(f"shortest_path: {exc}")
        if not errors and writer is not None:
            try:
                with model_call_context(role="question_writer", seed_id=seed.id):
                    question_generation = writer.write(seed)
                question = str(question_generation["question"])
            except QuestionWriterGateError as exc:
                question_generation = exc.to_dict()
                errors.append(f"question writer: {exc}")
            except Exception as exc:
                errors.append(f"question writer: {exc}")
        packet: dict[str, Any] = {
            "schema_version": ENGINE_SCHEMA,
            "seed_id": seed.id, "question_fingerprint": fingerprint,
            "question": question, "question_generation": question_generation,
            "chain": chain,
            "shortest_path": shortest, "deterministic_errors": errors,
        }
        if errors:
            status = (
                "infrastructure_error"
                if has_infrastructure_error(errors)
                else "deterministic_reject"
            )
            packet["status"] = status
            ledger.record(
                fingerprint=fingerprint, seed_id=seed.id, model_id=seed.model_id,
                cutoff_date=seed.cutoff.cutoff_date, target_as_of=seed.target_as_of,
                answer_qid=str(identity["answer_qid"]), status=status,
                semantic_identity=identity, seed=seed_value, audit=packet,
            )
        else:
            packet["status"] = "judge_pending"
            jobs.append((len(packets), seed, seed_value))
        packets.append(packet)

    accepted_by_index: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=judge_workers) as executor:
        futures = {
            executor.submit(
                judge.judge, str(packets[index]["question"]),
                list(packets[index]["chain"]),
            ): (index, seed, seed_value)
            for index, seed, seed_value in jobs
        }
        for future in as_completed(futures):
            index, seed, seed_value = futures[future]
            packet = packets[index]
            fingerprint, identity = _ledger_fields(seed_value)
            try:
                judged = future.result()
            except Exception as exc:
                packet["status"] = "infrastructure_error"
                packet["judge_error"] = str(exc)
                ledger_status = "infrastructure_error"
            else:
                packet["judge"] = judged
                if judged.get("decision") == "pass":
                    packet["status"] = "machine_pass_human_review_required"
                    accepted_by_index[index] = build_case(
                        seed, list(packet["chain"]), judged,
                        question=str(packet["question"]),
                        question_generation=packet.get("question_generation"),
                    )
                    ledger_status = "machine_pass_human_review_required"
                else:
                    packet["status"] = "judge_reject"
                    ledger_status = "judge_reject"
            ledger.record(
                fingerprint=fingerprint, seed_id=seed.id, model_id=seed.model_id,
                cutoff_date=seed.cutoff.cutoff_date, target_as_of=seed.target_as_of,
                answer_qid=str(identity["answer_qid"]), status=ledger_status,
                semantic_identity=identity, seed=seed_value, audit=packet,
            )
    return packets, [accepted_by_index[index] for index in sorted(accepted_by_index)]


def _write_json(path: str, value: Any) -> None:
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(target)


def _manifest_contract_sha256(manifest: dict[str, Any]) -> str:
    """Hash preregistered inputs/configuration, never observed run outcomes."""
    keys = (
        "schema_version", "model_id", "cutoff_date", "until",
        "registry_schema", "registry_version", "registry_sha256",
        "profile_sha256s", "admitted_properties", "config",
    )
    return _canonical_sha256({key: manifest[key] for key in keys})


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate renewable cutoff-relative questions from relation profiles"
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--profile", action="append", required=True,
        help="relation profile JSON; repeat to merge immutable profiling windows",
    )
    parser.add_argument("--registry")
    parser.add_argument("--until", default=date.today().isoformat())
    parser.add_argument("--min-hops", type=int, default=4)
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--min-families", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--max-anchors", type=int, default=100)
    parser.add_argument("--max-expansions-per-state", type=int, default=8)
    parser.add_argument("--inverse-limit", type=int, default=100)
    parser.add_argument("--allow-repeated-properties", action="store_true")
    parser.add_argument("--max-property-occurrences", type=int, default=3)
    parser.add_argument("--min-temporal-hops", type=int, default=3)
    parser.add_argument("--tail-state-cap", type=int, default=12)
    parser.add_argument("--no-terminal-attribute-tails", action="store_true")
    parser.add_argument("--max-seeds", type=int, default=20)
    parser.add_argument("--max-per-anchor", type=int, default=2)
    parser.add_argument("--max-per-property-sequence", type=int, default=1)
    parser.add_argument("--max-per-relation-family", type=int, default=8)
    parser.add_argument("--max-per-property", type=int, default=5)
    parser.add_argument(
        "--allow-easy-tails", action="store_true",
        help="allow high-prior tails such as citizenship, birthplace, and native language",
    )
    parser.add_argument("--popularity-month")
    parser.add_argument("--no-popularity-ranking", action="store_true")
    parser.add_argument(
        "--generator-model",
        help="LLM question writer; omit only to use the explicit deterministic fallback",
    )
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-workers", type=int, default=4)
    parser.add_argument("--judge-min-confidence", type=float, default=0.8)
    parser.add_argument("--judge-cache-path")
    parser.add_argument(
        "--edge-judge-model",
        help="semantic before/after judge; defaults to --judge-model in formal mode",
    )
    parser.add_argument("--edge-judge-cache-path")
    parser.add_argument("--seeds-only", action="store_true")
    parser.add_argument("--backlink-branch-cap", type=int, default=5)
    parser.add_argument("--arena-node-cap", type=int, default=200)
    parser.add_argument("--cache-path", default="renewable_question_generation.db")
    parser.add_argument("--ledger-path", default="renewable_question_ledger.db")
    parser.add_argument("--retry-rejected", action="store_true")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--request-interval", type=float, default=0.75)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--seeds-output", default="renewable_multihop_seeds.json")
    parser.add_argument("--packets-output", default="renewable_question_packets.jsonl")
    parser.add_argument("--cases-output", default="renewable_generated_cases.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    positive = (
        args.beam_width, args.max_anchors, args.max_expansions_per_state,
        args.inverse_limit, args.max_seeds, args.max_per_anchor,
        args.max_per_property_sequence, args.judge_workers,
        args.max_per_relation_family, args.max_per_property,
        args.backlink_branch_cap, args.arena_node_cap,
        args.max_property_occurrences,
        args.min_temporal_hops, args.tail_state_cap,
    )
    if min(positive) <= 0:
        parser.error("all count and cap arguments must be > 0")
    if not 0 <= args.judge_min_confidence <= 1:
        parser.error("--judge-min-confidence must be between 0 and 1")
    if args.request_interval < 0:
        parser.error("--request-interval must be >= 0")
    if not args.seeds_only:
        if not args.judge_model:
            parser.error("--judge-model is required unless --seeds-only")
        if not os.environ.get("OPENROUTER_API_KEY"):
            parser.error("OPENROUTER_API_KEY is required for generation and judging")
        if args.generator_model and args.generator_model == args.judge_model:
            parser.error("--generator-model and --judge-model must differ")
    elif args.generator_model:
        parser.error("--generator-model cannot be used with --seeds-only")
    if args.edge_judge_model and not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is required for edge contrast judging")
    output_paths = [args.seeds_output, args.packets_output]
    if not args.seeds_only:
        output_paths.append(args.cases_output)
    for output in output_paths:
        assert_new_output_path(output)
        if Path(output).exists() and not args.overwrite:
            parser.error(f"refusing to overwrite {output}; pass --overwrite explicitly")

    try:
        registry = load_temporal_relation_registry(args.registry)
        admissions = load_relation_admissions(
            args.profile, registry, require_active=True,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    popularity = None
    if not args.no_popularity_ranking:
        try:
            popularity = fetch_top_pageviews(
                lang=args.lang, month=args.popularity_month
            )
        except (OSError, ValueError, requests.RequestException) as exc:
            print(f"[error] popularity snapshot unavailable: {exc}")
            return 2
    backend = WikipediaPageBackend(
        cache_path=args.cache_path, lang=args.lang, offline_only=args.offline,
        min_request_interval=args.request_interval,
    )
    ledger = QuestionLedger(args.ledger_path)
    judge = None
    writer = None
    edge_judge = None
    try:
        edge_judge_model = args.edge_judge_model or (
            None if args.seeds_only else args.judge_model
        )
        if edge_judge_model:
            edge_judge = TemporalEdgeContrastJudge(
                edge_judge_model, min_confidence=args.judge_min_confidence,
                cache_path=(
                    args.edge_judge_cache_path
                    or f"{args.cache_path}.edge_judge.db"
                ),
            )
        engine = RenewableQuestionEngine(
            model_id=args.model_id, until=args.until, registry=registry,
            admissions=admissions, backend=backend, popularity=popularity,
            min_hops=args.min_hops, max_hops=args.max_hops,
            min_families=args.min_families, beam_width=args.beam_width,
            max_anchors=args.max_anchors,
            max_expansions_per_state=args.max_expansions_per_state,
            inverse_limit=args.inverse_limit,
            allow_repeated_properties=args.allow_repeated_properties,
            max_property_occurrences=args.max_property_occurrences,
            edge_contrast_judge=edge_judge,
            terminal_attribute_tails=not args.no_terminal_attribute_tails,
            min_temporal_hops=args.min_temporal_hops,
            tail_state_cap=args.tail_state_cap,
            allow_easy_tails=args.allow_easy_tails,
        )
        discovered = engine.discover()
        unseen = []
        for seed in discovered:
            fingerprint, _ = _ledger_fields(seed)
            if ledger.excludes(fingerprint, retry_rejected=args.retry_rejected):
                engine.packets.append({
                    "stage": "ledger", "status": "duplicate",
                    "seed_id": seed["id"], "question_fingerprint": fingerprint,
                })
            else:
                unseen.append(seed)
        selected = select_diverse_seeds(
            unseen, max_total=args.max_seeds,
            max_per_anchor=args.max_per_anchor,
            max_per_property_sequence=args.max_per_property_sequence,
            max_per_family=args.max_per_relation_family,
            max_per_property=args.max_per_property,
        )
        run_manifest = {
            "schema_version": ENGINE_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_id": args.model_id,
            "cutoff_date": get_model_cutoff(args.model_id).cutoff_date,
            "until": args.until,
            "registry_schema": registry.schema_version,
            "registry_version": registry.registry_version,
            "registry_sha256": _file_sha256(args.registry) if args.registry else (
                _canonical_sha256(registry.to_dict())
            ),
            "profile_sha256s": sorted({
                value for admission in admissions for value in admission.profile_sha256s
            }),
            "admitted_properties": [admission.spec.property_id for admission in admissions],
            "config": {
                "min_hops": args.min_hops, "max_hops": args.max_hops,
                "min_families": args.min_families, "beam_width": args.beam_width,
                "max_anchors": args.max_anchors,
                "max_expansions_per_state": args.max_expansions_per_state,
                "inverse_limit": args.inverse_limit,
                "allow_repeated_properties": args.allow_repeated_properties,
                "max_property_occurrences": args.max_property_occurrences,
                "terminal_attribute_tails": not args.no_terminal_attribute_tails,
                "min_temporal_hops": args.min_temporal_hops,
                "tail_state_cap": args.tail_state_cap,
                "allow_easy_tails": args.allow_easy_tails,
                "max_per_relation_family": args.max_per_relation_family,
                "max_per_property": args.max_per_property,
                "popularity_month": popularity.month if popularity else None,
                "edge_judge_model": edge_judge_model,
                "generator_model": args.generator_model,
                "judge_model": args.judge_model,
                "backlink_branch_cap": args.backlink_branch_cap,
                "arena_node_cap": args.arena_node_cap,
            },
            "counts": {
                "discovered": len(discovered), "unseen": len(unseen),
                "selected": len(selected),
            },
        }
        run_manifest["contract_sha256"] = _manifest_contract_sha256(run_manifest)
        _write_json(args.seeds_output, {"_renewal": run_manifest, "seeds": selected})

        admission_packets: list[dict[str, Any]] = []
        cases: list[dict[str, Any]] = []
        if args.seeds_only:
            for seed in selected:
                fingerprint, identity = _ledger_fields(seed)
                ledger.record(
                    fingerprint=fingerprint, seed_id=str(seed["id"]),
                    model_id=args.model_id,
                    cutoff_date=str(identity["cutoff_date"]),
                    target_as_of=str(identity["target_as_of"]),
                    answer_qid=str(identity["answer_qid"]), status="pending",
                    semantic_identity=identity, seed=seed,
                    audit={"status": "pending_formal_admission"},
                )
        else:
            judge = MultiHopQuestionJudge(
                args.judge_model, min_confidence=args.judge_min_confidence,
                cache_path=args.judge_cache_path or f"{args.cache_path}.judge.db",
            )
            writer = (
                MultiHopQuestionWriter(args.generator_model)
                if args.generator_model else None
            )
            admission_packets, cases = admit_seeds(
                selected, backend=backend, judge=judge, writer=writer, ledger=ledger,
                judge_workers=args.judge_workers,
                backlink_branch_cap=args.backlink_branch_cap,
                arena_node_cap=args.arena_node_cap,
            )
        admission_statuses: dict[str, int] = {}
        for packet in admission_packets:
            status = str(packet.get("status", "unknown"))
            admission_statuses[status] = admission_statuses.get(status, 0) + 1
        run_manifest["counts"].update({
            "pending_formal_admission": len(selected) if args.seeds_only else 0,
            "machine_pass_human_review_required": len(cases),
            "admission_statuses": admission_statuses,
        })
        # The early seed write is a crash-recovery checkpoint.  On normal
        # completion, atomically refresh it with final observational counts.
        _write_json(args.seeds_output, {"_renewal": run_manifest, "seeds": selected})
        if not args.seeds_only:
            _write_json(args.cases_output, {"_renewal": run_manifest, "cases": cases})
        _write_jsonl(args.packets_output, [
            {"stage": "manifest", **run_manifest},
            *engine.packets, *admission_packets,
        ])
        print(
            f"[done] discovered={len(discovered)} selected={len(selected)} "
            f"pending_formal={len(selected) if args.seeds_only else 0} "
            f"machine_pass={len(cases)} ledger={ledger.counts()}",
            flush=True,
        )
        completed_count = len(selected) if args.seeds_only else len(cases)
        return 0 if completed_count else 1
    finally:
        if judge is not None:
            judge.close()
        if edge_judge is not None:
            edge_judge.close()
        ledger.close()
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
