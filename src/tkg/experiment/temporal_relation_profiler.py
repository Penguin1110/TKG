"""Mine and profile renewable Wikidata temporal relation specifications.

This stage proposes relation-level promotion candidates.  It never activates a
relation or creates a formal question without the downstream deterministic and
LLM gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

import requests

from tkg.api.openrouter import OpenRouterError, call_model
from tkg.experiment.question_generation import call_json_model
from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_relation_registry import (
    TemporalRelationSpec, load_temporal_relation_registry,
)
from tkg.wikipedia.backend import WikipediaError, WikipediaPageBackend


WDQS_URL = "https://query.wikidata.org/sparql"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
USER_AGENT = "TKG-renewable-temporal-relation-profiler/0.1 (research)"
DATE_OFFSETS = (0, 1, 3, 7)
PROFILE_SCHEMA = "temporal-relation-profile-v1"
RELATION_SEMANTIC_CHECKS = (
    "direction", "relation_semantics", "temporal_support",
    "not_mere_cooccurrence",
)


def semantic_judge_schema_errors(value: dict[str, Any]) -> list[str]:
    """Validate every affirmative relation-judge assertion explicitly."""
    errors = []
    checks = value.get("checks")
    if not isinstance(checks, dict):
        errors.append("checks_not_object")
    else:
        for key in RELATION_SEMANTIC_CHECKS:
            if checks.get(key) is not True:
                errors.append(f"check_not_true:{key}")
    if not str(value.get("expressed_relation", "")).strip():
        errors.append("missing_expressed_relation")
    if not str(value.get("reason", "")).strip():
        errors.append("missing_reason")
    return errors


def _utc_timestamp(value: str) -> str:
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("date must use strict YYYY-MM-DD form")
    return value + "T00:00:00Z"


def _binding(row: dict[str, Any], key: str) -> str:
    value = row.get(key, {})
    return str(value.get("value", "")).strip() if isinstance(value, dict) else ""


def _qid(value: str) -> str:
    result = value.rsplit("/", 1)[-1]
    if not result.startswith(("Q", "P")):
        raise ValueError(f"not a Wikidata entity URI: {value!r}")
    return result


def _article_title(value: str) -> str:
    path = unquote(urlparse(value).path)
    if "/wiki/" not in path:
        raise ValueError(f"not a Wikipedia article URL: {value!r}")
    return " ".join(path.split("/wiki/", 1)[1].replace("_", " ").split())


def _date_value(value: str) -> str | None:
    if not value:
        return None
    normalized = value.lstrip("+")
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).date().isoformat()


@dataclass(frozen=True)
class TemporalBinding:
    property_id: str
    source_qid: str
    source_title: str
    target_qid: str
    target_title: str
    start: str | None
    end: str | None
    point: str | None
    event_date: str

    @classmethod
    def from_row(
        cls, spec: TemporalRelationSpec, row: dict[str, Any],
        *, since: str | None = None, until: str | None = None,
    ) -> "TemporalBinding":
        start = _date_value(_binding(row, "start"))
        end = _date_value(_binding(row, "end"))
        point = _date_value(_binding(row, "point"))
        candidates = (
            (point, start, end) if spec.time_mode == "point" else (start, point, end)
        )
        event_date = next((
            value for value in candidates
            if value is not None
            and (since is None or value >= since)
            and (until is None or value <= until)
        ), None)
        if event_date is None:
            raise ValueError("binding has no usable temporal qualifier")
        return cls(
            property_id=spec.property_id,
            source_qid=_qid(_binding(row, "source")),
            source_title=_article_title(_binding(row, "sourceArticle")),
            target_qid=_qid(_binding(row, "target")),
            target_title=_article_title(_binding(row, "targetArticle")),
            start=start, end=end, point=point, event_date=event_date,
        )


def relation_sample_query(
    spec: TemporalRelationSpec, *, since: str, until: str, limit: int
) -> str:
    """Return recent entity-valued statements with temporal qualifiers and enwiki pages."""
    start = _utc_timestamp(since)
    end = _utc_timestamp(until)
    if spec.time_mode == "point":
        time_filter = f'BOUND(?point) && ?point >= "{start}"^^xsd:dateTime && ?point <= "{end}"^^xsd:dateTime'
    elif spec.time_mode == "interval":
        time_filter = (
            f'(BOUND(?start) && ?start >= "{start}"^^xsd:dateTime && '
            f'?start <= "{end}"^^xsd:dateTime) || '
            f'(BOUND(?end) && ?end >= "{start}"^^xsd:dateTime && '
            f'?end <= "{end}"^^xsd:dateTime)'
        )
    else:
        time_filter = (
            f'(BOUND(?start) && ?start >= "{start}"^^xsd:dateTime && ?start <= "{end}"^^xsd:dateTime) || '
            f'(BOUND(?end) && ?end >= "{start}"^^xsd:dateTime && ?end <= "{end}"^^xsd:dateTime) || '
            f'(BOUND(?point) && ?point >= "{start}"^^xsd:dateTime && ?point <= "{end}"^^xsd:dateTime)'
        )
    prop = spec.property_id
    return f"""SELECT DISTINCT ?source ?sourceArticle ?target ?targetArticle ?start ?end ?point WHERE {{
  ?source p:{prop} ?statement .
  ?statement ps:{prop} ?target .
  FILTER(isIRI(?target))
  OPTIONAL {{ ?statement pq:P580 ?start . }}
  OPTIONAL {{ ?statement pq:P582 ?end . }}
  OPTIONAL {{ ?statement pq:P585 ?point . }}
  FILTER({time_filter})
  ?sourceArticle schema:about ?source ; schema:isPartOf <https://en.wikipedia.org/> .
  ?targetArticle schema:about ?target ; schema:isPartOf <https://en.wikipedia.org/> .
}} ORDER BY DESC(?start) DESC(?point) DESC(?end) ?source ?target LIMIT {int(limit)}"""


def property_discovery_query(*, since: str, until: str, limit: int) -> str:
    """Find properties participating in at least one recent time-qualified statement."""
    start = _utc_timestamp(since)
    end = _utc_timestamp(until)
    return f"""SELECT DISTINCT ?property ?propertyLabel ?qualifier WHERE {{
  VALUES (?qualifier ?qualifierPredicate) {{
    (wd:P580 pq:P580)
    (wd:P582 pq:P582)
    (wd:P585 pq:P585)
  }}
  ?statement ?qualifierPredicate ?time .
  FILTER(?time >= "{start}"^^xsd:dateTime && ?time <= "{end}"^^xsd:dateTime)
  ?property wikibase:claim ?claimPredicate ;
            wikibase:statementProperty ?statementPredicate .
  ?source ?claimPredicate ?statement .
  ?statement ?statementPredicate ?target .
  FILTER(isIRI(?target))
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}} LIMIT {int(limit)}"""


def _query_bindings(
    query: str,
    *,
    request_get: Callable = requests.get,
    max_retries: int = 3,
    timeout: float = 90,
) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = request_get(
                WDQS_URL,
                params={"query": query, "format": "json"},
                headers={
                    "Accept": "application/sparql-results+json",
                    "User-Agent": USER_AGENT,
                },
                timeout=timeout,
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(f"HTTP {response.status_code}")
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("results", {}).get("bindings", [])
            if not isinstance(rows, list):
                raise ValueError("WDQS response has no results.bindings list")
            return rows
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"WDQS query failed: {last_error}")


def discover_temporal_properties(
    *, since: str, until: str, limit: int, request_get: Callable = requests.get,
    timeout: float = 20,
) -> tuple[list[dict[str, Any]], str]:
    query = property_discovery_query(since=since, until=until, limit=limit)
    rows = _query_bindings(
        query, request_get=request_get, max_retries=1, timeout=timeout
    )
    by_property: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            property_id = _qid(_binding(row, "property"))
            qualifier = _qid(_binding(row, "qualifier"))
        except ValueError:
            continue
        record = by_property.setdefault(property_id, {
            "property_id": property_id,
            "label": _binding(row, "propertyLabel") or property_id,
            "qualifiers": [],
            "status": "discovered",
        })
        if qualifier not in record["qualifiers"]:
            record["qualifiers"].append(qualifier)
    return list(by_property.values()), hashlib.sha256(query.encode()).hexdigest()


def _wikidata_api_json(
    params: dict[str, Any], *, request_get: Callable = requests.get,
    timeout: float = 20,
) -> dict[str, Any]:
    response = request_get(
        WIKIDATA_API_URL,
        params={"format": "json", "formatversion": "2", **params},
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        timeout=timeout,
    )
    if response.status_code == 429 or response.status_code >= 500:
        raise requests.HTTPError(f"HTTP {response.status_code}")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("error"):
        raise ValueError(f"invalid Wikidata API response: {payload.get('error')!r}")
    return payload


def discover_temporal_properties_from_recent_changes(
    *,
    since: str,
    until: str,
    property_limit: int,
    recent_entity_limit: int = 100,
    request_get: Callable = requests.get,
    timeout: float = 20,
) -> tuple[list[dict[str, Any]], str]:
    """Mine temporal properties from a bounded, renewable recent-change sample."""
    changes = _wikidata_api_json({
        "action": "query", "list": "recentchanges", "rcnamespace": "0",
        "rctype": "edit|new", "rclimit": min(500, recent_entity_limit),
        "rcprop": "title|timestamp|ids",
    }, request_get=request_get, timeout=timeout)
    qids = []
    for row in changes.get("query", {}).get("recentchanges", []):
        qid = str(row.get("title", ""))
        if re.fullmatch(r"Q[1-9]\d*", qid) and qid not in qids:
            qids.append(qid)
        if len(qids) >= recent_entity_limit:
            break

    discovered: dict[str, dict[str, Any]] = {}
    for index in range(0, len(qids), 50):
        chunk = qids[index:index + 50]
        payload = _wikidata_api_json({
            "action": "wbgetentities", "ids": "|".join(chunk), "props": "claims",
        }, request_get=request_get, timeout=timeout)
        for source_qid, entity in payload.get("entities", {}).items():
            claims = entity.get("claims", {}) if isinstance(entity, dict) else {}
            if not isinstance(claims, dict):
                continue
            for property_id, statements in claims.items():
                if not re.fullmatch(r"P[1-9]\d*", str(property_id)):
                    continue
                if not isinstance(statements, list):
                    continue
                for statement in statements:
                    if not isinstance(statement, dict):
                        continue
                    target = statement.get("mainsnak", {}).get(
                        "datavalue", {}
                    ).get("value", {})
                    target_qid = target.get("id") if isinstance(target, dict) else None
                    if not isinstance(target_qid, str) or not re.fullmatch(
                        r"Q[1-9]\d*", target_qid
                    ):
                        continue
                    qualifier_hits = []
                    for qualifier_id in ("P580", "P582", "P585"):
                        for snak in statement.get("qualifiers", {}).get(qualifier_id, []):
                            raw = snak.get("datavalue", {}).get("value", {})
                            when = _date_value(
                                str(raw.get("time", "")) if isinstance(raw, dict) else ""
                            )
                            if when is not None and since <= when <= until:
                                qualifier_hits.append(qualifier_id)
                    if not qualifier_hits:
                        continue
                    record = discovered.setdefault(str(property_id), {
                        "property_id": str(property_id), "label": str(property_id),
                        "qualifiers": [], "recent_statement_count": 0,
                        "sample_source_qids": [], "status": "discovered",
                    })
                    record["recent_statement_count"] += 1
                    for qualifier_id in qualifier_hits:
                        if qualifier_id not in record["qualifiers"]:
                            record["qualifiers"].append(qualifier_id)
                    if (
                        source_qid not in record["sample_source_qids"]
                        and len(record["sample_source_qids"]) < 3
                    ):
                        record["sample_source_qids"].append(source_qid)

    ranked = sorted(
        discovered.values(),
        key=lambda row: (-int(row["recent_statement_count"]), row["property_id"]),
    )[:property_limit]
    property_ids = [str(row["property_id"]) for row in ranked]
    labels: dict[str, str] = {}
    for index in range(0, len(property_ids), 50):
        chunk = property_ids[index:index + 50]
        payload = _wikidata_api_json({
            "action": "wbgetentities", "ids": "|".join(chunk),
            "props": "labels", "languages": "en",
        }, request_get=request_get, timeout=timeout)
        for property_id, entity in payload.get("entities", {}).items():
            label = entity.get("labels", {}).get("en", {}).get("value")
            if label:
                labels[str(property_id)] = str(label)
    for row in ranked:
        row["label"] = labels.get(str(row["property_id"]), str(row["property_id"]))
    manifest = {
        "method": "wikidata_recent_changes",
        "since": since, "until": until,
        "recent_entity_limit": recent_entity_limit,
        "sampled_qids": qids,
    }
    return ranked, hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()


def query_relation_samples(
    spec: TemporalRelationSpec,
    *,
    since: str,
    until: str,
    limit: int,
    request_get: Callable = requests.get,
) -> tuple[list[TemporalBinding], str, int]:
    query = relation_sample_query(spec, since=since, until=until, limit=limit)
    rows = _query_bindings(query, request_get=request_get)
    samples = []
    seen = set()
    for row in rows:
        try:
            binding = TemporalBinding.from_row(
                spec, row, since=since, until=until
            )
        except ValueError:
            continue
        key = (
            binding.source_qid, binding.target_qid, binding.event_date,
            binding.start, binding.end, binding.point,
        )
        if key not in seen:
            seen.add(key)
            samples.append(binding)
    samples.sort(key=lambda value: (value.event_date, value.source_title), reverse=True)
    return samples, hashlib.sha256(query.encode()).hexdigest(), len(rows)


def relation_time_buckets(
    since: str, until: str, bucket_count: int,
) -> list[tuple[str, str]]:
    """Split an inclusive date window into deterministic non-overlapping buckets."""
    start = date.fromisoformat(since)
    end = date.fromisoformat(until)
    if start > end or bucket_count <= 0:
        raise ValueError("invalid temporal bucket window or count")
    total_days = (end - start).days + 1
    count = min(bucket_count, total_days)
    boundaries = [
        start + timedelta(days=(total_days * index) // count)
        for index in range(count + 1)
    ]
    return [
        (
            boundaries[index].isoformat(),
            (boundaries[index + 1] - timedelta(days=1)).isoformat(),
        )
        for index in range(count)
    ]


def query_relation_samples_stratified(
    spec: TemporalRelationSpec,
    *,
    since: str,
    until: str,
    limit: int,
    time_buckets: int,
    request_get: Callable = requests.get,
) -> tuple[list[TemporalBinding], str, int]:
    """Keep older portions of a long profiling window represented."""
    buckets = relation_time_buckets(since, until, time_buckets)
    base, remainder = divmod(limit, len(buckets))
    allocations = [base + int(index < remainder) for index in range(len(buckets))]
    samples: list[TemporalBinding] = []
    query_hashes = []
    raw_count = 0
    for (bucket_since, bucket_until), bucket_limit in zip(buckets, allocations):
        if bucket_limit <= 0:
            continue
        values, query_hash, count = query_relation_samples(
            spec, since=bucket_since, until=bucket_until,
            limit=bucket_limit, request_get=request_get,
        )
        samples.extend(values)
        query_hashes.append({
            "since": bucket_since, "until": bucket_until,
            "limit": bucket_limit, "query_sha256": query_hash,
        })
        raw_count += count
    unique: dict[tuple[Any, ...], TemporalBinding] = {}
    for sample in samples:
        key = (
            sample.source_qid, sample.target_qid, sample.event_date,
            sample.start, sample.end, sample.point,
        )
        unique[key] = sample
    result = sorted(
        unique.values(),
        key=lambda value: (value.event_date, value.source_title),
        reverse=True,
    )[:limit]
    manifest_hash = hashlib.sha256(json.dumps({
        "method": "stratified_time_buckets_v1", "property_id": spec.property_id,
        "buckets": query_hashes,
    }, sort_keys=True).encode()).hexdigest()
    return result, manifest_hash, raw_count


def _has_link(page, target_title: str) -> bool:
    return any(link.target.casefold() == target_title.casefold() for link in page.links)


def _evidence_window(page, target_title: str) -> tuple[str, str] | None:
    links = [
        link for link in page.links
        if link.target.casefold() == target_title.casefold()
    ]
    if not links:
        return None
    alias = max((link.anchor for link in links), key=len)
    position = page.content.casefold().find(alias.casefold())
    if position < 0:
        return None
    return (
        page.content[max(0, position - 100):position + len(alias) + 180].strip(),
        alias,
    )


def verify_wikipedia_binding(binding: TemporalBinding, backend) -> dict[str, Any]:
    """Measure hyperlink support and prior-day novelty without promoting the relation."""
    base = date.fromisoformat(binding.event_date)
    errors = []
    supported_page = None
    supported_as_of = None
    evidence = None
    for offset in DATE_OFFSETS:
        as_of = (base + timedelta(days=offset)).isoformat()
        try:
            page = backend.fetch_page(binding.source_title, as_of=as_of)
        except WikipediaError as exc:
            errors.append(f"{as_of}: {exc}")
            continue
        candidate_evidence = _evidence_window(page, binding.target_title)
        if _has_link(page, binding.target_title) and candidate_evidence is not None:
            supported_page = page
            supported_as_of = as_of
            evidence = candidate_evidence
            break
    prior_absent = None
    prior_revision_id = None
    if supported_page is not None:
        prior_as_of = (base - timedelta(days=1)).isoformat()
        try:
            prior = backend.fetch_page(binding.source_title, as_of=prior_as_of)
            prior_revision_id = prior.revision_id
            prior_absent = not _has_link(prior, binding.target_title)
        except WikipediaError as exc:
            errors.append(f"{prior_as_of}: {exc}")
    return {
        **asdict(binding),
        "status": (
            "link_supported" if supported_page is not None else "no_visible_link"
        ),
        "supported_as_of": supported_as_of,
        "revision_id": supported_page.revision_id if supported_page else None,
        "source_url": supported_page.source_url if supported_page else None,
        "evidence": evidence[0] if evidence else None,
        "target_alias": evidence[1] if evidence else None,
        "prior_day_absent": prior_absent,
        "prior_revision_id": prior_revision_id,
        "errors": errors,
    }


class RelationEvidenceJudge:
    """Independent cached judge for source-relation-target evidence semantics."""

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
                "CREATE TABLE IF NOT EXISTS relation_evidence_judge_cache ("
                "cache_key TEXT PRIMARY KEY, model TEXT NOT NULL, "
                "result_json TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            with self._lock:
                self._conn.close()
                self._conn = None

    def _load(self, cache_key: str) -> dict[str, Any] | None:
        if self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT result_json FROM relation_evidence_judge_cache "
                "WHERE cache_key=?", (cache_key,),
            ).fetchone()
        return dict(json.loads(row[0])) if row else None

    def _store(self, cache_key: str, result: dict[str, Any]) -> None:
        if self._conn is None:
            return
        stored = {key: value for key, value in result.items() if key != "cache_hit"}
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO relation_evidence_judge_cache("
                "cache_key,model,result_json,created_at) VALUES(?,?,?,?)",
                (
                    cache_key, self.model,
                    json.dumps(stored, ensure_ascii=False, sort_keys=True),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()

    def judge(
        self, spec: TemporalRelationSpec, check: dict[str, Any]
    ) -> dict[str, Any]:
        visible = {
            "property_id": spec.property_id,
            "relation_label": spec.label,
            "time_mode": spec.time_mode,
            "source_kind": spec.source_kind,
            "target_kind": spec.target_kind,
            "semantic_guidance": spec.semantic_guidance,
            "source_title": check.get("source_title"),
            "target_title": check.get("target_title"),
            "event_date": check.get("event_date"),
            "supported_as_of": check.get("supported_as_of"),
            "start": check.get("start"),
            "end": check.get("end"),
            "point": check.get("point"),
            "evidence": check.get("evidence"),
            "target_alias": check.get("target_alias"),
        }
        cache_key = hashlib.sha256(json.dumps({
            "prompt_version": 4,
            "model": self.model,
            "min_confidence": self.min_confidence,
            "input": visible,
        }, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cached = self._load(cache_key)
        if cached is not None:
            return {**cached, "cache_hit": True}
        prompt = f"""Audit one proposed temporal Wikidata relation against visible Wikipedia evidence.

Input:
{json.dumps(visible, ensure_ascii=False, indent=2)}

The rendered excerpt writes hyperlinks as [visible anchor -> destination page]. An infobox
field immediately followed by its value, such as "Prime Minister [Name -> Name]" or
"Head coach [Name -> Name]", is an explicit assertion even without a complete sentence.
Flattened patterns such as "[President -> President of X] [Name -> Name]" likewise assert
that Name is President of X; use the relation-specific semantic_guidance in the input to
decide whether that role is an accepted expression of the property.

Decide whether the excerpt states that the source has the named relation to the target at
the relevant snapshot or interval. Reject mere co-occurrence and different roles. In
particular: Chancellor is not CEO; party leader or honorary president is not automatically
chairperson; senator, member, officeholder, student, software developer, or publisher is
not evidence of employer unless the excerpt explicitly says employed by, works for, or an
equivalent employment relation. A role-label hyperlink whose destination happens to be
the target organization, such as "[Member of the Senate -> Senate of Colombia]", expresses
membership/position held, not employer. Accept a genuine synonym only when the local
context establishes the same role. Do not use outside knowledge.

Return one JSON object with decision (pass/reject), confidence (0..1),
expressed_relation, checks (direction, relation_semantics, temporal_support,
not_mere_cooccurrence), and reason.
"""
        raw, response = call_json_model(self.model, prompt, self.call_model_fn)
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        decision = str(raw.get("decision", "reject"))
        schema_errors = semantic_judge_schema_errors(raw)
        if decision != "pass" or confidence < self.min_confidence or schema_errors:
            decision = "reject"
        result = {
            **raw, "decision": decision, "confidence": confidence,
            "schema_gate_errors": schema_errors,
            "raw_response": response, "cache_hit": False,
        }
        self._store(cache_key, result)
        return result


def _recommendation(
    *,
    kg_sampled: int,
    wiki_checked: int,
    wiki_supported: int,
    min_kg_samples: int,
    min_wiki_supported: int,
    min_wiki_support_rate: float,
    semantic_judge_requested: bool,
    semantic_checked: int,
    semantic_supported: int,
    min_semantic_supported: int,
    min_semantic_support_rate: float,
    query_error: str | None,
) -> str:
    if query_error:
        return "infrastructure_error"
    if kg_sampled < min_kg_samples:
        return "quarantine_low_recent_kg_yield"
    if wiki_checked == 0:
        return "profiled_kg_only"
    support_rate = wiki_supported / wiki_checked
    if wiki_supported < min_wiki_supported or support_rate < min_wiki_support_rate:
        return "quarantine_low_wikipedia_yield"
    if not semantic_judge_requested:
        return "wikipedia_link_candidate_semantic_review_required"
    if semantic_checked == 0:
        return "semantic_judge_infrastructure_error"
    semantic_rate = semantic_supported / semantic_checked
    if (
        semantic_supported >= min_semantic_supported
        and semantic_rate >= min_semantic_support_rate
    ):
        return "semantic_validated_candidate_human_review_required"
    return "quarantine_low_semantic_yield"


def build_relation_profile(
    spec: TemporalRelationSpec,
    *,
    samples: list[TemporalBinding],
    raw_row_count: int,
    query_sha256: str,
    wiki_checks: list[dict[str, Any]],
    query_error: str | None,
    min_kg_samples: int,
    min_wiki_supported: int,
    min_wiki_support_rate: float,
    semantic_judge_requested: bool = False,
    min_semantic_supported: int = 1,
    min_semantic_support_rate: float = 0.5,
) -> dict[str, Any]:
    wiki_supported = sum(row["status"] == "link_supported" for row in wiki_checks)
    prior_absent = sum(row.get("prior_day_absent") is True for row in wiki_checks)
    semantic_results = [
        row["semantic_judge"] for row in wiki_checks
        if isinstance(row.get("semantic_judge"), dict)
        and row["semantic_judge"].get("decision") in {"pass", "reject"}
    ]
    semantic_supported = sum(
        row.get("decision") == "pass" for row in semantic_results
    )
    semantic_infrastructure_errors = sum(
        isinstance(row.get("semantic_judge"), dict)
        and row["semantic_judge"].get("decision") == "infrastructure_error"
        for row in wiki_checks
    )
    kg_sampled = len(samples)
    wiki_checked = len(wiki_checks)
    return {
        "relation": spec.to_dict(),
        "query_sha256": query_sha256,
        "query_error": query_error,
        "metrics": {
            "raw_rows": raw_row_count,
            "kg_sampled": kg_sampled,
            "distinct_sources": len({row.source_qid for row in samples}),
            "distinct_targets": len({row.target_qid for row in samples}),
            "with_start": sum(row.start is not None for row in samples),
            "with_end": sum(row.end is not None for row in samples),
            "with_point": sum(row.point is not None for row in samples),
            "complete_intervals": sum(
                row.start is not None and row.end is not None for row in samples
            ),
            "wiki_checked": wiki_checked,
            "wiki_supported": wiki_supported,
            "wiki_support_rate": (
                wiki_supported / wiki_checked if wiki_checked else None
            ),
            "semantic_checked": len(semantic_results),
            "semantic_supported": semantic_supported,
            "semantic_support_rate": (
                semantic_supported / len(semantic_results)
                if semantic_results else None
            ),
            "semantic_infrastructure_errors": semantic_infrastructure_errors,
            "prior_day_absent": prior_absent,
            "prior_day_novelty_rate": (
                prior_absent / wiki_supported if wiki_supported else None
            ),
        },
        "recommendation": _recommendation(
            kg_sampled=kg_sampled,
            wiki_checked=wiki_checked,
            wiki_supported=wiki_supported,
            min_kg_samples=min_kg_samples,
            min_wiki_supported=min_wiki_supported,
            min_wiki_support_rate=min_wiki_support_rate,
            semantic_judge_requested=semantic_judge_requested,
            semantic_checked=len(semantic_results),
            semantic_supported=semantic_supported,
            min_semantic_supported=min_semantic_supported,
            min_semantic_support_rate=min_semantic_support_rate,
            query_error=query_error,
        ),
        "samples": [asdict(row) for row in samples],
        "wikipedia_checks": wiki_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile renewable temporal relations before registry promotion"
    )
    parser.add_argument("--registry")
    parser.add_argument("--properties", help="comma-separated registry property IDs")
    parser.add_argument("--since", required=True, help="inclusive YYYY-MM-DD")
    parser.add_argument("--until", default=date.today().isoformat(), help="inclusive YYYY-MM-DD")
    parser.add_argument("--kg-limit", type=int, default=25)
    parser.add_argument(
        "--time-buckets", type=int, default=1,
        help="stratify KG samples across this many equal date windows",
    )
    parser.add_argument("--wiki-samples", type=int, default=2)
    parser.add_argument("--query-workers", type=int, default=2)
    parser.add_argument("--min-kg-samples", type=int, default=3)
    parser.add_argument("--min-wiki-supported", type=int, default=1)
    parser.add_argument("--min-wiki-support-rate", type=float, default=0.25)
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-workers", type=int, default=4)
    parser.add_argument("--judge-min-confidence", type=float, default=0.8)
    parser.add_argument("--judge-cache-path")
    parser.add_argument("--min-semantic-supported", type=int, default=1)
    parser.add_argument("--min-semantic-support-rate", type=float, default=0.5)
    parser.add_argument("--mine-properties", type=int, default=0)
    parser.add_argument(
        "--mining-entities", type=int, default=100,
        help="number of recent changed QIDs scanned for new temporal properties",
    )
    parser.add_argument(
        "--mining-timeout", type=float, default=20,
        help="fail-open timeout for the optional global property-mining stage",
    )
    parser.add_argument("--cache-path", default="temporal_relation_profile.db")
    parser.add_argument("--request-interval", type=float, default=0.75)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--output", default="temporal_relation_profile.json")
    parser.add_argument("--packets-output", default="temporal_relation_profile_packets.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        start = date.fromisoformat(args.since)
        end = date.fromisoformat(args.until)
    except ValueError as exc:
        parser.error(f"invalid date: {exc}")
    if start > end:
        parser.error("--since must not be after --until")
    if (
        args.kg_limit <= 0 or args.time_buckets <= 0
        or args.query_workers <= 0 or args.judge_workers <= 0
    ):
        parser.error(
            "--kg-limit, --time-buckets, --query-workers, and --judge-workers must be > 0"
        )
    if args.wiki_samples < 0 or args.mine_properties < 0:
        parser.error("--wiki-samples and --mine-properties must be >= 0")
    if args.mining_entities <= 0:
        parser.error("--mining-entities must be > 0")
    if (
        args.min_kg_samples <= 0 or args.min_wiki_supported <= 0
        or args.min_semantic_supported <= 0
    ):
        parser.error("promotion counts must be > 0")
    if not (
        0 <= args.min_wiki_support_rate <= 1
        and 0 <= args.min_semantic_support_rate <= 1
        and 0 <= args.judge_min_confidence <= 1
    ):
        parser.error("support rates and judge confidence must be between 0 and 1")
    if args.request_interval < 0 or args.mining_timeout <= 0:
        parser.error("--request-interval must be >= 0 and --mining-timeout must be > 0")
    if args.judge_model and not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is required when --judge-model is set")
    for path in (args.output, args.packets_output):
        assert_new_output_path(path)
        if Path(path).exists() and not args.overwrite:
            parser.error(f"refusing to overwrite {path}; pass --overwrite explicitly")

    try:
        registry = load_temporal_relation_registry(args.registry)
        wanted = {
            value.strip() for value in (args.properties or "").split(",")
            if value.strip()
        }
        specs = registry.select(wanted or None)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[error] registry unavailable: {exc}", file=sys.stderr)
        return 2

    mined_properties: list[dict[str, Any]] = []
    mining_error = None
    mining_query_sha256 = None
    query_results: dict[str, tuple[list[TemporalBinding], str, int, str | None]] = {}
    with ThreadPoolExecutor(max_workers=args.query_workers) as executor:
        futures = {
            executor.submit(
                query_relation_samples_stratified, spec,
                since=args.since, until=args.until, limit=args.kg_limit,
                time_buckets=args.time_buckets,
            ): spec
            for spec in specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                samples, query_sha256, raw_count = future.result()
                query_results[spec.property_id] = (
                    samples, query_sha256, raw_count, None,
                )
                print(
                    f"[KG] {spec.property_id} {spec.label}: {len(samples)} samples",
                    flush=True,
                )
            except (RuntimeError, requests.RequestException, ValueError) as exc:
                query_results[spec.property_id] = ([], "", 0, str(exc))
                print(
                    f"[KG error] {spec.property_id} {spec.label}: {exc}",
                    file=sys.stderr, flush=True,
                )

    backend = WikipediaPageBackend(
        cache_path=args.cache_path, lang=args.lang,
        min_request_interval=args.request_interval,
    )
    profile_inputs: list[tuple[
        TemporalRelationSpec, list[TemporalBinding], str, int, str | None,
        list[dict[str, Any]],
    ]] = []
    try:
        for spec in specs:
            samples, query_sha256, raw_count, query_error = query_results[spec.property_id]
            wiki_checks = []
            for sample in samples[:args.wiki_samples]:
                check = verify_wikipedia_binding(sample, backend)
                wiki_checks.append(check)
                print(
                    f"[Wikipedia {check['status']}] {spec.property_id}: "
                    f"{sample.source_title} -> {sample.target_title} @{sample.event_date}",
                    flush=True,
                )
            profile_inputs.append((
                spec, samples, query_sha256, raw_count, query_error, wiki_checks,
            ))
    finally:
        backend.close()

    judge = None
    if args.judge_model:
        judge_cache_path = (
            args.judge_cache_path or f"{args.cache_path}.relation_judge.db"
        )
        judge = RelationEvidenceJudge(
            args.judge_model,
            min_confidence=args.judge_min_confidence,
            cache_path=judge_cache_path,
        )
        try:
            with ThreadPoolExecutor(max_workers=args.judge_workers) as executor:
                judge_futures = {
                    executor.submit(judge.judge, spec, check): (spec, check)
                    for spec, _, _, _, _, wiki_checks in profile_inputs
                    for check in wiki_checks
                    if check.get("status") == "link_supported"
                }
                for judge_future in as_completed(judge_futures):
                    spec, check = judge_futures[judge_future]
                    try:
                        judge_result = judge_future.result()
                    except (OpenRouterError, RuntimeError, ValueError) as exc:
                        judge_result = {
                            "decision": "infrastructure_error",
                            "confidence": 0.0,
                            "reason": str(exc),
                            "cache_hit": False,
                        }
                    check["semantic_judge"] = judge_result
                    print(
                        f"[semantic {judge_result['decision']}] {spec.property_id}: "
                        f"{check['source_title']} -> {check['target_title']} "
                        f"cache_hit={judge_result.get('cache_hit', False)}",
                        flush=True,
                    )
        finally:
            judge.close()

    profiles = []
    packet_stream = Path(args.packets_output).open("w", encoding="utf-8")
    try:
        for spec, samples, query_sha256, raw_count, query_error, wiki_checks in profile_inputs:
            profile = build_relation_profile(
                spec, samples=samples, raw_row_count=raw_count,
                query_sha256=query_sha256, wiki_checks=wiki_checks,
                query_error=query_error, min_kg_samples=args.min_kg_samples,
                min_wiki_supported=args.min_wiki_supported,
                min_wiki_support_rate=args.min_wiki_support_rate,
                semantic_judge_requested=bool(args.judge_model),
                min_semantic_supported=args.min_semantic_supported,
                min_semantic_support_rate=args.min_semantic_support_rate,
            )
            profiles.append(profile)
            packet_stream.write(json.dumps(profile, ensure_ascii=False) + "\n")
            packet_stream.flush()
    finally:
        packet_stream.close()

    # Property discovery is deliberately fail-open and runs after known relation
    # profiling so exploratory mining cannot block yield.
    if args.mine_properties:
        try:
            mined_properties, mining_query_sha256 = (
                discover_temporal_properties_from_recent_changes(
                    since=args.since, until=args.until,
                    property_limit=args.mine_properties,
                    recent_entity_limit=args.mining_entities,
                    timeout=args.mining_timeout,
                )
            )
        except (RuntimeError, requests.RequestException, ValueError) as exc:
            mining_error = str(exc)
            print(f"[warning] property mining failed: {exc}", file=sys.stderr, flush=True)

    active_ids = {relation.property_id for relation in registry.relations}
    for record in mined_properties:
        record["already_in_registry"] = record["property_id"] in active_ids
    summary: dict[str, int] = {}
    for profile in profiles:
        key = str(profile["recommendation"])
        summary[key] = summary.get(key, 0) + 1
    payload = {
        "schema_version": PROFILE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "registry": {
            "schema_version": registry.schema_version,
            "registry_version": registry.registry_version,
            "selected_relation_count": len(specs),
        },
        "window": {"since": args.since, "until": args.until},
        "config": {
            "kg_limit": args.kg_limit,
            "time_buckets": args.time_buckets,
            "wiki_samples": args.wiki_samples,
            "query_workers": args.query_workers,
            "judge_model": args.judge_model,
            "judge_workers": args.judge_workers,
            "judge_min_confidence": args.judge_min_confidence,
            "mining_timeout": args.mining_timeout,
            "mining_entities": args.mining_entities,
            "min_kg_samples": args.min_kg_samples,
            "min_wiki_supported": args.min_wiki_supported,
            "min_wiki_support_rate": args.min_wiki_support_rate,
            "min_semantic_supported": args.min_semantic_supported,
            "min_semantic_support_rate": args.min_semantic_support_rate,
        },
        "property_mining": {
            "method": "wikidata_recent_changes",
            "query_sha256": mining_query_sha256,
            "error": mining_error,
            "candidates": mined_properties,
        },
        "summary": summary,
        "profiles": profiles,
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] {len(profiles)} relations profiled: {summary}", flush=True)
    return 0 if any(
        profile["metrics"]["kg_sampled"] for profile in profiles
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
