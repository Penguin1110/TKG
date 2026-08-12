"""Build auditable cutoff-relative, multi-hop Wikipedia questions.

The seed supplies semantic relations; Wikipedia supplies the revision text and
hyperlink graph.  The program composes the relative question deterministically,
then an independent LLM judges whether the evidence really expresses the
claimed relation chain.  It never invents a relation from hyperlink topology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tkg.api.openrouter import call_model
from tkg.experiment.case_validation import validate_chain_route
from tkg.experiment.model_cutoffs import ModelCutoff, get_model_cutoff
from tkg.experiment.question_generation import (
    _fold, _strings, call_json_model,
)
from tkg.experiment.results import assert_new_output_path
from tkg.wikipedia.backend import WikipediaError, WikipediaPageBackend
from tkg.wikipedia.snapshot import temporal_reverse_bfs


SCHEMA_VERSION = "wikipedia-cutoff-relative-multihop-v4"
MIN_HOPS = 2
MAX_HOPS = 6
TIME_POLICIES = {"advance_required", "same_snapshot"}
WHOLE_CHAIN_CHECKS = (
    "evidence_semantics", "relation_order_and_composition",
    "cutoff_and_snapshot_anchoring", "multi_hop_and_uniqueness",
    "no_entity_leakage",
)


def has_infrastructure_error(errors: list[str]) -> bool:
    """Separate retryable provider failures from semantic gate failures."""
    markers = (
        "Wikipedia fetch failed", "HTTP 429", "HTTP 5", "timed out",
        "connection", "temporarily unavailable", "API",
    )
    return any(
        marker.casefold() in error.casefold()
        for error in errors for marker in markers
    )


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class ChainHopSeed:
    source_title: str
    target_title: str
    relation: str
    relative_clause: str
    as_of: str
    evidence: str
    target_aliases: tuple[str, ...]
    incoming_time_policy: str
    property_id: str | None = None
    relation_family: str | None = None
    structured_evidence: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any], index: int) -> "ChainHopSeed":
        required = (
            "source_title", "target_title", "relation", "relative_clause",
            "as_of", "evidence", "target_aliases",
        )
        missing = [field for field in required if not value.get(field)]
        if missing:
            raise ValueError(f"hop {index}: missing fields: {', '.join(missing)}")
        aliases = _strings(value.get("target_aliases"))
        if aliases is None:
            raise ValueError(f"hop {index}: target_aliases must be a non-empty string list")
        clause = str(value["relative_clause"]).strip()
        if clause.count("{source}") != 1:
            raise ValueError(f"hop {index}: relative_clause needs exactly one {{source}}")
        default_policy = "advance_required"
        time_policy = str(value.get("incoming_time_policy", default_policy))
        if time_policy not in TIME_POLICIES:
            raise ValueError(
                f"hop {index}: incoming_time_policy must be one of "
                f"{sorted(TIME_POLICIES)!r}"
            )
        if time_policy == "same_snapshot":
            tail_required = ("property_id", "relation_family", "structured_evidence")
            tail_missing = [field for field in tail_required if not value.get(field)]
            if tail_missing:
                raise ValueError(
                    f"hop {index}: same_snapshot attribute tail missing "
                    f"{', '.join(tail_missing)}"
                )
        _parse_time(str(value["as_of"]))
        return cls(
            source_title=str(value["source_title"]).strip(),
            target_title=str(value["target_title"]).strip(),
            relation=str(value["relation"]).strip(),
            relative_clause=clause,
            as_of=str(value["as_of"]),
            evidence=str(value["evidence"]).strip(),
            target_aliases=tuple(aliases),
            incoming_time_policy=time_policy,
            property_id=(str(value["property_id"]) if value.get("property_id") else None),
            relation_family=(
                str(value["relation_family"]) if value.get("relation_family") else None
            ),
            structured_evidence=(
                dict(value["structured_evidence"])
                if isinstance(value.get("structured_evidence"), dict) else None
            ),
        )


@dataclass(frozen=True)
class MultiHopSeed:
    id: str
    model_id: str
    cutoff: ModelCutoff
    target_as_of: str
    anchor_label: str
    answer_kind: str
    category: str
    hops: tuple[ChainHopSeed, ...]
    old_answer_keywords: tuple[str, ...] = ()
    selection_metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MultiHopSeed":
        for field in ("id", "model_id", "target_as_of", "anchor_label", "hops"):
            if not value.get(field):
                raise ValueError(f"multi-hop seed missing {field}")
        cutoff = get_model_cutoff(str(value["model_id"]))
        supplied_cutoff = value.get("cutoff_date")
        if supplied_cutoff and str(supplied_cutoff) != cutoff.cutoff_date:
            raise ValueError(
                f"{value['id']}: cutoff_date {supplied_cutoff!r} disagrees with pinned "
                f"registry date {cutoff.cutoff_date!r}"
            )
        target_as_of = str(value["target_as_of"])
        if _parse_time(target_as_of) <= _parse_time(cutoff.cutoff_date):
            raise ValueError(f"{value['id']}: target_as_of must be after model cutoff")
        raw_hops = value["hops"]
        if not isinstance(raw_hops, list) or not MIN_HOPS <= len(raw_hops) <= MAX_HOPS:
            raise ValueError(
                f"{value['id']}: reasoning chain must contain {MIN_HOPS}..{MAX_HOPS} hops"
            )
        hops = tuple(ChainHopSeed.from_dict(item, i) for i, item in enumerate(raw_hops))
        if "registered knowledge cutoff" not in hops[0].relative_clause.casefold():
            raise ValueError(
                f"{value['id']}: first relative_clause must name the registered "
                "knowledge cutoff"
            )
        if hops[0].as_of != cutoff.cutoff_date:
            raise ValueError(
                f"{value['id']}: first hop must use cutoff snapshot {cutoff.cutoff_date}"
            )
        if hops[-1].as_of != target_as_of:
            raise ValueError(f"{value['id']}: final hop must use target_as_of")
        if not any(_parse_time(hop.as_of) > _parse_time(cutoff.cutoff_date) for hop in hops[1:]):
            raise ValueError(f"{value['id']}: chain has no post-cutoff hop")
        for index, (left, right) in enumerate(zip(hops, hops[1:])):
            if left.target_title.casefold() != right.source_title.casefold():
                raise ValueError(
                    f"{value['id']}: disconnected hops {index}->{index + 1}: "
                    f"{left.target_title!r} != {right.source_title!r}"
                )
            left_time = _parse_time(left.as_of)
            right_time = _parse_time(right.as_of)
            if right.incoming_time_policy == "advance_required":
                if right_time <= left_time:
                    raise ValueError(
                        f"{value['id']}: hop {index + 1} must use a later snapshot after "
                        f"changing to entity {left.target_title!r}"
                    )
            elif right_time != left_time:
                raise ValueError(
                    f"{value['id']}: same_snapshot hop {index + 1} must retain "
                    f"snapshot {left.as_of}"
                )
        entity_path = [hops[0].source_title, *(hop.target_title for hop in hops)]
        folded_path = [title.casefold() for title in entity_path]
        if len(set(folded_path)) != len(folded_path):
            raise ValueError(f"{value['id']}: reasoning chain contains an entity cycle")
        old = value.get("old_answer_keywords", [])
        if not isinstance(old, list) or not all(isinstance(item, str) for item in old):
            raise ValueError(f"{value['id']}: old_answer_keywords must be a string list")
        return cls(
            id=str(value["id"]), model_id=cutoff.model_id, cutoff=cutoff,
            target_as_of=target_as_of, anchor_label=str(value["anchor_label"]).strip(),
            answer_kind=str(value.get("answer_kind", "Who")).strip() or "Who",
            category=str(value.get("category", "other")), hops=hops,
            old_answer_keywords=tuple(item.strip() for item in old if item.strip()),
            selection_metadata=(
                dict(value["selection_metadata"])
                if isinstance(value.get("selection_metadata"), dict) else None
            ),
        )


def compose_canonical_question(seed: MultiHopSeed) -> str:
    """Return the exact nested logical form used for machine auditing."""
    description = seed.hops[0].relative_clause.replace("{source}", seed.anchor_label)
    for hop in seed.hops[1:]:
        description = hop.relative_clause.replace("{source}", description)
    return f"At the target snapshot, {seed.answer_kind.casefold()} is {description}?"


def _sentence_clause(clause: str, source: str) -> str:
    rendered = " ".join(clause.replace("{source}", source).split()).strip()
    if rendered.casefold().startswith("the "):
        rendered = rendered[4:]
    return rendered.rstrip(".?")


def compose_relative_question(seed: MultiHopSeed) -> str:
    """Render the logical chain as short steps without exposing hidden entities.

    The nested canonical form remains available for audit, but is unsuitable as
    the tested-model prompt: its sentence depth grows with every hop.  This
    renderer keeps the same ordered semantics while each sentence contains only
    one relation.
    """
    first = _sentence_clause(seed.hops[0].relative_clause, seed.anchor_label)
    sentences = [f"Start with {seed.anchor_label}.", f"First, identify the {first}."]
    for hop in seed.hops[1:-1]:
        clause = _sentence_clause(hop.relative_clause, "the entity just identified")
        sentences.append(f"Next, identify the {clause}.")
    final = _sentence_clause(
        seed.hops[-1].relative_clause, "the entity just identified"
    )
    sentences.append(
        f"At the target snapshot, {seed.answer_kind.casefold()} is the {final}?"
    )
    return " ".join(sentences)


def _validated_prior_relation_contrast(
    hop: ChainHopSeed, previous_page, later_page,
) -> bool:
    """Validate a saved semantic-contrast override for a pre-existing target link."""
    structured = hop.structured_evidence or {}
    contrast = structured.get("prior_relation_contrast")
    if not isinstance(contrast, dict):
        return False
    try:
        confidence = float(contrast.get("confidence", 0.0))
    except (TypeError, ValueError):
        return False
    before_evidence = contrast.get("before_evidence")
    required = (
        contrast.get("method") == "independent_llm_semantic_contrast"
        and contrast.get("decision") == "pass"
        and confidence >= 0.8
        and isinstance(contrast.get("judge_model"), str)
        and bool(str(contrast.get("judge_model", "")).strip())
        and contrast.get("property_id") == hop.property_id
        and contrast.get("before_revision_id") == previous_page.revision_id
        and contrast.get("after_revision_id") == later_page.revision_id
        and isinstance(before_evidence, str)
        and bool(before_evidence.strip())
        and _fold(before_evidence) in _fold(previous_page.content)
        and _fold(hop.evidence) in _fold(later_page.content)
    )
    return bool(required)


def validate_chain(
    seed: MultiHopSeed,
    backend,
    *,
    validated_prefix: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Verify the chain, optionally reusing a previously verified exact prefix."""
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    question = compose_relative_question(seed)
    hidden_aliases: list[str] = []
    previous_target_page = None
    seen_state_edges: set[tuple[int, int]] = set()
    start_index = 0
    if validated_prefix:
        if len(validated_prefix) >= len(seed.hops):
            return ["validated prefix must be shorter than the candidate chain"], []
        prefix_fields = (
            "source_title", "target_title", "relation", "relative_clause", "as_of",
            "evidence", "incoming_time_policy", "property_id", "relation_family",
        )
        for index, record in enumerate(validated_prefix):
            hop = seed.hops[index]
            expected = {
                "source_title": hop.source_title,
                "target_title": hop.target_title,
                "relation": hop.relation,
                "relative_clause": hop.relative_clause,
                "as_of": hop.as_of,
                "evidence": hop.evidence,
                "incoming_time_policy": hop.incoming_time_policy,
                "property_id": hop.property_id,
                "relation_family": hop.relation_family,
            }
            if any(record.get(field) != expected[field] for field in prefix_fields):
                return [f"validated prefix differs from candidate at hop {index}"], []
            if list(hop.target_aliases) != record.get("target_aliases"):
                return [f"validated prefix aliases differ at hop {index}"], []
        records = [dict(record) for record in validated_prefix]
        start_index = len(records)
        for hop in seed.hops[:start_index]:
            hidden_aliases.extend([hop.target_title, *hop.target_aliases])
        seen_state_edges = {
            (int(record["source_revision_id"]), int(record["target_revision_id"]))
            for record in records
        }
        previous_hop = seed.hops[start_index - 1]
        try:
            previous_target_page = backend.fetch_page(
                previous_hop.target_title, as_of=previous_hop.as_of
            )
        except WikipediaError as exc:
            return [f"validated prefix target fetch failed: {exc}"], records

    for index in range(start_index, len(seed.hops)):
        hop = seed.hops[index]
        prior_raw_target_absent: bool | None = None
        prior_relation_contrast_verified = False
        try:
            if (
                previous_target_page is not None
                and hop.incoming_time_policy == "same_snapshot"
                and previous_target_page.title.casefold() == hop.source_title.casefold()
                and previous_target_page.as_of == hop.as_of
            ):
                source = previous_target_page
            else:
                source = backend.fetch_page(hop.source_title, as_of=hop.as_of)
            target = backend.fetch_page(hop.target_title, as_of=hop.as_of)
        except WikipediaError as exc:
            errors.append(f"hop {index}: Wikipedia fetch failed: {exc}")
            continue
        if source.title.casefold() != hop.source_title.casefold():
            errors.append(f"hop {index}: source canonical title changed to {source.title!r}")
        if target.title.casefold() != hop.target_title.casefold():
            errors.append(f"hop {index}: target canonical title changed to {target.title!r}")
        if previous_target_page is not None:
            if previous_target_page.title.casefold() != source.title.casefold():
                errors.append(f"hop {index}: fetched chain is disconnected")
            if hop.incoming_time_policy == "advance_required":
                if previous_target_page.revision_id == source.revision_id:
                    errors.append(
                        f"hop {index}: later snapshot resolves to the same source revision"
                    )
                premature_targets = {
                    link.target.casefold() for link in previous_target_page.links
                }
                previous_content = _fold(previous_target_page.content)
                premature_aliases = [
                    alias for alias in hop.target_aliases
                    if _fold(alias) in previous_content
                ]
                prior_raw_target_absent = (
                    target.title.casefold() not in premature_targets
                    and not premature_aliases
                )
                if not prior_raw_target_absent:
                    prior_relation_contrast_verified = _validated_prior_relation_contrast(
                        hop, previous_target_page, source
                    )
                if not prior_raw_target_absent and not prior_relation_contrast_verified:
                    errors.append(
                        f"hop {index}: target appears before the required temporal switch without "
                        "a validated prior-relation semantic contrast"
                    )
            elif previous_target_page.revision_id != source.revision_id:
                errors.append(
                    f"hop {index}: same_snapshot source does not reuse the prior target revision"
                )
        if _fold(hop.evidence) not in _fold(source.content):
            errors.append(f"hop {index}: evidence is not a verbatim source-revision substring")
        evidence_folded = _fold(hop.evidence)
        for alias in hop.target_aliases:
            if _fold(alias) not in evidence_folded:
                errors.append(f"hop {index}: alias {alias!r} is absent from evidence")
        targets = {link.target.casefold() for link in source.links}
        if target.title.casefold() not in targets:
            errors.append(f"hop {index}: source revision has no hyperlink to target")
        edge = (source.revision_id, target.revision_id)
        if edge in seen_state_edges:
            errors.append(f"hop {index}: duplicate page-version edge")
        seen_state_edges.add(edge)
        hidden_aliases.extend([hop.target_title, *hop.target_aliases])
        records.append({
            "index": index,
            "source_title": source.title,
            "source_revision_id": source.revision_id,
            "source_timestamp": source.timestamp,
            "source_url": source.source_url,
            "target_title": target.title,
            "target_revision_id": target.revision_id,
            "target_timestamp": target.timestamp,
            "target_url": target.source_url,
            "as_of": hop.as_of,
            "relation": hop.relation,
            "relative_clause": hop.relative_clause,
            "evidence": hop.evidence,
            "target_aliases": list(hop.target_aliases),
            "incoming_time_policy": hop.incoming_time_policy,
            "property_id": hop.property_id,
            "relation_family": hop.relation_family,
            "structured_evidence": hop.structured_evidence,
            "prior_snapshot_absence_verified": prior_raw_target_absent is True,
            "prior_raw_target_absent": prior_raw_target_absent,
            "prior_relation_contrast_verified": prior_relation_contrast_verified,
            "prior_as_of": seed.hops[index - 1].as_of if index > 0 else None,
            "prior_revision_id": (
                previous_target_page.revision_id
                if previous_target_page is not None else None
            ),
        })
        previous_target_page = target
    folded_question = _fold(question)
    for alias in hidden_aliases:
        if _fold(alias) in folded_question:
            errors.append(f"question leaks hidden entity alias {alias!r}")
    if seed.anchor_label.casefold() not in question.casefold():
        errors.append("question omits the public anchor label")
    final_aliases = {_fold(item) for item in seed.hops[-1].target_aliases}
    if final_aliases & {_fold(item) for item in seed.old_answer_keywords}:
        errors.append("old and target answer aliases overlap")
    return errors, records


def build_temporal_waypoints(chain: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand relation hops into temporal and same-snapshot graph states."""
    waypoints: list[dict[str, Any]] = []
    for index, hop in enumerate(chain):
        if index == 0 or hop.get("incoming_time_policy") != "same_snapshot":
            source = {
                "title": hop["source_title"],
                "revision_id": hop["source_revision_id"],
                "as_of": hop["as_of"],
                "incoming_edge": "start" if index == 0 else "temporal",
                "relation_hop": index,
                "role": "relation_source",
            }
            waypoints.append(source)
        target = {
            "title": hop["target_title"],
            "revision_id": hop["target_revision_id"],
            "as_of": hop["as_of"],
            "incoming_edge": "hyperlink",
            "relation_hop": index,
            "role": "relation_target",
        }
        waypoints.append(target)
    for index, waypoint in enumerate(waypoints):
        waypoint["index"] = index
    return waypoints


class MultiHopQuestionJudge:
    """Independent semantic judge after deterministic revision/link gates."""

    def __init__(
        self, model: str, *, min_confidence: float = 0.8,
        call_model_fn: Callable[..., str] = call_model,
        cache_path: str | None = None,
    ):
        self.model = model
        self.min_confidence = min_confidence
        self.call_model_fn = call_model_fn
        self._cache_lock = threading.Lock()
        self._cache_conn = None
        if cache_path:
            self._cache_conn = sqlite3.connect(cache_path, check_same_thread=False)
            self._cache_conn.execute(
                "CREATE TABLE IF NOT EXISTS judge_cache ("
                "cache_key TEXT PRIMARY KEY, model TEXT NOT NULL, result_json TEXT NOT NULL, "
                "created_at TEXT NOT NULL)"
            )
            self._cache_conn.commit()

    def close(self) -> None:
        if self._cache_conn is not None:
            with self._cache_lock:
                self._cache_conn.close()
                self._cache_conn = None

    def _load_cached(self, cache_key: str) -> dict[str, Any] | None:
        if self._cache_conn is None:
            return None
        with self._cache_lock:
            row = self._cache_conn.execute(
                "SELECT result_json FROM judge_cache WHERE cache_key=?", (cache_key,)
            ).fetchone()
        return dict(json.loads(row[0])) if row else None

    def _store_cached(self, cache_key: str, result: dict[str, Any]) -> None:
        if self._cache_conn is None:
            return
        stored = {key: value for key, value in result.items() if key != "cache_hit"}
        with self._cache_lock:
            self._cache_conn.execute(
                "INSERT OR IGNORE INTO judge_cache("
                "cache_key,model,result_json,created_at) VALUES(?,?,?,?)",
                (
                    cache_key, self.model,
                    json.dumps(stored, ensure_ascii=False, sort_keys=True),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._cache_conn.commit()

    def judge(self, question: str, chain: list[dict[str, Any]]) -> dict[str, Any]:
        visible = [{
            "index": hop["index"], "relation": hop["relation"],
            "relative_clause": hop["relative_clause"], "as_of": hop["as_of"],
            "evidence": hop["evidence"], "source_title": hop["source_title"],
            "target_title": hop["target_title"],
            "incoming_time_policy": hop.get("incoming_time_policy"),
            "property_id": hop.get("property_id"),
            "relation_family": hop.get("relation_family"),
            "structured_evidence": hop.get("structured_evidence"),
        } for hop in chain]
        cache_key = hashlib.sha256(json.dumps({
            "prompt_version": 2,
            "schema_version": SCHEMA_VERSION,
            "model": self.model,
            "min_confidence": self.min_confidence,
            "question": question,
            "chain": visible,
        }, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cached = self._load_cached(cache_key)
        if cached is not None:
            return {**cached, "cache_hit": True}
        prompt = f"""Audit this cutoff-relative multi-hop Wikipedia question.

Question: {question}
Ordered verified chain:
{json.dumps(visible, ensure_ascii=False, indent=2)}

Exact-substring and hyperlink checks already passed. Pass only if:
- every evidence excerpt semantically supports its named relation;
- the question composes all relations once, in the same order;
- "registered knowledge cutoff" anchors hop 0 and "target snapshot" anchors the answer;
- the chain is genuinely multi-hop and uniquely identifies the final target;
- no intermediate or final entity is named in the question.
Return one JSON object with decision (pass/reject), confidence (0..1), checks as an
object with exactly these boolean assertions: evidence_semantics,
relation_order_and_composition, cutoff_and_snapshot_anchoring,
multi_hop_and_uniqueness, no_entity_leakage; also return
reason, and rejected_hops (list of integers).
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
            for key in WHOLE_CHAIN_CHECKS:
                if checks.get(key) is not True:
                    schema_errors.append(f"check_not_true:{key}")
        rejected_hops = raw.get("rejected_hops")
        if not isinstance(rejected_hops, list) or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in rejected_hops
        ):
            schema_errors.append("rejected_hops_not_integer_list")
        elif decision == "pass" and rejected_hops:
            schema_errors.append("pass_has_rejected_hops")
        if not str(raw.get("reason", "")).strip():
            schema_errors.append("missing_reason")
        if decision != "pass" or confidence < self.min_confidence or schema_errors:
            decision = "reject"
        result = {
            **raw, "decision": decision, "confidence": confidence,
            "schema_gate_errors": schema_errors,
            "raw_response": response, "cache_hit": False,
        }
        self._store_cached(cache_key, result)
        return result


def build_case(
    seed: MultiHopSeed, chain: list[dict[str, Any]], judge_result: dict[str, Any]
) -> dict[str, Any]:
    question = compose_relative_question(seed)
    canonical_question = compose_canonical_question(seed)
    required_dates = list(dict.fromkeys([hop.as_of for hop in seed.hops]))
    waypoints = build_temporal_waypoints(chain)
    expected_distance = len(waypoints) - 1
    return {
        "id": seed.id,
        "category": seed.category,
        "wikipedia_title": chain[-1]["target_title"],
        "wikipedia_before": seed.cutoff.cutoff_date,
        "wikipedia_as_of": seed.target_as_of,
        "required_snapshot_dates": required_dates,
        "temporal_question": question,
        "start_title": chain[0]["source_title"],
        "hide_pivot_title": True,
        "reasoning_hop_count": len(chain),
        "expected_navigation_distance": expected_distance,
        "semantic_shortest_distance": expected_distance,
        "required_temporal_switches": sum(
            hop.get("incoming_time_policy", "advance_required") == "advance_required"
            for hop in chain[1:]
        ),
        "temporal_waypoints": waypoints,
        "old_answer_keywords": list(seed.old_answer_keywords),
        "new_answer_keywords": list(seed.hops[-1].target_aliases),
        "knowledge_cutoff": {
            **seed.cutoff.to_dict(), "model_ids": [seed.model_id],
            "role": "candidate_prior_only_pk_gate_is_authoritative",
        },
        "reasoning_chain": chain,
        "relation_families": list(dict.fromkeys(
            str(hop.get("relation_family")) for hop in chain
            if hop.get("relation_family")
        )),
        "selection_metadata": seed.selection_metadata,
        "_generation": {
            "schema_version": SCHEMA_VERSION,
            "status": "machine_pass_human_review_required",
            "question_style": "short_ordered_steps_v1",
            "canonical_question": canonical_question,
            "judge": {key: value for key, value in judge_result.items()
                      if key != "raw_response"},
        },
    }


def validate_shortest_arena(
    seed: MultiHopSeed,
    chain: list[dict[str, Any]],
    backend,
    *,
    branch_cap: int,
    node_cap: int = 500,
) -> dict[str, Any]:
    """Audit the semantic route and diagnose raw shortcuts in a bounded arena."""
    case = build_case(seed, chain, {"decision": "pending", "confidence": 0.0})
    navigation = temporal_reverse_bfs(
        backend,
        case["wikipedia_title"],
        seed.target_as_of,
        case["required_snapshot_dates"],
        case["expected_navigation_distance"],
        branch_cap=branch_cap,
        max_nodes=node_cap,
        required_waypoints=case["temporal_waypoints"],
    )
    contract = validate_chain_route(case, navigation)
    if contract is None:
        raise ValueError("missing reasoning-chain route contract")
    manifest_hash = hashlib.sha256(json.dumps({
        "target_key": navigation["target_key"],
        "states": navigation["states"],
        "arena_edges": navigation["arena_edges"],
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {
        "passed": True,
        "shortest_distance": contract["raw_distance"],
        "raw_shortest_distance": contract["raw_distance"],
        "semantic_shortest_distance": contract["semantic_distance"],
        "raw_shortest_matches_semantic": contract["raw_shortest_match"],
        "route_keys": contract["route_keys"],
        "edge_kinds": contract["edge_kinds"],
        "arena_node_count": len(navigation["states"]),
        "arena_edge_count": len(navigation["arena_edges"]),
        "arena_node_cap": navigation["max_nodes"],
        "arena_truncated": navigation["arena_truncated"],
        "arena_discovery_mode": navigation["discovery_mode"],
        "arena_sha256": manifest_hash,
        "coverage_note": navigation["coverage_note"],
    }


def load_seeds(path: str) -> list[MultiHopSeed]:
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".jsonl"):
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        values = parsed.get("seeds", []) if isinstance(parsed, dict) else parsed
    if not isinstance(values, list):
        raise ValueError("seed file must be a list or {'seeds': [...]} object")
    return [MultiHopSeed.from_dict(value) for value in values]


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate cutoff-relative multi-hop cases from verified Wikipedia links"
    )
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-min-confidence", type=float, default=0.8)
    parser.add_argument(
        "--judge-workers", type=int, default=4,
        help="maximum concurrent LLM judge requests",
    )
    parser.add_argument(
        "--judge-cache-path",
        help="persistent judge cache; default is <cache-path>.judge.db",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--cache-path", default="wikipedia_multihop_generation.db")
    parser.add_argument("--backlink-branch-cap", type=int, default=25)
    parser.add_argument("--arena-node-cap", type=int, default=500)
    parser.add_argument(
        "--request-interval", type=float, default=0.75,
        help="minimum seconds between Wikimedia API calls",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", default="multihop_question_packets.jsonl")
    parser.add_argument("--cases-output", default="generated_multihop_cases.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.judge_min_confidence <= 1:
        parser.error("--judge-min-confidence must be between 0 and 1")
    if args.judge_workers <= 0:
        parser.error("--judge-workers must be > 0")
    if args.backlink_branch_cap <= 0 or args.arena_node_cap <= 0:
        parser.error("--backlink-branch-cap and --arena-node-cap must be > 0")
    if args.request_interval < 0:
        parser.error("--request-interval must be >= 0")
    if not args.validate_only:
        if not args.judge_model:
            parser.error("--judge-model is required unless --validate-only")
        if not os.environ.get("OPENROUTER_API_KEY"):
            parser.error("OPENROUTER_API_KEY is required for judging")
    paths = [args.output] + ([] if args.validate_only else [args.cases_output])
    for path in paths:
        assert_new_output_path(path)
        if Path(path).exists() and not args.overwrite:
            parser.error(f"refusing to overwrite {path}; pass --overwrite explicitly")
    seeds = load_seeds(args.seed_file)
    backend = WikipediaPageBackend(
        cache_path=args.cache_path, lang=args.lang, offline_only=args.offline,
        min_request_interval=args.request_interval,
    )
    judge_cache_path = args.judge_cache_path or f"{args.cache_path}.judge.db"
    judge = None if args.validate_only else MultiHopQuestionJudge(
        args.judge_model, min_confidence=args.judge_min_confidence,
        cache_path=judge_cache_path,
    )
    packets: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    judge_jobs: list[tuple[int, MultiHopSeed]] = []
    try:
        for seed in seeds:
            errors, chain = validate_chain(seed, backend)
            shortest_path = None
            if not errors:
                try:
                    shortest_path = validate_shortest_arena(
                        seed, chain, backend,
                        branch_cap=args.backlink_branch_cap,
                        node_cap=args.arena_node_cap,
                    )
                except (WikipediaError, ValueError) as exc:
                    errors.append(f"shortest_path: {exc}")
            packet: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "packet_id": hashlib.sha256(
                    f"{seed.id}|{seed.model_id}|{seed.target_as_of}".encode()
                ).hexdigest()[:16],
                "seed_id": seed.id,
                "question": compose_relative_question(seed),
                "knowledge_cutoff": seed.cutoff.to_dict(),
                "chain": chain,
                "shortest_path": shortest_path,
                "deterministic_errors": errors,
            }
            if errors:
                packet["status"] = (
                    "infrastructure_error"
                    if has_infrastructure_error(errors)
                    else "deterministic_reject"
                )
            elif args.validate_only:
                packet["status"] = "deterministic_pass"
            else:
                packet["status"] = "judge_pending"
                judge_jobs.append((len(packets), seed))
            packets.append(packet)
            print(
                f"[{packet['status']}] {seed.id}: {len(chain)} verified hops",
                flush=True,
            )

        accepted_by_index: dict[int, dict[str, Any]] = {}
        if judge_jobs:
            assert judge is not None
            with ThreadPoolExecutor(max_workers=args.judge_workers) as executor:
                futures = {
                    executor.submit(
                        judge.judge,
                        str(packets[index]["question"]),
                        list(packets[index]["chain"]),
                    ): (index, seed)
                    for index, seed in judge_jobs
                }
                for future in as_completed(futures):
                    index, seed = futures[future]
                    packet = packets[index]
                    try:
                        judged = future.result()
                    except Exception as exc:
                        packet["status"] = "infrastructure_error"
                        packet["judge_error"] = str(exc)
                        print(
                            f"[infrastructure_error] {seed.id}: judge request failed",
                            flush=True,
                        )
                        continue
                    packet["judge"] = judged
                    packet["status"] = (
                        "machine_pass_human_review_required"
                        if judged["decision"] == "pass" else "judge_reject"
                    )
                    if judged["decision"] == "pass":
                        accepted_by_index[index] = build_case(
                            seed, list(packet["chain"]), judged
                        )
                    print(
                        f"[{packet['status']}] {seed.id}: judge "
                        f"cache_hit={judged.get('cache_hit', False)}",
                        flush=True,
                    )
        accepted = [accepted_by_index[index] for index in sorted(accepted_by_index)]
    finally:
        backend.close()
        if judge is not None:
            judge.close()
    _write_jsonl(args.output, packets)
    if not args.validate_only:
        with open(args.cases_output, "w", encoding="utf-8") as fh:
            json.dump({"cases": accepted}, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    failed_infrastructure = any(
        row.get("status") == "infrastructure_error" for row in packets
    )
    return 0 if (
        all(not row["deterministic_errors"] for row in packets)
        and not failed_infrastructure
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
