"""Joint graph ranking with compact answer-plus-evidence submission v2.4."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tkg.api.openrouter import call_model
from tkg.experiment.compact_submission_v24 import (
    COMPACT_SUBMISSION_SCHEMA_V24, CompactSubmissionV24,
    CompactSubmissionValidationV24, compact_submission_from_dict_v24,
)
from tkg.experiment.joint_controller_v23 import (
    JointCandidateActionV23, JointControllerContractErrorV23, SUBMIT_SLOT_ID_V23,
    assert_joint_public_payload_v23,
)
from tkg.experiment.temporal_beam_ranker import RankerCache, _extract_json
from tkg.experiment.temporal_eval_schema_v2 import PublicTemporalCaseV2


COMPACT_JOINT_PROTOCOL_V24 = "joint_rank_compact_submit_v2.4"
COMPACT_JOINT_RESPONSE_SCHEMA_V24 = "joint-rank-compact-submit-response-v2.4"


@dataclass(frozen=True)
class CompactJointOutputV24:
    scores: dict[str, float]
    reasoning_summary: str
    extracted_entities: tuple[str, ...]
    evidence_notes: tuple[str, ...]
    submission: CompactSubmissionV24 | None
    submission_schema_validation: CompactSubmissionValidationV24
    abstain_reason: str | None
    attempts: tuple[dict[str, Any], ...]
    score_kind: str = "api_fallback_utility_softmax_log_score"
    controller_protocol: str = COMPACT_JOINT_PROTOCOL_V24


@dataclass(frozen=True)
class CompactSubmitSlotActionV24:
    kind: str = "SUBMIT_SLOT"
    label: str = "Submit a compact answer with visible bridge and tail evidence IDs"
    action_id: str = SUBMIT_SLOT_ID_V23
    params: dict[str, Any] | None = None
    environment_order: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id, "kind": self.kind,
            "params": self.params or {}, "label": self.label,
            "environment_order": self.environment_order,
        }


class ApiCompactJointControllerV24:
    controller_name = "api_joint_rank_compact_submit_external_controller_v2.4"

    def __init__(
        self, model: str, *, cache_path: str | Path,
        call_model_fn: Callable = call_model, max_dense_actions: int = 30,
        max_attempts: int = 2, max_evidence_chars: int = 20_000,
    ):
        if max_attempts != 2:
            raise ValueError("v2.4 permits exactly one retry")
        self.model = model
        self.cache = RankerCache(cache_path)
        self.call_model_fn = call_model_fn
        self.max_dense_actions = max_dense_actions
        self.max_attempts = max_attempts
        self.max_evidence_chars = max_evidence_chars

    def close(self) -> None:
        self.cache.close()

    def _public_payload(
        self, case: PublicTemporalCaseV2, state: Any,
        actions: list[JointCandidateActionV23], seed: int,
        budget: dict[str, int],
    ) -> dict[str, Any]:
        remaining = self.max_evidence_chars
        evidence = []
        for page in reversed(state.collected_evidence):
            content = str(page.get("content") or "")[:remaining]
            if not content:
                break
            evidence.append({
                "evidence_id": page["evidence_id"], "title": page["title"],
                "revision_id": page["revision_id"], "timestamp": page["timestamp"],
                "content": content,
            })
            remaining -= len(content)
        payload = {
            "controller_protocol": COMPACT_JOINT_PROTOCOL_V24,
            "question": case.question, "cutoff_date": case.cutoff_date,
            "target_date": case.target_date,
            "current_state": {
                "page": state.current_page, "revision_id": state.current_revision_id,
                "revision_timestamp": state.current_revision_timestamp,
                "snapshot_as_of": state.snapshot_as_of,
                "reasoning_summary": state.reasoning_summary,
                "extracted_entities": list(state.extracted_entities),
            },
            "visible_evidence": list(reversed(evidence)),
            "candidate_actions": [action.to_dict() for action in actions],
            "search_budget": dict(budget), "tie_seed": seed,
        }
        assert_joint_public_payload_v23(payload)
        return payload

    def prompt_for(
        self, case: PublicTemporalCaseV2, state: Any,
        actions: list[JointCandidateActionV23], *, seed: int,
        budget: dict[str, int], corrective: bool = False,
    ) -> str:
        payload = self._public_payload(case, state, actions, seed, budget)
        retry = (
            "\nCORRECTIVE RETRY: Follow the exact JSON shapes below. Score every "
            "action ID once. Submission must have exactly four fields."
            if corrective else ""
        )
        return f"""Rank every candidate graph action and optionally submit an answer.
Use only visible evidence in PUBLIC PAYLOAD. Return JSON only. Always score the
fixed submit slot. Submit only when cited bridge evidence identifies the requested
post-cutoff event/person and cited tail evidence supports the final answer.

PUBLIC PAYLOAD:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}

Return exactly:
{{
  "schema_version": "{COMPACT_JOINT_RESPONSE_SCHEMA_V24}",
  "reasoning_summary": "short summary",
  "extracted_entities": ["visible entity"],
  "evidence_notes": ["visible observation"],
  "action_utilities": [
    {{"action_id": "every supplied action ID exactly once", "utility": 0.0}}
  ],
  "submission": null,
  "abstain_reason": "missing bridge or tail evidence"
}}

When complete evidence is visible, submission must be exactly:
{{
  "schema_version": "{COMPACT_SUBMISSION_SCHEMA_V24}",
  "answer": "one entity or noun phrase, 1 to 8 words",
  "bridge_evidence_ids": ["visible evidence ID supporting the temporal bridge"],
  "tail_evidence_ids": ["visible evidence ID containing and supporting the answer"]
}}

Do not restate subject/relation/object/event-time claims. The private post-hoc
evaluator checks those. Utilities must be finite numbers in [-100, 100].{retry}"""

    @staticmethod
    def _ranking(raw: dict[str, Any], expected: set[str]) -> dict[str, float]:
        rows = raw.get("action_utilities")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise JointControllerContractErrorV23("action_utilities must be a list")
        ids = [row.get("action_id") for row in rows]
        if len(ids) != len(set(ids)):
            raise JointControllerContractErrorV23("duplicate compact joint action ID")
        returned = set(ids)
        if returned != expected:
            raise JointControllerContractErrorV23(
                f"compact_joint_id_mismatch:missing={len(expected-returned)}:"
                f"unexpected={len(returned-expected)}"
            )
        result = {}
        for row in rows:
            if isinstance(row.get("utility"), bool):
                raise JointControllerContractErrorV23("utility is not numeric")
            try:
                value = float(row.get("utility"))
            except (TypeError, ValueError) as exc:
                raise JointControllerContractErrorV23("utility is not numeric") from exc
            if not math.isfinite(value) or not -100 <= value <= 100:
                raise JointControllerContractErrorV23("utility is invalid")
            result[str(row["action_id"])] = value
        return result

    @staticmethod
    def _submission(raw: Any) -> tuple[
        CompactSubmissionV24 | None, CompactSubmissionValidationV24, bool,
    ]:
        if raw is None:
            return None, CompactSubmissionValidationV24(
                "abstained", False, "controller returned submission=null",
            ), False
        if not isinstance(raw, dict):
            return None, CompactSubmissionValidationV24(
                "invalid_schema", False, "submission must be object or null",
            ), True
        try:
            parsed = compact_submission_from_dict_v24(raw)
        except (KeyError, TypeError, ValueError) as exc:
            return None, CompactSubmissionValidationV24(
                "invalid_schema", False, str(exc),
            ), True
        return parsed, CompactSubmissionValidationV24(
            "schema_valid_pending_public_gate", False, "compact schema parsed",
        ), False

    def control(
        self, case: PublicTemporalCaseV2, state: Any,
        actions: list[JointCandidateActionV23], *, seed: int,
        budget: dict[str, int],
    ) -> CompactJointOutputV24:
        if not actions or len(actions) > self.max_dense_actions:
            raise JointControllerContractErrorV23("compact candidate count invalid")
        expected = {action.action_id for action in actions}
        if len(expected) != len(actions) or SUBMIT_SLOT_ID_V23 not in expected:
            raise JointControllerContractErrorV23(
                "compact candidates must be unique and contain submit slot"
            )
        attempts = []
        last_error = ""
        for attempt in range(self.max_attempts):
            prompt = self.prompt_for(
                case, state, actions, seed=seed, budget=budget,
                corrective=bool(attempt),
            )
            cache_key = hashlib.sha256(json.dumps({
                "model": self.model, "protocol": COMPACT_JOINT_PROTOCOL_V24,
                "attempt": attempt, "prompt": prompt,
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            response = self.cache.get(cache_key)
            cache_hit = response is not None
            if response is None:
                response = self.call_model_fn(
                    self.model,
                    [
                        {"role": "system", "content": (
                            "Rank every action and optionally cite compact evidence. JSON only."
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                self.cache.put(cache_key, response)
            record = {
                "attempt": attempt + 1, "cache_key": cache_key,
                "cache_hit": cache_hit, "prompt": prompt,
                "raw_response": response, "ranking_contract_valid": False,
            }
            try:
                raw = _extract_json(response)
                record["parsed_response"] = raw
                if raw.get("schema_version") != COMPACT_JOINT_RESPONSE_SCHEMA_V24:
                    raise JointControllerContractErrorV23("v2.4 response schema mismatch")
                scores = self._ranking(raw, expected)
                record["ranking_contract_valid"] = True
                submission, validation, retry_schema = self._submission(
                    raw.get("submission")
                )
                record["submission_validation"] = validation.to_dict()
                attempts.append(record)
                if retry_schema and attempt + 1 < self.max_attempts:
                    continue
                entities = raw.get("extracted_entities")
                notes = raw.get("evidence_notes")
                return CompactJointOutputV24(
                    scores=scores,
                    reasoning_summary=" ".join(str(raw.get("reasoning_summary") or "").split()),
                    extracted_entities=tuple(entities) if isinstance(entities, list) and all(
                        isinstance(item, str) for item in entities
                    ) else (),
                    evidence_notes=tuple(notes) if isinstance(notes, list) and all(
                        isinstance(item, str) for item in notes
                    ) else (),
                    submission=submission,
                    submission_schema_validation=validation,
                    abstain_reason=(
                        str(raw["abstain_reason"])
                        if raw.get("abstain_reason") is not None else None
                    ),
                    attempts=tuple(attempts),
                )
            except (KeyError, TypeError, ValueError, JointControllerContractErrorV23) as exc:
                last_error = str(exc)
                record["error"] = last_error
                attempts.append(record)
        raise JointControllerContractErrorV23(
            "compact_joint_ranking_invalid_after_retry:" + last_error + ":" +
            json.dumps(attempts, ensure_ascii=False, sort_keys=True)
        )
