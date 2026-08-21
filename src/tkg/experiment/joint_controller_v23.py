"""One-call joint graph-action ranking and optional submission for live v2.3."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from tkg.api.openrouter import call_model
from tkg.experiment.temporal_beam import RankerContractError
from tkg.experiment.temporal_beam_ranker import RankerCache, _extract_json
from tkg.experiment.temporal_eval_schema_v2 import (
    PublicTemporalCaseV2, StructuredSubmissionV2, structured_submission_from_dict,
)
from tkg.experiment.temporal_evaluation_v2 import evidence_id_v2


JOINT_CONTROLLER_PROTOCOL_V23 = "joint_rank_submit_v2.3"
JOINT_RESPONSE_SCHEMA_V23 = "joint-rank-submit-response-v2.3"
SUBMIT_SLOT_ID_V23 = "submit_slot:v1"
FORBIDDEN_INFERENCE_KEY_V23 = re.compile(
    r"(?:accepted.*alias|private|reference|witness|gold|expected_(?:page|revision)|"
    r"distance_to|correctness|evaluator|qid)", re.IGNORECASE,
)
QID_V23 = re.compile(r"\b[QP][1-9]\d*\b")


class JointControllerContractErrorV23(RankerContractError):
    pass


@dataclass(frozen=True)
class SubmitSlotActionV23:
    kind: str = "SUBMIT_SLOT"
    label: str = "Submit a structured answer supported by visible evidence"
    action_id: str = SUBMIT_SLOT_ID_V23
    params: dict[str, Any] | None = None
    environment_order: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind,
            "params": self.params or {},
            "label": self.label,
            "environment_order": self.environment_order,
        }


class JointCandidateActionV23(Protocol):
    @property
    def action_id(self) -> str:
        ...

    @property
    def kind(self) -> str:
        ...

    @property
    def label(self) -> str:
        ...

    def to_dict(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class SubmissionValidationV23:
    status: str
    valid: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "valid": self.valid, "reason": self.reason}


@dataclass(frozen=True)
class JointControllerOutputV23:
    scores: dict[str, float]
    score_kind: str
    reasoning_summary: str
    extracted_entities: tuple[str, ...]
    evidence_notes: tuple[str, ...]
    submission: StructuredSubmissionV2 | None
    abstain_reason: str | None
    submission_validation: SubmissionValidationV23
    attempts: tuple[dict[str, Any], ...]
    controller_protocol: str = JOINT_CONTROLLER_PROTOCOL_V23


class JointRankAndSubmitControllerV23(Protocol):
    controller_name: str

    def control(
        self, public_case: PublicTemporalCaseV2, state: Any,
        actions: list[JointCandidateActionV23], *, seed: int,
        budget: dict[str, int],
    ) -> JointControllerOutputV23:
        ...


def assert_joint_public_payload_v23(payload: Any) -> None:
    """Recursive structural guard applied immediately before every model call."""
    leaves: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if FORBIDDEN_INFERENCE_KEY_V23.search(str(key)):
                    raise AssertionError(f"forbidden joint inference key: {key}")
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)
        elif value is not None:
            leaves.append(str(value))

    walk(payload)
    if QID_V23.search(" ".join(leaves)):
        raise AssertionError("Wikidata identifier leaked into joint inference payload")


def _submission_schema(
    raw: Any,
) -> tuple[StructuredSubmissionV2 | None, SubmissionValidationV23, bool]:
    """Return submission, status, and whether malformed schema warrants retry."""
    if raw is None:
        return None, SubmissionValidationV23(
            "abstained", False, "controller returned submission=null",
        ), False
    if not isinstance(raw, dict):
        return None, SubmissionValidationV23(
            "invalid_schema", False, "submission must be an object or null",
        ), True
    try:
        submission = structured_submission_from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        return None, SubmissionValidationV23(
            "invalid_schema", False, str(exc),
        ), True
    return submission, SubmissionValidationV23(
        "schema_valid_pending_public_gate", False,
        "schema parsed; public evidence gate runs in the runner",
    ), False


def validate_submission_public_v23(
    submission: StructuredSubmissionV2,
    evidence_pages: list[dict[str, Any]],
) -> SubmissionValidationV23:
    answer = " ".join(submission.answer.split())
    if not 1 <= len(answer.split()) <= 8:
        return SubmissionValidationV23(
            "invalid_schema", False, "answer must contain 1 to 8 words",
        )
    claims = [*submission.critical_claims, submission.tail_claim]
    if not submission.critical_claims:
        return SubmissionValidationV23(
            "invalid_claim_fields", False, "at least one critical claim is required",
        )
    if any(not claim.subject.strip() or not claim.relation.strip() or not claim.object.strip()
           for claim in claims):
        return SubmissionValidationV23(
            "invalid_claim_fields", False, "claim fields must be non-empty",
        )
    for claim in claims:
        if claim.event_time is not None and not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", claim.event_time,
        ):
            return SubmissionValidationV23(
                "invalid_event_time_format", False,
                "event_time must be YYYY-MM-DD or null",
            )
    by_id = {
        str(page.get("evidence_id") or evidence_id_v2(page)): page
        for page in evidence_pages
    }
    cited = [evidence_id for claim in claims for evidence_id in claim.supporting_evidence_ids]
    if not cited or any(evidence_id not in by_id for evidence_id in cited):
        return SubmissionValidationV23(
            "invalid_evidence_ownership", False,
            "every cited evidence ID must belong to the current trajectory",
        )
    tail_pages = [
        by_id[evidence_id]
        for evidence_id in submission.tail_claim.supporting_evidence_ids
        if evidence_id in by_id
    ]
    normalized_answer = " ".join(answer.casefold().split()).strip(" .?!")
    if not any(
        normalized_answer in " ".join(str(page.get("content") or "").casefold().split())
        for page in tail_pages
    ):
        return SubmissionValidationV23(
            "invalid_literal_support", False,
            "answer literal is absent from cited tail evidence",
        )
    return SubmissionValidationV23(
        "valid", True, "public schema, ownership, visibility, and literal gates passed",
    )


class ApiJointRankAndSubmitControllerV23:
    controller_name = "api_joint_rank_submit_external_controller_v2.3"

    def __init__(
        self, model: str, *, cache_path: str | Path,
        call_model_fn: Callable = call_model, max_dense_actions: int = 30,
        max_attempts: int = 2, max_evidence_chars: int = 20_000,
    ):
        if max_attempts != 2:
            raise ValueError("v2.3 permits exactly one controller retry")
        self.model = model
        self.cache = RankerCache(cache_path)
        self.call_model_fn = call_model_fn
        self.max_dense_actions = max_dense_actions
        self.max_attempts = max_attempts
        self.max_evidence_chars = max_evidence_chars

    def close(self) -> None:
        self.cache.close()

    def _visible_evidence(self, state: Any) -> list[dict[str, Any]]:
        remaining = self.max_evidence_chars
        result = []
        for page in reversed(state.collected_evidence):
            content = str(page.get("content") or "")[:remaining]
            if not content:
                break
            result.append({
                "evidence_id": page["evidence_id"],
                "title": page["title"],
                "revision_id": page["revision_id"],
                "timestamp": page["timestamp"],
                "content": content,
            })
            remaining -= len(content)
            if remaining <= 0:
                break
        return list(reversed(result))

    def _public_payload(
        self, public_case: PublicTemporalCaseV2, state: Any,
        actions: list[JointCandidateActionV23], seed: int,
        budget: dict[str, int],
    ) -> dict[str, Any]:
        payload = {
            "controller_protocol": JOINT_CONTROLLER_PROTOCOL_V23,
            "question": public_case.question,
            "cutoff_date": public_case.cutoff_date,
            "target_date": public_case.target_date,
            "current_state": {
                "page": state.current_page,
                "revision_id": state.current_revision_id,
                "revision_timestamp": state.current_revision_timestamp,
                "snapshot_as_of": state.snapshot_as_of,
                "reasoning_summary": state.reasoning_summary,
                "extracted_entities": list(state.extracted_entities),
            },
            "visible_evidence": self._visible_evidence(state),
            "candidate_actions": [action.to_dict() for action in actions],
            "search_budget": dict(budget),
            "tie_seed": seed,
        }
        assert_joint_public_payload_v23(payload)
        return payload

    def prompt_for(
        self, public_case: PublicTemporalCaseV2, state: Any,
        actions: list[JointCandidateActionV23], *, seed: int,
        budget: dict[str, int], corrective: bool = False,
    ) -> str:
        payload = self._public_payload(public_case, state, actions, seed, budget)
        retry = (
            "\nCORRECTIVE RETRY: Return valid JSON; score every candidate action ID "
            "exactly once. If submission cannot be formed, use null."
            if corrective else ""
        )
        return f"""Jointly rank graph actions and decide whether visible evidence supports
a complete structured answer. Use only the public payload. Return JSON only and no
hidden chain-of-thought. The fixed submit slot contains no answer. Always score it.
Set submission to null when bridge, event time, tail, or composition evidence is
incomplete. An abstain reason is diagnostic only.

PUBLIC PAYLOAD:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}

Return exactly this schema:
{{
  "schema_version": "{JOINT_RESPONSE_SCHEMA_V23}",
  "reasoning_summary": "short state summary",
  "extracted_entities": ["visible entity"],
  "evidence_notes": ["visible observation"],
  "action_utilities": [
    {{"action_id": "every supplied ID exactly once", "utility": 0.0}}
  ],
  "submission": null,
  "abstain_reason": "why evidence is incomplete, or null when submitting"
}}

When submitting, replace null with a structured-temporal-evidence-submission-v2
object containing answer, critical_claims, tail_claim, event_time, and only visible
supporting_evidence_ids. Utilities must be finite numbers in [-100, 100].{retry}"""

    @staticmethod
    def _parse_ranking(
        raw: dict[str, Any], expected: set[str],
    ) -> dict[str, float]:
        rows = raw.get("action_utilities")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise JointControllerContractErrorV23("action_utilities must be a list")
        ids = [row.get("action_id") for row in rows]
        if len(ids) != len(set(ids)):
            raise JointControllerContractErrorV23("duplicate_joint_action_id")
        returned = set(ids)
        if returned != expected:
            raise JointControllerContractErrorV23(
                f"joint_action_id_coverage_mismatch:missing={len(expected-returned)}:"
                f"unexpected={len(returned-expected)}"
            )
        scores = {}
        for row in rows:
            value = row.get("utility")
            if isinstance(value, bool):
                raise JointControllerContractErrorV23("joint utility is not numeric")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise JointControllerContractErrorV23(
                    "joint utility is not numeric"
                ) from exc
            if not math.isfinite(numeric) or not -100 <= numeric <= 100:
                raise JointControllerContractErrorV23("joint utility is invalid")
            scores[str(row["action_id"])] = numeric
        return scores

    def control(
        self, public_case: PublicTemporalCaseV2, state: Any,
        actions: list[JointCandidateActionV23], *, seed: int,
        budget: dict[str, int],
    ) -> JointControllerOutputV23:
        if not actions or len(actions) > self.max_dense_actions:
            raise JointControllerContractErrorV23("joint candidate count is invalid")
        expected = {action.action_id for action in actions}
        if len(expected) != len(actions) or SUBMIT_SLOT_ID_V23 not in expected:
            raise JointControllerContractErrorV23(
                "joint candidates must be unique and contain submit_slot:v1"
            )
        attempts: list[dict[str, Any]] = []
        last_ranking_error = ""
        for attempt in range(self.max_attempts):
            prompt = self.prompt_for(
                public_case, state, actions, seed=seed, budget=budget,
                corrective=bool(attempt),
            )
            cache_key = hashlib.sha256(json.dumps({
                "model": self.model,
                "protocol": JOINT_CONTROLLER_PROTOCOL_V23,
                "response_schema": JOINT_RESPONSE_SCHEMA_V23,
                "attempt": attempt,
                "prompt": prompt,
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )).hexdigest()
            response = self.cache.get(cache_key)
            cache_hit = response is not None
            if response is None:
                response = self.call_model_fn(
                    self.model,
                    [
                        {"role": "system", "content": (
                            "Jointly score every action and optionally submit. JSON only."
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                self.cache.put(cache_key, response)
            record: dict[str, Any] = {
                "attempt": attempt + 1,
                "cache_key": cache_key,
                "cache_hit": cache_hit,
                "prompt": prompt,
                "raw_response": response,
                "ranking_contract_valid": False,
            }
            try:
                raw = _extract_json(response)
                record["parsed_response"] = raw
                if raw.get("schema_version") != JOINT_RESPONSE_SCHEMA_V23:
                    raise JointControllerContractErrorV23(
                        "joint response schema_version mismatch"
                    )
                scores = self._parse_ranking(raw, expected)
                record["ranking_contract_valid"] = True
                submission, submission_validation, schema_retry = _submission_schema(
                    raw.get("submission")
                )
                record["submission_validation"] = submission_validation.to_dict()
                attempts.append(record)
                if schema_retry and attempt + 1 < self.max_attempts:
                    continue
                entities = raw.get("extracted_entities")
                notes = raw.get("evidence_notes")
                return JointControllerOutputV23(
                    scores=scores,
                    score_kind="api_fallback_utility_softmax_log_score",
                    reasoning_summary=" ".join(
                        str(raw.get("reasoning_summary") or "").split()
                    ),
                    extracted_entities=tuple(
                        " ".join(value.split()) for value in entities
                    ) if isinstance(entities, list) and all(
                        isinstance(value, str) for value in entities
                    ) else (),
                    evidence_notes=tuple(
                        " ".join(value.split()) for value in notes
                    ) if isinstance(notes, list) and all(
                        isinstance(value, str) for value in notes
                    ) else (),
                    submission=submission,
                    abstain_reason=(
                        str(raw.get("abstain_reason"))
                        if raw.get("abstain_reason") is not None else None
                    ),
                    submission_validation=submission_validation,
                    attempts=tuple(attempts),
                )
            except (KeyError, TypeError, ValueError, JointControllerContractErrorV23) as exc:
                last_ranking_error = str(exc)
                record["error"] = last_ranking_error
                attempts.append(record)
        raise JointControllerContractErrorV23(
            "joint_ranking_invalid_after_retry:" + last_ranking_error + ":" +
            json.dumps(attempts, ensure_ascii=False, sort_keys=True)
        )


class CallableJointControllerV23:
    """Deterministic test/open-weight adapter with one state-controller call."""

    controller_name = "callable_joint_rank_submit_v2.3"

    def __init__(
        self,
        control_fn: Callable[
            [PublicTemporalCaseV2, Any, list[JointCandidateActionV23]],
            tuple[dict[str, float], StructuredSubmissionV2 | None],
        ],
        *, score_kind: str = "length_normalized_conditional_logprob",
    ):
        self.control_fn = control_fn
        self.score_kind = score_kind
        self.calls = 0

    def control(
        self, public_case: PublicTemporalCaseV2, state: Any,
        actions: list[JointCandidateActionV23], *, seed: int,
        budget: dict[str, int],
    ) -> JointControllerOutputV23:
        del seed, budget
        self.calls += 1
        scores, submission = self.control_fn(public_case, state, actions)
        expected = {action.action_id for action in actions}
        if set(scores) != expected or len(expected) != len(actions):
            raise JointControllerContractErrorV23("callable joint score coverage mismatch")
        if any(not math.isfinite(float(value)) for value in scores.values()):
            raise JointControllerContractErrorV23("callable joint utility is non-finite")
        status = SubmissionValidationV23(
            "abstained" if submission is None else "schema_valid_pending_public_gate",
            False,
            "deterministic controller output",
        )
        return JointControllerOutputV23(
            scores={key: float(value) for key, value in scores.items()},
            score_kind=self.score_kind,
            reasoning_summary="deterministic visible-state controller",
            extracted_entities=(), evidence_notes=(), submission=submission,
            abstain_reason="incomplete visible evidence" if submission is None else None,
            submission_validation=status,
            attempts=({
                "attempt": 1, "ranking_contract_valid": True,
                "submission_validation": status.to_dict(), "fixture": True,
            },),
        )
