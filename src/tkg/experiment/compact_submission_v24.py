"""Compact answer-plus-evidence submission and private post-hoc evaluation v2.4."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from tkg.experiment.temporal_eval_schema_v2 import (
    EvaluationCaseV2, StructuredSubmissionV2, SubmittedClaimV2,
)
from tkg.experiment.temporal_evaluation_v2 import (
    SemanticClaimJudgeV2, evidence_id_v2, validate_structured_submission_v2,
)


COMPACT_SUBMISSION_SCHEMA_V24 = "compact-temporal-evidence-submission-v2.4"


@dataclass(frozen=True)
class CompactSubmissionV24:
    answer: str
    bridge_evidence_ids: tuple[str, ...]
    tail_evidence_ids: tuple[str, ...]
    schema_version: str = COMPACT_SUBMISSION_SCHEMA_V24

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "bridge_evidence_ids": list(self.bridge_evidence_ids),
            "tail_evidence_ids": list(self.tail_evidence_ids),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class CompactSubmissionValidationV24:
    status: str
    valid: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compact_submission_from_dict_v24(value: dict[str, Any]) -> CompactSubmissionV24:
    if value.get("schema_version") != COMPACT_SUBMISSION_SCHEMA_V24:
        raise ValueError("compact submission schema_version mismatch")
    allowed = {
        "schema_version", "answer", "bridge_evidence_ids", "tail_evidence_ids",
    }
    if set(value) != allowed:
        raise ValueError("compact submission fields mismatch")
    answer = value.get("answer")
    bridge = value.get("bridge_evidence_ids")
    tail = value.get("tail_evidence_ids")
    if not isinstance(answer, str):
        raise ValueError("compact answer must be a string")
    if not isinstance(bridge, list) or not all(isinstance(item, str) for item in bridge):
        raise ValueError("bridge_evidence_ids must be a string list")
    if not isinstance(tail, list) or not all(isinstance(item, str) for item in tail):
        raise ValueError("tail_evidence_ids must be a string list")
    return CompactSubmissionV24(
        answer=" ".join(answer.split()),
        bridge_evidence_ids=tuple(dict.fromkeys(bridge)),
        tail_evidence_ids=tuple(dict.fromkeys(tail)),
    )


def validate_compact_submission_public_v24(
    submission: CompactSubmissionV24,
    evidence_pages: list[dict[str, Any]],
) -> CompactSubmissionValidationV24:
    if not 1 <= len(submission.answer.split()) <= 8:
        return CompactSubmissionValidationV24(
            "invalid_answer", False, "answer must contain 1 to 8 words",
        )
    if not submission.bridge_evidence_ids:
        return CompactSubmissionValidationV24(
            "missing_bridge_evidence", False, "bridge evidence IDs are required",
        )
    if not submission.tail_evidence_ids:
        return CompactSubmissionValidationV24(
            "missing_tail_evidence", False, "tail evidence IDs are required",
        )
    by_id = {
        str(page.get("evidence_id") or evidence_id_v2(page)): page
        for page in evidence_pages
    }
    cited = (*submission.bridge_evidence_ids, *submission.tail_evidence_ids)
    if any(evidence_id not in by_id for evidence_id in cited):
        return CompactSubmissionValidationV24(
            "invalid_evidence_ownership", False,
            "all cited evidence IDs must belong to the trajectory",
        )
    answer = " ".join(submission.answer.casefold().split()).strip(" .?!")
    if not any(
        answer in " ".join(str(by_id[evidence_id].get("content") or "").casefold().split())
        for evidence_id in submission.tail_evidence_ids
    ):
        return CompactSubmissionValidationV24(
            "invalid_literal_support", False,
            "answer literal is absent from cited tail evidence",
        )
    return CompactSubmissionValidationV24(
        "valid", True, "answer and trajectory evidence IDs pass the public gate",
    )


def evaluate_compact_submission_posthoc_v24(
    *, case: EvaluationCaseV2, submission: CompactSubmissionV24,
    trajectory_evidence: list[dict[str, Any]], trajectory_actions_valid: bool,
    semantic_judge: SemanticClaimJudgeV2 | None = None,
) -> dict[str, Any]:
    """Private evaluator supplies claim shapes; the model supplies only evidence IDs."""
    expanded = StructuredSubmissionV2(
        answer=submission.answer,
        critical_claims=tuple(
            SubmittedClaimV2(
                subject=claim.subject, relation=claim.relation,
                object=claim.object, event_time=claim.event_time,
                supporting_evidence_ids=submission.bridge_evidence_ids,
                claim_id=claim.claim_id,
            )
            for claim in case.critical_claims
        ),
        tail_claim=SubmittedClaimV2(
            subject=case.tail_relation.subject,
            relation=case.tail_relation.relation,
            object=submission.answer,
            event_time=case.tail_relation.event_time,
            supporting_evidence_ids=submission.tail_evidence_ids,
            claim_id=case.tail_relation.claim_id,
        ),
    )
    result = validate_structured_submission_v2(
        case=case, submission=expanded,
        trajectory_evidence=trajectory_evidence,
        trajectory_actions_valid=trajectory_actions_valid,
        semantic_judge=semantic_judge,
    )
    return {
        **result,
        "submission_schema": COMPACT_SUBMISSION_SCHEMA_V24,
        "model_supplied_claim_shapes": False,
        "private_evaluator_supplied_claim_shapes": True,
        "compact_submission": submission.to_dict(),
        "expanded_private_evaluation_input": json.loads(json.dumps(
            expanded.to_dict(), ensure_ascii=False,
        )),
    }
