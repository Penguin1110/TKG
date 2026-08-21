"""Open-world structured evidence evaluation and reference-only diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from tkg.experiment.temporal_eval_schema_v2 import (
    EvaluationCaseV2, PrivateReferenceRouteV2, RequiredClaimV2,
    StructuredSubmissionV2, SubmittedClaimV2, normalized,
)


EVALUATION_SCHEMA_V2 = "open-world-temporal-evaluation-result-v2"
DIAGNOSTIC_SCHEMA_V2 = "reference-route-diagnostics-v2"


def evidence_id_v2(page: dict[str, Any]) -> str:
    payload = {
        key: page.get(key)
        for key in ("title", "revision_id", "timestamp", "as_of", "content", "links")
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return "evidence_" + hashlib.sha256(encoded).hexdigest()[:24]


@dataclass(frozen=True)
class SemanticDecisionV2:
    supported: bool | None
    confidence: float
    reason: str
    model: str
    version: str
    judge_input: dict[str, Any]
    judge_output: dict[str, Any]
    deterministic_guards: dict[str, Any]
    review_status: str = "machine_pass_human_review_required"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SemanticClaimJudgeV2(Protocol):
    def judge(
        self, required: RequiredClaimV2, submitted: SubmittedClaimV2,
        evidence: list[dict[str, Any]],
    ) -> SemanticDecisionV2:
        ...


def _claim_shape_matches(
    required: RequiredClaimV2, submitted: SubmittedClaimV2, *,
    allowed_objects: set[str] | None = None,
) -> bool:
    objects = allowed_objects or {normalized(required.object)}
    return (
        normalized(required.subject) == normalized(submitted.subject)
        and normalized(required.relation) == normalized(submitted.relation)
        and normalized(submitted.object) in objects
    )


def _witness_support(
    required: RequiredClaimV2, submitted: SubmittedClaimV2,
    evidence_by_id: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    cited = [
        evidence_by_id[value] for value in submitted.supporting_evidence_ids
        if value in evidence_by_id
    ]
    matched = []
    for witness in required.witnesses:
        excerpt = normalized(witness.evidence_excerpt)
        for page in cited:
            if (
                normalized(page.get("title")) == normalized(witness.page_title)
                and page.get("revision_id") == witness.revision_id
                and excerpt
                and excerpt in normalized(page.get("content"))
            ):
                matched.append({
                    "page_title": witness.page_title,
                    "revision_id": witness.revision_id,
                    "evidence_hash": witness.evidence_hash,
                    "support_type": witness.support_type,
                    "validation_status": witness.validation_status,
                })
    return bool(matched), {
        "policy": "private_multi_witness_semantic_support_v2",
        "matched_witnesses": matched,
    }


def _validate_claim(
    *, required: RequiredClaimV2, submitted: SubmittedClaimV2,
    evidence_by_id: dict[str, dict[str, Any]],
    semantic_judge: SemanticClaimJudgeV2 | None,
    allowed_objects: set[str] | None = None,
) -> dict[str, Any]:
    cited_ids = tuple(dict.fromkeys(submitted.supporting_evidence_ids))
    evidence_owned = bool(cited_ids) and all(value in evidence_by_id for value in cited_ids)
    shape = _claim_shape_matches(
        required, submitted, allowed_objects=allowed_objects,
    )
    event_time_valid = (
        required.event_time is None
        or submitted.event_time == required.event_time
    )
    witness_supported, witness_record = _witness_support(
        required, submitted, evidence_by_id,
    ) if evidence_owned and shape else (False, {
        "policy": "private_multi_witness_semantic_support_v2",
        "matched_witnesses": [],
    })
    semantic_record: dict[str, Any]
    semantic_supported: bool
    if witness_supported:
        semantic_supported = True
        semantic_record = {
            **witness_record,
            "judge_used": False,
            "review_status": "machine_pass_human_review_required",
        }
    elif evidence_owned and shape and semantic_judge is not None:
        decision = semantic_judge.judge(
            required, submitted, [evidence_by_id[value] for value in cited_ids],
        )
        semantic_supported = decision.supported is True
        semantic_record = {
            "policy": "auditable_semantic_judge_v2",
            "judge_used": True,
            "decision": decision.to_dict(),
            "review_status": decision.review_status,
        }
    else:
        semantic_supported = False
        semantic_record = {
            **witness_record,
            "judge_used": False,
            "review_status": "unjudgeable_no_matching_witness_or_judge",
        }
    return {
        "claim_id": required.claim_id,
        "claim_shape_match": shape,
        "evidence_ids_from_trajectory": evidence_owned,
        "semantic_relation_support_passed": semantic_supported,
        "event_time_valid": event_time_valid,
        "support": semantic_record,
        "passed": bool(
            shape and evidence_owned and semantic_supported and event_time_valid
        ),
    }


def validate_structured_submission_v2(
    *, case: EvaluationCaseV2, submission: StructuredSubmissionV2,
    trajectory_evidence: list[dict[str, Any]],
    trajectory_actions_valid: bool,
    semantic_judge: SemanticClaimJudgeV2 | None = None,
) -> dict[str, Any]:
    """Validate any evidence chain; the private reference route is not consulted."""
    evidence_by_id = {
        str(page.get("evidence_id") or evidence_id_v2(page)): page
        for page in trajectory_evidence
    }
    aliases = {normalized(value) for value in case.accepted_final_answer_aliases}
    final_answer_correct = normalized(submission.answer) in aliases

    submitted_by_id = {
        claim.claim_id: claim for claim in submission.critical_claims
        if claim.claim_id
    }
    unused = [claim for claim in submission.critical_claims if not claim.claim_id]
    claim_results = []
    for required in case.critical_claims:
        submitted = submitted_by_id.get(required.claim_id)
        if submitted is None:
            submitted = next((
                claim for claim in unused
                if _claim_shape_matches(required, claim)
            ), None)
            if submitted is not None:
                unused.remove(submitted)
        if submitted is None:
            claim_results.append({
                "claim_id": required.claim_id,
                "claim_shape_match": False,
                "evidence_ids_from_trajectory": False,
                "semantic_relation_support_passed": False,
                "event_time_valid": False,
                "support": {"review_status": "missing_submitted_claim"},
                "passed": False,
            })
            continue
        claim_results.append(_validate_claim(
            required=required,
            submitted=submitted,
            evidence_by_id=evidence_by_id,
            semantic_judge=semantic_judge,
        ))

    tail_objects = {normalized(case.tail_relation.object), *aliases}
    tail_result = _validate_claim(
        required=case.tail_relation,
        submitted=submission.tail_claim,
        evidence_by_id=evidence_by_id,
        semantic_judge=semantic_judge,
        allowed_objects=tail_objects,
    )
    tail_cited = [
        evidence_by_id[value]
        for value in submission.tail_claim.supporting_evidence_ids
        if value in evidence_by_id
    ]
    literal_support = bool(normalized(submission.answer)) and any(
        normalized(submission.answer) in normalized(page.get("content"))
        for page in tail_cited
    )
    critical_complete = bool(claim_results) and all(
        row["passed"] for row in claim_results
    )
    temporal_valid = bool(claim_results) and all(
        row["event_time_valid"] for row in claim_results
    )
    semantic_supported = critical_complete and tail_result["passed"]
    final_bridge_object = (
        normalized(case.critical_claims[-1].object)
        if case.critical_claims else ""
    )
    composition_valid = (
        final_bridge_object == normalized(case.tail_relation.subject)
        and normalized(submission.tail_claim.subject)
        == normalized(case.tail_relation.subject)
        and normalized(submission.tail_claim.object) in tail_objects
        and final_answer_correct
    )
    end_to_end = bool(
        trajectory_actions_valid
        and final_answer_correct
        and literal_support
        and critical_complete
        and tail_result["passed"]
        and temporal_valid
        and composition_valid
    )
    return {
        "schema_version": EVALUATION_SCHEMA_V2,
        "case_id": case.case_id,
        "trajectory_actions_valid": trajectory_actions_valid,
        "final_answer_correct": final_answer_correct,
        "literal_support_gate_passed": literal_support,
        "critical_claim_results": claim_results,
        "critical_bridge_count": len(claim_results),
        "critical_bridges_acquired": sum(bool(row["passed"]) for row in claim_results),
        "critical_bridge_acquisition_rate": (
            sum(bool(row["passed"]) for row in claim_results) / len(claim_results)
            if claim_results else 0.0
        ),
        "critical_bridge_evidence_complete": critical_complete,
        "tail_claim_result": tail_result,
        "semantically_supported_submission": semantic_supported,
        "temporally_valid_submission": temporal_valid,
        "composition_valid": composition_valid,
        "end_to_end_validated_answer_accuracy": int(end_to_end),
        "end_to_end_success": end_to_end,
        "evaluation_status": (
            "machine_pass_human_review_required" if end_to_end
            else "machine_validation_failed"
        ),
    }


def _action_matches(action: dict[str, Any], reference: dict[str, Any]) -> bool:
    if action.get("kind") != reference.get("kind"):
        return False
    params = action.get("params")
    if not isinstance(params, dict):
        return False
    if reference.get("kind") == "FOLLOW_LINK":
        return normalized(params.get("page_title")) == normalized(
            reference.get("page_title")
        )
    if reference.get("kind") == "SWITCH_SNAPSHOT":
        try:
            left = params.get("revision_id")
            right = reference.get("revision_id")
            if left is None or right is None:
                return False
            return int(left) == int(right)
        except (TypeError, ValueError):
            return False
    return False


def reference_route_diagnostics_v2(
    *, route: PrivateReferenceRouteV2 | None,
    funnel_steps: list[dict[str, Any]], action_trace: list[dict[str, Any]],
    case: EvaluationCaseV2, evaluation: dict[str, Any],
    search_stop_reason: str,
) -> dict[str, Any]:
    """Conditional diagnostics; no value here can invalidate open-world success."""
    details = []
    route_actions = list(route.actions) if route is not None else []
    for reference_action in route_actions:
        expected = reference_action.to_dict()
        matching_steps = [
            step for step in funnel_steps
            if normalized(step.get("parent_page")) == normalized(
                reference_action.parent_page
            )
            and step.get("parent_revision_id") == reference_action.parent_revision_id
        ]
        environment = any(
            any(_action_matches(row, expected) for row in step.get(
                "environment_legal_actions", []
            )) for step in matching_steps
        )
        retrieved = any(
            any(_action_matches(row, expected) for row in step.get(
                "solver_retrieved_actions", []
            )) for step in matching_steps
        )
        compacted_rows = [
            row for step in matching_steps
            for row in step.get("compacted_ranker_actions", [])
            if _action_matches(row, expected)
        ]
        compacted = bool(compacted_rows)
        valid_rank_steps = [
            step for step in matching_steps
            if step.get("ranker_contract_valid") is True
            and any(_action_matches(row, expected) for row in step.get(
                "compacted_ranker_actions", []
            ))
        ]
        ranks: list[int] = []
        for step in valid_rank_steps:
            scores = step.get("ranker_scores", {})
            ordered = sorted(scores, key=lambda key: (-float(scores[key]), key))
            matching_ids = {
                row.get("action_id") for row in step.get("compacted_ranker_actions", [])
                if _action_matches(row, expected)
            }
            ranks.extend(index + 1 for index, key in enumerate(ordered) if key in matching_ids)
        expanded = any(
            row.get("action_id") in set(step.get("expanded_actions", []))
            for step in matching_steps for row in compacted_rows
        )
        details.append({
            "kind": reference_action.kind,
            "parent_page": reference_action.parent_page,
            "parent_revision_id": reference_action.parent_revision_id,
            "environment_legal": environment,
            "solver_retrieved": retrieved,
            "compacted_for_ranker": compacted,
            "ranker_evaluable": bool(valid_rank_steps),
            "rank_of_reference_action": min(ranks) if ranks else "not_evaluable",
            "beam_expanded": expanded if valid_rank_steps else "not_evaluable",
        })

    trace_index = 0
    for row in action_trace:
        if trace_index >= len(route_actions):
            break
        action = row.get("action", row)
        if _action_matches(action, route_actions[trace_index].to_dict()):
            trace_index += 1
    route_completed = bool(route_actions) and trace_index == len(route_actions)
    reference_links = [row for row in details if row["kind"] == "FOLLOW_LINK"]
    reference_revisions = [row for row in details if row["kind"] == "SWITCH_SNAPSHOT"]
    labels = []
    if reference_links and not all(row["solver_retrieved"] for row in reference_links):
        labels.append("REFERENCE_LINK_NOT_RECALLED")
    if reference_revisions and not all(
        row["solver_retrieved"] for row in reference_revisions
    ):
        labels.append("REFERENCE_WITNESS_REVISION_NOT_RECALLED")
    if route_actions and not route_completed:
        labels.append("REFERENCE_ROUTE_NOT_COMPLETED")
    if any(
        row["solver_retrieved"] and not row["compacted_for_ranker"]
        for row in details
    ):
        labels.append("REFERENCE_ACTION_NOT_RANKED")
    if any(
        row["compacted_for_ranker"] and not row["ranker_evaluable"] for row in details
    ):
        labels.append("RANKER_CONTRACT_FAILURE")
    if not evaluation.get("end_to_end_success"):
        if search_stop_reason in {"max_expansions", "max_steps", "search_budget_exhausted"}:
            labels.append("SEARCH_BUDGET_EXHAUSTED")
        labels.append("NO_VALIDATED_EVIDENCE_CHAIN_FOUND")
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_V2,
        "reference_route_usage": "diagnostic_only_not_a_unique_gold_trajectory",
        "reference_action_details": details,
        "reference_link_retrieval_rate": (
            sum(bool(row["solver_retrieved"]) for row in reference_links)
            / len(reference_links)
            if reference_links else None
        ),
        "reference_witness_revision_retrieval_rate": (
            sum(bool(row["solver_retrieved"]) for row in reference_revisions)
            / len(reference_revisions) if reference_revisions else None
        ),
        "reference_route_completion_rate": (
            trace_index / len(route_actions) if route_actions else None
        ),
        "reference_route_recalled": route_completed,
        "alternative_valid_route_found": bool(
            evaluation.get("end_to_end_success") and not route_completed
        ),
        "diagnostic_labels": list(dict.fromkeys(labels)),
    }


_PRIVATE_KEY = re.compile(
    r"(?:accepted.*alias|required_critical_claim|gold_critical_claim|witness_set|"
    r"reference_route|gold|private|qid|"
    r"distance_to_reference|expected_next|reference_revision)", re.IGNORECASE,
)


def assert_no_gold_leak_v2(
    payload: Any, *, private_case: EvaluationCaseV2,
    visible_evidence_text: str = "",
) -> None:
    """Fail if a public inference payload contains private schema or values."""
    public_text = json.dumps(
        private_case.public_view().to_dict(), ensure_ascii=False, sort_keys=True,
    ) + " " + visible_evidence_text
    public_folded = normalized(public_text)
    leaves: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if _PRIVATE_KEY.search(str(key)):
                    raise AssertionError(f"private inference key leaked: {key}")
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)
        elif value is not None:
            leaves.append(str(value))

    walk(payload)
    serialized = " ".join(leaves)
    if re.search(r"\b[QP][1-9]\d*\b", serialized):
        raise AssertionError("Wikidata identifier leaked into inference payload")
    secrets = set(private_case.accepted_final_answer_aliases)
    for claim in (*private_case.critical_claims, private_case.tail_relation):
        secrets.update((claim.subject, claim.object))
        secrets.update(str(witness.revision_id) for witness in claim.witnesses)
    for route in private_case.reference_routes:
        for action in route.actions:
            secrets.add(str(action.revision_id or ""))
            secrets.add(str(action.page_title or ""))
    leaf_values = {normalized(value) for value in leaves}
    for secret in secrets:
        folded = normalized(secret)
        if folded and folded not in public_folded and folded in leaf_values:
            raise AssertionError(f"private inference value leaked: {secret}")


def aggregate_primary_metrics_v2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "end_to_end_validated_answer_accuracy": None,
            "critical_bridge_acquisition_rate": None,
            "semantically_supported_submission_rate": None,
            "temporally_valid_submission_rate": None,
        }
    return {
        "n": len(rows),
        "end_to_end_validated_answer_accuracy": sum(
            int(row.get("end_to_end_success") is True) for row in rows
        ) / len(rows),
        "critical_bridge_acquisition_rate": sum(
            float(row.get("critical_bridge_acquisition_rate", 0.0)) for row in rows
        ) / len(rows),
        "semantically_supported_submission_rate": sum(
            int(row.get("semantically_supported_submission") is True) for row in rows
        ) / len(rows),
        "temporally_valid_submission_rate": sum(
            int(row.get("temporally_valid_submission") is True) for row in rows
        ) / len(rows),
    }
