from __future__ import annotations

from tkg.experiment.joint_controller_v23 import SUBMIT_SLOT_ID_V23
from tkg.experiment.temporal_live_v23_synthetic import (
    run_joint_synthetic_v23, synthetic_artifact_v23,
)


def _complete_step(result):
    return next(
        step for step in result.audit_steps
        if step.get("parent_state", {}).get("current_page") == "Scientist Z"
    )


def test_positive_complete_evidence_submits_and_scores_trace() -> None:
    result, controller = run_joint_synthetic_v23()
    assert result.final_state.submitted is not None
    assert result.final_state.submitted["answer"] == "Harbor City"
    trace = result.final_state.action_trace[-1]
    assert trace["submit_slot_id"] == SUBMIT_SLOT_ID_V23
    assert trace["payload_canonical_sha256"]
    assert trace["public_submission_validation"]["status"] == "valid"
    assert trace["cumulative_score_after_action"] == result.final_state.cumulative_score
    assert result.to_dict()["score_recomputable"] is True
    assert controller.calls == len(result.audit_steps)


def test_negative_incomplete_states_abstain_and_graph_search_continues() -> None:
    result, _ = run_joint_synthetic_v23(max_expansions=3)
    assert result.final_state.submitted is None
    assert result.expansions == 3
    for step in result.audit_steps:
        slot = next(
            row for row in step["candidate_actions"]
            if row["action_id"] == SUBMIT_SLOT_ID_V23
        )
        assert slot["expanded"] is False
        assert slot["pruning_reason"] == "JOINT_SUBMISSION_ABSTAINED"
        assert any(row["expanded"] for row in step["candidate_actions"] if row is not slot)


def test_invalid_submission_does_not_block_or_consume_graph_expansion() -> None:
    result, _ = run_joint_synthetic_v23(mode="invalid", max_expansions=8)
    step = _complete_step(result)
    slot = next(
        row for row in step["candidate_actions"]
        if row["action_id"] == SUBMIT_SLOT_ID_V23
    )
    graph = [
        row for row in step["candidate_actions"]
        if row["action_id"] != SUBMIT_SLOT_ID_V23 and row["expanded"]
    ]
    assert slot["expanded"] is False
    assert slot["pruning_reason"] == "JOINT_SUBMISSION_PAYLOAD_INVALID"
    assert slot["submission_validation"]["status"] == "invalid_evidence_ownership"
    assert len(graph) == 1
    assert result.final_state.submitted is None


def test_submit_slot_is_always_retained_at_dense_limit_30() -> None:
    result, _ = run_joint_synthetic_v23(max_expansions=2)
    step = result.audit_steps[1]
    compacted = step["action_funnel"]["compacted_ranker_actions"]
    assert len(compacted) == 30
    assert SUBMIT_SLOT_ID_V23 in {row["action_id"] for row in compacted}
    reference = [
        row for row in step["action_funnel"]["solver_retrieved_actions"]
        if row.get("params", {}).get("page_title") == "Reference Route"
    ][0]
    record = next(
        row for row in step["action_funnel"]["compaction_records"]
        if row["action_id"] == reference["action_id"]
    )
    assert record["reason"] == "dense_limit_document_order"


def test_multipath_success_does_not_require_private_reference_route() -> None:
    artifact = synthetic_artifact_v23()
    assert artifact["posthoc_private_evaluation"]["end_to_end_success"] is True
    diagnostics = artifact["posthoc_reference_diagnostics"]
    assert diagnostics["reference_route_recalled"] is False
    assert diagnostics["alternative_valid_route_found"] is True


def test_beam_competition_retains_finished_submit_and_unfinished_follow() -> None:
    result, _ = run_joint_synthetic_v23(
        beam_width=5, max_actions_per_state=2, max_expansions=100,
    )
    step = next(
        row for row in result.audit_steps
        if row.get("parent_state", {}).get("current_page") == "Scientist Z"
        and any(
            candidate.get("params", {}).get("page_title") == "Further Context"
            for candidate in row["candidate_actions"]
        )
    )
    retained = set(step["retained_actions"])
    assert SUBMIT_SLOT_ID_V23 in retained
    follow = next(
        row for row in step["candidate_actions"]
        if row["kind"] == "FOLLOW_LINK" and row["params"]["page_title"] == "Further Context"
    )
    assert follow["action_id"] in retained
    assert result.final_state.submitted is not None

    greedy, _ = run_joint_synthetic_v23(
        beam_width=1, max_actions_per_state=2, max_expansions=24,
    )
    greedy_step = next(
        row for row in greedy.audit_steps
        if row.get("parent_state", {}).get("current_page") == "Scientist Z"
    )
    assert greedy_step["retained_actions"] == [SUBMIT_SLOT_ID_V23]


def test_same_seed_is_reproducible_and_budget_is_strict() -> None:
    left, _ = run_joint_synthetic_v23(max_expansions=4)
    right, _ = run_joint_synthetic_v23(max_expansions=4)
    assert left.to_dict() == right.to_dict()
    assert left.expansions == 4
    assert left.stop_reason == "max_expansions"
    assert left.final_state.submitted is None


def test_v23_artifact_declares_one_joint_call_and_no_standalone_proposer() -> None:
    result, _ = run_joint_synthetic_v23()
    payload = result.to_dict()
    assert payload["controller_protocol"] == "joint_rank_submit_v2.3"
    assert payload["model_calls_per_state_for_rank_and_submit"] == 1
    assert payload["standalone_submission_proposer_used"] is False
