from __future__ import annotations

from tkg.experiment.compact_joint_controller_v24 import CompactJointOutputV24
from tkg.experiment.compact_submission_v24 import (
    CompactSubmissionV24, CompactSubmissionValidationV24,
    compact_submission_from_dict_v24, evaluate_compact_submission_posthoc_v24,
)
from tkg.experiment.joint_controller_v23 import SUBMIT_SLOT_ID_V23
from tkg.experiment.temporal_environment_v2 import TemporalWikipediaEnvironmentV2
from tkg.experiment.temporal_eval_schema_v2 import load_cases_v2
from tkg.experiment.temporal_live_runner_v23 import LiveSearchConfigV23
from tkg.experiment.temporal_live_runner_v24 import run_live_temporal_search_v24
from tkg.experiment.temporal_live_v23_synthetic import (
    SYNTHETIC_CASE_V23, SyntheticJointBackendV23, public_case_v23,
)


class CompactFixtureController:
    controller_name = "compact-fixture-v2.4"

    def control(self, case, state, actions, *, seed, budget):
        del case, seed, budget
        complete = (
            any("Scientist Z became Director of Lab" in str(page.get("content"))
                for page in state.collected_evidence)
            and any("Scientist Z was born in Harbor City" in str(page.get("content"))
                    for page in state.collected_evidence)
        )
        scores = {}
        for action in actions:
            row = action.to_dict()
            score = -8.0
            if action.action_id == SUBMIT_SLOT_ID_V23:
                score = 0.0 if complete else -9.0
            elif action.kind == "FOLLOW_LINK" and row["params"].get("page_title") in {
                "Alternative Route", "Scientist Z",
            }:
                score = -0.1
            elif action.kind == "SWITCH_SNAPSHOT" and row["params"].get(
                "revision_id"
            ) == 111:
                score = -0.1
            elif action.kind == "LIST_LINKS":
                score = -0.05 if state.current_revision_id == 111 else -0.2
            elif action.kind == "LIST_REVISIONS":
                score = -0.15 if state.current_page == "Alternative Route" else -1.0
            scores[action.action_id] = score
        submission = None
        status = CompactSubmissionValidationV24(
            "abstained", False, "fixture evidence incomplete",
        )
        if complete:
            bridge = next(
                page for page in state.collected_evidence
                if "Scientist Z became Director of Lab" in str(page.get("content"))
            )
            tail = next(
                page for page in state.collected_evidence
                if "Scientist Z was born in Harbor City" in str(page.get("content"))
            )
            submission = CompactSubmissionV24(
                answer="Harbor City",
                bridge_evidence_ids=(bridge["evidence_id"],),
                tail_evidence_ids=(tail["evidence_id"],),
            )
            status = CompactSubmissionValidationV24(
                "schema_valid_pending_public_gate", False, "fixture parsed",
            )
        return CompactJointOutputV24(
            scores=scores, reasoning_summary="fixture", extracted_entities=(),
            evidence_notes=(), submission=submission,
            submission_schema_validation=status, abstain_reason=None,
            attempts=({"fixture": True},), score_kind="fixture_log_score",
        )


def test_v24_runs_from_start_and_submits_compact_payload() -> None:
    backend = SyntheticJointBackendV23()
    result = run_live_temporal_search_v24(
        public_case=public_case_v23(), backend=backend,
        environment=TemporalWikipediaEnvironmentV2(backend),
        controller=CompactFixtureController(),
        config=LiveSearchConfigV23(
            beam_width=1, max_expansions=16, max_actions_per_state=1,
            dense_action_limit=30, seed=23,
        ),
    )
    assert result.final_state.submitted is not None
    assert result.final_state.submitted == {
        "answer": "Harbor City",
        "bridge_evidence_ids": result.final_state.submitted["bridge_evidence_ids"],
        "tail_evidence_ids": result.final_state.submitted["tail_evidence_ids"],
        "schema_version": "compact-temporal-evidence-submission-v2.4",
    }
    assert result.to_dict()["model_supplied_claim_shapes"] is False
    evaluation = evaluate_compact_submission_posthoc_v24(
        case=load_cases_v2(SYNTHETIC_CASE_V23)[0],
        submission=compact_submission_from_dict_v24(result.final_state.submitted),
        trajectory_evidence=list(result.final_state.collected_evidence),
        trajectory_actions_valid=all(
            row["result"] == "ok" for row in result.final_state.action_trace
        ),
    )
    assert evaluation["end_to_end_success"] is True
