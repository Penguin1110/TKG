from __future__ import annotations

import pytest

from tkg.experiment.joint_controller_v23 import (
    CallableJointControllerV23, JointCandidateActionV23, SUBMIT_SLOT_ID_V23,
)
from tkg.experiment.temporal_environment_v2 import TemporalWikipediaEnvironmentV2
from tkg.experiment.temporal_live_runner_v23 import (
    LiveSearchConfigV23, run_live_temporal_search_v23,
)
from tkg.experiment.temporal_live_runner_v231 import (
    LIVE_TRAJECTORY_SCHEMA_V231, run_live_temporal_search_v231,
)
from tkg.experiment.temporal_live_v23_synthetic import (
    SyntheticJointBackendV23, public_case_v23,
)


def dead_end_policy(case, state, actions: list[JointCandidateActionV23]):
    del case
    scores = {}
    for action in actions:
        row = action.to_dict()
        if action.action_id == SUBMIT_SLOT_ID_V23:
            score = -9.0
        elif action.kind == "LIST_LINKS":
            score = 0.0
        elif action.kind == "LIST_REVISIONS":
            score = -1.0
        elif row.get("params", {}).get("page_title") == "Joint Distractor 00":
            score = 0.0
        else:
            score = -5.0
        scores[action.action_id] = score
    return scores, None


def _inputs():
    backend = SyntheticJointBackendV23()
    return backend, TemporalWikipediaEnvironmentV2(backend), LiveSearchConfigV23(
        beam_width=1, max_expansions=16, max_actions_per_state=1,
        dense_action_limit=30, seed=23,
    )


def test_frozen_v23_dead_end_crashes_but_v231_terminalizes() -> None:
    backend, environment, config = _inputs()
    with pytest.raises(RuntimeError, match="no retained state"):
        run_live_temporal_search_v23(
            public_case=public_case_v23(), backend=backend,
            environment=environment,
            controller=CallableJointControllerV23(dead_end_policy), config=config,
        )

    backend, environment, config = _inputs()
    result = run_live_temporal_search_v231(
        public_case=public_case_v23(), backend=backend, environment=environment,
        controller=CallableJointControllerV23(dead_end_policy), config=config,
    )
    assert result.schema_version == LIVE_TRAJECTORY_SCHEMA_V231
    assert result.stop_reason == "exhausted_no_legal_progress"
    assert result.final_state.stop_reason == "exhausted_no_legal_progress"
    assert result.final_state.error == ""
    assert result.final_state.submitted is None
    assert result.audit_steps[-1]["complete_trajectory_available"] is True
    assert result.audit_steps[-1]["terminal_status"] == "exhausted_no_legal_progress"
    assert result.to_dict()["score_recomputable"] is True
