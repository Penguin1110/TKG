from __future__ import annotations

import json
from pathlib import Path

import pytest

from tkg.experiment.joint_controller_v23 import (
    ApiJointRankAndSubmitControllerV23, JointControllerContractErrorV23,
    JOINT_RESPONSE_SCHEMA_V23, SUBMIT_SLOT_ID_V23, SubmitSlotActionV23,
    assert_joint_public_payload_v23,
)
from tkg.experiment.temporal_environment_v2 import EnvironmentActionV2
from tkg.experiment.temporal_live_runner_v22 import initial_live_state_v22
from tkg.experiment.temporal_live_v23_synthetic import (
    SyntheticJointBackendV23, public_case_v23,
)


def _actions():
    return [
        EnvironmentActionV2(
            "LIST_LINKS", {"cursor": None, "page_size": 50}, "list links",
        ),
        SubmitSlotActionV23(),
    ]


def _response(actions, *, mode: str = "valid") -> str:
    rows = [
        {"action_id": action.action_id, "utility": index}
        for index, action in enumerate(actions)
    ]
    if mode == "missing":
        rows.pop()
    elif mode == "duplicate":
        rows.append(dict(rows[0]))
    elif mode == "unexpected":
        rows.append({"action_id": "unexpected:v1", "utility": 0})
    elif mode == "nonfinite":
        rows[0]["utility"] = "NaN"
    return json.dumps({
        "schema_version": JOINT_RESPONSE_SCHEMA_V23,
        "reasoning_summary": "summary",
        "extracted_entities": [],
        "evidence_notes": [],
        "action_utilities": rows,
        "submission": None,
        "abstain_reason": "incomplete",
    })


def test_joint_controller_scores_graph_and_submit_slot(tmp_path: Path) -> None:
    actions = _actions()
    controller = ApiJointRankAndSubmitControllerV23(
        "test", cache_path=tmp_path / "joint.db",
        call_model_fn=lambda *args, **kwargs: _response(actions),
    )
    state = initial_live_state_v22(public_case_v23(), SyntheticJointBackendV23())
    output = controller.control(
        public_case_v23(), state, actions, seed=23,
        budget={"max_expansions": 10},
    )
    assert set(output.scores) == {actions[0].action_id, SUBMIT_SLOT_ID_V23}
    assert output.submission is None
    assert output.submission_validation.status == "abstained"
    assert len(output.attempts) == 1
    controller.close()


@pytest.mark.parametrize("mode", ["missing", "duplicate", "unexpected", "nonfinite"])
def test_joint_dense_contract_retries_once_then_fails_closed(
    tmp_path: Path, mode: str,
) -> None:
    actions = _actions()
    calls = []

    def invalid(*args, **kwargs):
        del args, kwargs
        calls.append(1)
        return _response(actions, mode=mode)

    controller = ApiJointRankAndSubmitControllerV23(
        "test", cache_path=tmp_path / f"{mode}.db", call_model_fn=invalid,
    )
    state = initial_live_state_v22(public_case_v23(), SyntheticJointBackendV23())
    with pytest.raises(
        JointControllerContractErrorV23, match="joint_ranking_invalid_after_retry",
    ):
        controller.control(
            public_case_v23(), state, actions, seed=23,
            budget={"max_expansions": 10},
        )
    assert len(calls) == 2
    controller.close()


def test_malformed_submission_retries_but_keeps_valid_ranking_on_second_attempt(
    tmp_path: Path,
) -> None:
    actions = _actions()
    calls = []

    def malformed(*args, **kwargs):
        del args, kwargs
        calls.append(1)
        payload = json.loads(_response(actions))
        payload["submission"] = "not-an-object"
        return json.dumps(payload)

    controller = ApiJointRankAndSubmitControllerV23(
        "test", cache_path=tmp_path / "malformed.db", call_model_fn=malformed,
    )
    state = initial_live_state_v22(public_case_v23(), SyntheticJointBackendV23())
    output = controller.control(
        public_case_v23(), state, actions, seed=23,
        budget={"max_expansions": 10},
    )
    assert len(calls) == 2
    assert set(output.scores) == {action.action_id for action in actions}
    assert output.submission is None
    assert output.submission_validation.status == "invalid_schema"
    controller.close()


@pytest.mark.parametrize("key", [
    "private_case", "reference_route", "accepted_final_aliases",
    "posthoc_evaluator_output", "expected_revision",
])
def test_recursive_no_gold_leak_rejects_forbidden_keys(key: str) -> None:
    with pytest.raises(AssertionError, match="forbidden joint inference key"):
        assert_joint_public_payload_v23({"public": {"nested": {key: "secret"}}})


def test_recursive_no_gold_leak_rejects_wikidata_id() -> None:
    with pytest.raises(AssertionError, match="Wikidata"):
        assert_joint_public_payload_v23({"question": "Follow Q123"})

