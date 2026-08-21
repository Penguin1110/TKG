from __future__ import annotations

import math

from tkg.experiment.compact_joint_controller_v24 import CompactSubmitSlotActionV24
from tkg.experiment.open_weight_action_scorer_v24 import OpenWeightConditionalActionScorerV24
from tkg.experiment.open_weight_live_controller_v24 import (
    HierarchicalOpenWeightLiveControllerV24, OpenWeightLiveControllerV24,
)
from tkg.experiment.temporal_environment_v2 import EnvironmentActionV2
from tkg.experiment.temporal_live_v23_synthetic import public_case_v23


class FakeBackend:
    backend_name = "fake-logits"

    def conditional_token_logprobs(self, prompt, continuation):
        del prompt
        return [-1.0] if "FOLLOW_LINK" in continuation else [-2.0]


class State:
    current_page = "Page"
    current_revision_id = 1
    reasoning_summary = ""
    collected_evidence = ()


def test_open_weight_controller_returns_dense_actual_logprob_scores() -> None:
    actions = [
        EnvironmentActionV2("FOLLOW_LINK", {"page_title": "Person"}, "Follow Person"),
        CompactSubmitSlotActionV24(),
    ]
    controller = OpenWeightLiveControllerV24(
        OpenWeightConditionalActionScorerV24(FakeBackend()),
        compact_payload_proposer=lambda state: None,
        payload_proposer_name="test-only",
    )
    output = controller.control(
        public_case_v23(), State(), actions, seed=1,
        budget={"max_expansions": 10},
    )
    assert set(output.scores) == {action.action_id for action in actions}
    assert output.score_kind == "length_normalized_conditional_logprob"
    assert output.attempts[0]["ranking_contract_valid"] is True


def test_hierarchical_scores_form_one_probability_distribution() -> None:
    actions = [
        EnvironmentActionV2("FOLLOW_LINK", {"page_title": "Person"}, "Follow Person"),
        EnvironmentActionV2("LIST_REVISIONS", {"cursor": None}, "List revisions"),
        CompactSubmitSlotActionV24(),
    ]
    controller = HierarchicalOpenWeightLiveControllerV24(
        OpenWeightConditionalActionScorerV24(FakeBackend()),
        compact_payload_proposer=lambda state: None,
        payload_proposer_name="test-only",
    )
    output = controller.control(
        public_case_v23(), State(), actions, seed=1,
        budget={"max_expansions": 10},
    )
    assert math.isclose(sum(math.exp(value) for value in output.scores.values()), 1.0)
    assert output.attempts[0]["factorization"].startswith("P(mode|state)")


def test_live_prompt_candidate_listing_is_order_invariant() -> None:
    actions = [
        EnvironmentActionV2("FOLLOW_LINK", {"page_title": "B"}, "Follow B"),
        EnvironmentActionV2("FOLLOW_LINK", {"page_title": "A"}, "Follow A"),
    ]
    first = OpenWeightLiveControllerV24.prompt_for(
        public_case_v23(), State(), actions, {"max_expansions": 10},
    )
    second = OpenWeightLiveControllerV24.prompt_for(
        public_case_v23(), State(), list(reversed(actions)), {"max_expansions": 10},
    )
    assert first == second
