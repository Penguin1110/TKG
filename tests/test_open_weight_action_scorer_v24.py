from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from tkg.experiment.joint_controller_v23 import SubmitSlotActionV23
from tkg.experiment.open_weight_action_scorer_v24 import (
    OpenWeightConditionalActionScorerV24,
)
from tkg.experiment.temporal_environment_v2 import EnvironmentActionV2


class FakeLogitBackend:
    backend_name = "fake-causal-logits"

    def conditional_token_logprobs(self, prompt: str, continuation: str):
        del prompt
        if "SUBMIT_SLOT" in continuation:
            return [-0.2, -0.4]
        return [-1.0, -2.0, -3.0]


def test_scores_every_action_by_mean_token_conditional_logprob() -> None:
    actions = [
        EnvironmentActionV2(
            "LIST_LINKS", {"cursor": None, "page_size": 50}, "List links",
        ),
        SubmitSlotActionV23(),
    ]
    result = OpenWeightConditionalActionScorerV24(FakeLogitBackend()).score(
        "public state prompt", actions,
    )
    assert result.score_kind == "length_normalized_conditional_logprob"
    assert result.scores[actions[0].action_id] == pytest.approx(-2.0)
    assert result.scores[actions[1].action_id] == pytest.approx(-0.3)
    assert result.token_counts == {
        actions[0].action_id: 3,
        actions[1].action_id: 2,
    }


class NonFiniteBackend:
    backend_name = "bad"

    def conditional_token_logprobs(self, prompt: str, continuation: str):
        del prompt, continuation
        return [math.nan]


def test_nonfinite_model_logprob_fails_closed() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        OpenWeightConditionalActionScorerV24(NonFiniteBackend()).score(
            "prompt", [SubmitSlotActionV23()],
        )


class FakeBatchBackend:
    backend_name = "fake-batch"

    def conditional_token_logprobs(self, prompt: str, continuation: str):
        raise AssertionError("batch path should be used")

    def conditional_token_logprobs_batch(self, prompt: str, continuations: list[str]):
        del prompt
        return [[-float(index + 1)] for index, _ in enumerate(continuations)]


def test_batch_backend_scores_all_actions() -> None:
    actions = [
        EnvironmentActionV2("LIST_LINKS", {"cursor": None}, "List links"),
        SubmitSlotActionV23(),
    ]
    result = OpenWeightConditionalActionScorerV24(FakeBatchBackend()).score(
        "prompt", actions,
    )
    assert list(result.scores.values()) == [-1.0, -2.0]


@dataclass(frozen=True)
class RenamedAction:
    action_id: str
    kind: str = "FOLLOW_LINK"
    label: str = "Follow hyperlink to Person X"

    def to_dict(self):
        return {
            "action_id": self.action_id, "kind": self.kind,
            "label": self.label, "params": {"page_title": "Person X"},
        }


def test_opaque_action_id_is_not_part_of_scored_text() -> None:
    first = RenamedAction("opaque:first")
    second = RenamedAction("opaque:second")
    scorer = OpenWeightConditionalActionScorerV24(FakeLogitBackend())
    assert (
        scorer.score("prompt", [first]).serialized_actions[first.action_id]
        == scorer.score("prompt", [second]).serialized_actions[second.action_id]
    )
