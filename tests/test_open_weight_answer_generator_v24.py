from __future__ import annotations

from types import SimpleNamespace

from tkg.experiment.open_weight_answer_generator_v24 import (
    EvidenceConditionedAnswerGeneratorV24,
)
from tkg.experiment.temporal_live_v23_synthetic import public_case_v23


class FakeGenerator:
    backend_name = "fake-generator"

    def __init__(self, response: str):
        self.response = response

    def generate_text(self, prompt, *, max_new_tokens, system_prompt):
        assert "ev_bridge" in prompt and system_prompt
        assert max_new_tokens == 192
        return self.response


def _state():
    return SimpleNamespace(collected_evidence=[
        {"evidence_id": "ev_bridge", "title": "Office", "revision_id": 1,
         "timestamp": "2025-01-01", "content": "Ari became director in 2025."},
        {"evidence_id": "ev_tail", "title": "Ari", "revision_id": 2,
         "timestamp": "2024-01-01", "content": "Ari was born in Harbor City."},
    ])


def test_real_generator_parses_and_publicly_validates_compact_payload() -> None:
    backend = FakeGenerator(
        '{"schema_version":"compact-temporal-evidence-submission-v2.4",'
        '"answer":"Harbor City","bridge_evidence_ids":["ev_bridge"],'
        '"tail_evidence_ids":["ev_tail"]}'
    )
    result = EvidenceConditionedAnswerGeneratorV24(backend).propose(
        public_case_v23(), _state(),
    )
    assert result.status == "valid_compact_candidate"
    assert result.submission is not None
    assert result.submission.answer == "Harbor City"


def test_real_generator_abstains_on_incomplete_evidence() -> None:
    backend = FakeGenerator('{"abstain":true}')
    result = EvidenceConditionedAnswerGeneratorV24(backend).propose(
        public_case_v23(), _state(),
    )
    assert result.status == "abstained"
    assert result.submission is None
