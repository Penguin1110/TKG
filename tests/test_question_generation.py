"""Offline tests for temporal Wikipedia question generation."""

from __future__ import annotations

from tkg.experiment.case_validation import validate_case
from tkg.experiment.question_generation import (
    GenerationSeed,
    TemporalQuestionJudge,
    candidate_case,
    generation_prompt,
    revision_diff,
    validate_temporal_candidate,
)


BEFORE = """Example Corp is led by Alice Stone as chief executive officer.

Alice Stone chairs the Example Corp board.

Example Corp employs 100 people.
"""

AFTER = """Example Corp is led by Bob Reed as chief executive officer.

Bob Reed chairs the Example Corp board.

Example Corp employs 100 people.
"""


def _candidate():
    return {
        "id_suffix": "ceo_change",
        "category": "corporate",
        "question": "Who is the current chief executive officer of Example Corp?",
        "old_answer_keywords": ["Alice Stone"],
        "new_answer_keywords": ["Bob Reed"],
        "old_evidence": "Example Corp is led by Alice Stone as chief executive officer.",
        "new_evidence": "Example Corp is led by Bob Reed as chief executive officer.",
        "rationale": "The current officeholder changed.",
    }


def test_revision_diff_separates_changed_and_stable_blocks():
    diff = revision_diff(BEFORE, AFTER)
    assert any("Alice Stone" in block for block in diff["before_changed"])
    assert any("Bob Reed" in block for block in diff["after_changed"])
    assert diff["stable"] == ["Example Corp employs 100 people."]


def test_generator_prompt_requests_only_one_temporal_question():
    prompt = generation_prompt(
        GenerationSeed("Example Corp", "2024-01-01", "2025-01-01"),
        revision_diff(BEFORE, AFTER), 2,
    )
    assert "id_suffix, category, question" in prompt
    assert "ripples" not in prompt.casefold()
    assert "control" not in prompt.casefold()
    assert "pk question" not in prompt.casefold()


def test_deterministic_candidate_gate_accepts_supported_questions():
    assert validate_temporal_candidate(_candidate(), BEFORE, AFTER) == []


def test_deterministic_candidate_gate_rejects_hallucinated_evidence():
    candidate = _candidate()
    candidate["new_evidence"] = "Bob Reed became CEO in a secret announcement."
    errors = validate_temporal_candidate(candidate, BEFORE, AFTER)
    assert any("not a verbatim page substring" in error for error in errors)


def test_independent_judge_confidence_gate():
    def fake_call(model, messages, temperature=0.0):
        return '{"decision":"pass","confidence":0.95,"checks":{},"reason":"ok",' \
               '"rejected_items":[]}'

    passed = TemporalQuestionJudge(
        "judge/model", call_model_fn=fake_call, min_confidence=0.8
    ).judge(_candidate())
    assert passed["decision"] == "pass"

    def low_call(model, messages, temperature=0.0):
        return '{"decision":"pass","confidence":0.4,"checks":{},"reason":"weak",' \
               '"rejected_items":[]}'

    rejected = TemporalQuestionJudge(
        "judge/model", call_model_fn=low_call, min_confidence=0.8
    ).judge(_candidate())
    assert rejected["decision"] == "reject"


def test_accepted_candidate_converts_to_runner_case():
    page_base = {
        "title": "Example Corp", "timestamp": "2025-01-01T00:00:00Z",
        "source_url": "https://example.invalid/?oldid=1",
    }
    case = candidate_case(
        _candidate(),
        GenerationSeed("Example Corp", "2024-01-01", "2025-01-01", "corporate"),
        {**page_base, "revision_id": 1},
        {**page_base, "revision_id": 2},
        {"decision": "pass", "confidence": 0.95},
        0, "generator/model", "judge/model",
    )
    assert case["id"] == "example_corp_ceo_change"
    assert case["wikipedia_before"] == "2024-01-01"
    assert case["temporal_question"].startswith("Who is the current")
    assert "ripples" not in case and "control" not in case
    assert case["_generation"]["status"] == "machine_pass_human_review_required"
    assert validate_case(case) == []


def test_strict_case_validation_rejects_legacy_shape():
    legacy = {
        "id": "old", "pk_question": "Who?", "wikipedia_title": "Example",
        "wikipedia_as_of": "2025-01-01", "old_answer_keywords": ["A"],
        "new_answer_keywords": ["B"],
    }
    errors = validate_case(legacy)
    assert any("missing temporal_question" in error for error in errors)
    assert any("missing wikipedia_before" in error for error in errors)
    assert validate_case(legacy, allow_legacy=True) == []


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"[OK] {test.__name__}")
    print(f"All {len(tests)} question-generation tests passed.")


if __name__ == "__main__":
    main()
