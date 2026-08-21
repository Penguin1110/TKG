from __future__ import annotations

import json
from pathlib import Path

from tkg.experiment.compact_joint_controller_v24 import (
    ApiCompactJointControllerV24, COMPACT_JOINT_RESPONSE_SCHEMA_V24,
)
from tkg.experiment.compact_submission_v24 import (
    COMPACT_SUBMISSION_SCHEMA_V24, CompactSubmissionV24,
    compact_submission_from_dict_v24, evaluate_compact_submission_posthoc_v24,
    validate_compact_submission_public_v24,
)
from tkg.experiment.joint_controller_v23 import SUBMIT_SLOT_ID_V23, SubmitSlotActionV23
from tkg.experiment.temporal_environment_v2 import EnvironmentActionV2
from tkg.experiment.temporal_eval_schema_v2 import load_cases_v2
from tkg.experiment.temporal_live_v23_synthetic import (
    SYNTHETIC_CASE_V23, run_joint_synthetic_v23,
)


def _compact_from_success():
    search, _ = run_joint_synthetic_v23()
    bridge = next(
        page for page in search.final_state.collected_evidence
        if page["title"] == "Alternative Route" and page["revision_id"] == 111
    )
    tail = next(
        page for page in search.final_state.collected_evidence
        if page["title"] == "Scientist Z" and page["revision_id"] == 120
    )
    return search, CompactSubmissionV24(
        answer="Harbor City",
        bridge_evidence_ids=(bridge["evidence_id"],),
        tail_evidence_ids=(tail["evidence_id"],),
    )


def test_compact_submission_exact_schema_and_public_gate() -> None:
    search, submission = _compact_from_success()
    parsed = compact_submission_from_dict_v24({
        "schema_version": COMPACT_SUBMISSION_SCHEMA_V24,
        "answer": submission.answer,
        "bridge_evidence_ids": list(submission.bridge_evidence_ids),
        "tail_evidence_ids": list(submission.tail_evidence_ids),
    })
    assert parsed == submission
    validation = validate_compact_submission_public_v24(
        submission, list(search.final_state.collected_evidence),
    )
    assert validation.status == "valid"


def test_private_evaluator_reconstructs_claim_shapes_and_validates() -> None:
    search, submission = _compact_from_success()
    private_case = load_cases_v2(SYNTHETIC_CASE_V23)[0]
    result = evaluate_compact_submission_posthoc_v24(
        case=private_case, submission=submission,
        trajectory_evidence=list(search.final_state.collected_evidence),
        trajectory_actions_valid=True,
    )
    assert result["end_to_end_success"] is True
    assert result["model_supplied_claim_shapes"] is False
    assert result["private_evaluator_supplied_claim_shapes"] is True


def test_compact_submission_rejects_unowned_evidence() -> None:
    search, submission = _compact_from_success()
    invalid = CompactSubmissionV24(
        answer=submission.answer,
        bridge_evidence_ids=("not_owned",),
        tail_evidence_ids=submission.tail_evidence_ids,
    )
    result = validate_compact_submission_public_v24(
        invalid, list(search.final_state.collected_evidence),
    )
    assert result.status == "invalid_evidence_ownership"


def test_v24_prompt_requires_only_answer_and_evidence_ids(tmp_path: Path) -> None:
    search, submission = _compact_from_success()
    actions = [
        EnvironmentActionV2(
            "LIST_LINKS", {"cursor": None, "page_size": 50}, "List links",
        ),
        SubmitSlotActionV23(),
    ]
    response = json.dumps({
        "schema_version": COMPACT_JOINT_RESPONSE_SCHEMA_V24,
        "reasoning_summary": "complete",
        "extracted_entities": ["Scientist Z", "Harbor City"],
        "evidence_notes": [],
        "action_utilities": [
            {"action_id": actions[0].action_id, "utility": 1},
            {"action_id": SUBMIT_SLOT_ID_V23, "utility": 9},
        ],
        "submission": submission.to_dict(),
        "abstain_reason": None,
    })
    controller = ApiCompactJointControllerV24(
        "test", cache_path=tmp_path / "compact.db",
        call_model_fn=lambda *args, **kwargs: response,
    )
    output = controller.control(
        search.public_case, search.final_state, actions, seed=1,
        budget={"max_expansions": 40},
    )
    prompt = output.attempts[0]["prompt"]
    assert output.submission == submission
    assert "bridge_evidence_ids" in prompt
    assert "tail_evidence_ids" in prompt
    assert "Do not restate subject/relation/object/event-time claims" in prompt
    assert "supporting_evidence_ids" not in prompt
    controller.close()

