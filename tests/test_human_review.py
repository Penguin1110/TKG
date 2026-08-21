"""Human-review admission must fail closed and preserve explicit waivers."""

from __future__ import annotations

import json

import pytest

from tkg.experiment.human_review import (
    REVIEW_SCHEMA, case_review_sha256, resolve_human_reviews, review_template,
)


def _case() -> dict:
    return {
        "id": "review-case",
        "temporal_question": "Who is the leader?",
        "wikipedia_title": "Leader",
        "wikipedia_before": "2024-01-01",
        "wikipedia_as_of": "2025-01-01",
        "new_answer_keywords": ["New Person"],
        "old_answer_keywords": ["Old Person"],
        "_generation": {"status": "machine_pass_human_review_required"},
    }


def test_review_required_case_fails_closed_without_approval():
    with pytest.raises(ValueError, match="human review is required"):
        resolve_human_reviews(
            [_case()], review_file=None, waive_human_review=False
        )


def test_explicit_cli_waiver_is_never_recorded_as_approval():
    review = resolve_human_reviews(
        [_case()], review_file=None, waive_human_review=True
    )["review-case"]
    assert review["decision"] == "waived_by_user"
    assert review["source"] == "cli_flag"
    assert review["decision"] != "approved"


def test_review_file_is_bound_to_exact_case_hash(tmp_path):
    case = _case()
    path = tmp_path / "review.json"
    path.write_text(json.dumps({
        "schema_version": REVIEW_SCHEMA,
        "reviews": [{
            "case_id": case["id"],
            "case_sha256": case_review_sha256(case),
            "decision": "approved",
            "reviewer": "reviewer@example",
            "reviewed_at": "2026-08-13T12:00:00+08:00",
            "notes": "Checked question, answer, dates, and evidence.",
        }],
    }), encoding="utf-8")
    review = resolve_human_reviews(
        [case], review_file=str(path), waive_human_review=False
    )[case["id"]]
    assert review["decision"] == "approved"

    changed = {**case, "temporal_question": "A changed question?"}
    with pytest.raises(ValueError, match="hash does not match"):
        resolve_human_reviews(
            [changed], review_file=str(path), waive_human_review=False
        )


def test_review_template_is_hash_bound_but_deliberately_not_admissible(tmp_path):
    case = _case()
    draft = review_template([case])
    record = draft["reviews"][0]
    assert record["case_sha256"] == case_review_sha256(case)
    assert case_review_sha256(case) != case_review_sha256({**case, "hide_pivot_title": True})
    assert record["decision"] == "pending"
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(draft), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid human review decision"):
        resolve_human_reviews(
            [case], review_file=str(path), waive_human_review=False
        )
