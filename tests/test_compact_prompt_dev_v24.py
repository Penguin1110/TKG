from __future__ import annotations

from tkg.experiment.compact_prompt_dev_v24 import (
    PROMPT_DEV_SCHEMA_V24, build_compact_prompt_dev_manifest_v24,
)


def test_prompt_dev_set_is_independent_and_non_unlocking() -> None:
    manifest = build_compact_prompt_dev_manifest_v24()
    assert manifest["schema_version"] == PROMPT_DEV_SCHEMA_V24
    assert manifest["status"] == "development_only_not_a_gate"
    assert manifest["may_unlock_abcd"] is False
    assert manifest["state_counts"] == {
        "navigation": 4, "positive_submission": 4,
        "negative_submission": 4, "total": 12,
    }
    ids = {row["state_id"] for row in manifest["states"]}
    assert len(ids) == 12
    assert all("v24-dev" in state_id for state_id in ids)


def test_positive_prompt_dev_requires_only_compact_answer_evidence() -> None:
    manifest = build_compact_prompt_dev_manifest_v24()
    positive = next(
        row for row in manifest["states"]
        if row["development_type"] == "positive_submission"
    )
    assert len(positive["public"]["evidence"]) == 2
    assert "expected_answer" in positive["posthoc_development_label"]
