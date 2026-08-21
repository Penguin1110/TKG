from __future__ import annotations

from tkg.experiment.joint_controller_gate_v231 import (
    GATE_THRESHOLDS_V231, build_fresh_gate_manifest_v231,
)
from tkg.experiment.joint_controller_v23 import assert_joint_public_payload_v23


def test_fresh_gate_has_frozen_balanced_state_counts() -> None:
    manifest = build_fresh_gate_manifest_v231()
    assert manifest["state_counts"] == {
        "navigation": 20,
        "positive_submission": 20,
        "negative_submission": 20,
        "total": 60,
    }
    assert manifest["thresholds"] == GATE_THRESHOLDS_V231


def test_navigation_states_allow_multiple_progress_actions() -> None:
    manifest = build_fresh_gate_manifest_v231()
    navigation = [
        row for row in manifest["states"] if row["gate_type"] == "navigation"
    ]
    assert len(navigation) == 20
    for row in navigation:
        assert len(row["public"]["actions"]) == 30
        assert row["posthoc_label"]["progress_action_count"] == 2
        assert len(row["posthoc_label"]["progress_action_ids"]) == 2


def test_private_gate_labels_are_outside_public_controller_payload() -> None:
    manifest = build_fresh_gate_manifest_v231()
    for row in manifest["states"]:
        public = row["public"]
        assert "posthoc_label" not in public
        assert "gate_type" not in public
        assert "progress_action_ids" not in public
        assert_joint_public_payload_v23(public)
