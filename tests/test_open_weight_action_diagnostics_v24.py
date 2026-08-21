from __future__ import annotations

from tkg.experiment.open_weight_action_diagnostics_v24 import _navigation_state


def test_heldout_navigation_has_multiple_progress_actions() -> None:
    prompt, actions, progress = _navigation_state(3)
    assert "Heldout Navigator 3" in prompt
    assert len(progress) == 2
    assert len(actions) == 9
    assert progress <= {action.action_id for action in actions}
