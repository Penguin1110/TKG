from __future__ import annotations

from tkg.experiment.open_weight_live_synthetic_v24 import (
    OpenWeightMultiPathBackendV24, public_case_v24,
)


def test_new_synthetic_has_two_independent_valid_routes() -> None:
    backend = OpenWeightMultiPathBackendV24()
    hub = backend.fetch_page("Aurora Hub", as_of="2024-06-01")
    assert {link.target for link in hub.links} >= {
        "Aurora Museum leadership timeline", "Aurora Museum curator archive",
    }
    assert backend.fetch_revision(711).links[0].target == "Mira Sol"
    assert backend.fetch_revision(721).links[0].target == "Mira Sol"
    assert public_case_v24().start_page == "Aurora Hub"
