from __future__ import annotations

from tkg.experiment.validity_contract_v25 import (
    bounded_shortcut_diagnostic, validity_contract,
)


def _page(title: str, revision: int, content: str, links: list[str]):
    return {
        "title": title, "revision_id": revision, "content": content,
        "links": [{"target": link, "anchor": link} for link in links],
    }


def _chain():
    return [{
        "source_revision_id": 1, "target_revision_id": 2,
        "source_content_sha256": "a", "evidence": "Anchor links Person",
        "target_title": "Person", "incoming_time_policy": "advance_required",
        "_frozen_source_snapshot": _page("Anchor", 1, "Anchor", ["Person"]),
        "_frozen_target_snapshot": _page("Person", 2, "Person", ["City"]),
    }, {
        "source_revision_id": 3, "target_revision_id": 4,
        "source_content_sha256": "b", "evidence": "Person was born in City",
        "target_title": "City", "incoming_time_policy": "advance_required",
        "prior_snapshot_absence_verified": True,
        "_frozen_source_snapshot": _page("Person", 3, "Person was born in City", ["City"]),
        "_frozen_target_snapshot": _page("City", 4, "City", []),
    }]


def test_validity_does_not_require_global_shortest() -> None:
    contract = validity_contract(
        seed_id="seed", chain=_chain(), question="Where was the successor born?",
        deterministic_errors=[], question_leakage_errors=[],
        cutoff_date="2024-06-01", critical_event_dates=["2025-01-01"],
    )
    assert contract["passed"] is True
    assert contract["shortest_path_is_admission_requirement"] is False


def test_bounded_shortcut_report_is_explicitly_non_global() -> None:
    chain = _chain()
    chain[0]["_frozen_source_snapshot"]["links"].append({
        "target": "City", "anchor": "City",
    })
    report = bounded_shortcut_diagnostic(chain, final_aliases=["City"])
    assert report["shortcut_status"] == "SHORTCUT_FOUND"
    assert report["shortest_path_status"] == "bounded_lower_bound"
    assert report["global_shortest_complete"] is False
    assert report["admission_effect"] == "none"
