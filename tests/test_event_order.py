"""World-event ordering certificates for first/next temporal edges."""

from tkg.experiment.event_order import (
    build_event_order_certificate, event_order_certificate_errors,
)


def test_event_order_certificate_proves_unique_earliest_candidate():
    certificate = build_event_order_certificate(
        boundary_event_date="2024-07-05",
        selected_event_date="2025-09-05",
        selected_target_qid="Q291057",
        candidate_events=[
            {"event_date": "2026-07-20", "target_qid": "Q999"},
            {"event_date": "2025-09-05", "target_qid": "Q291057"},
        ],
        coverage_end="2026-08-01", source="test_fixture",
    )
    assert event_order_certificate_errors(
        certificate, expected_boundary="2024-07-05",
        expected_event_date="2025-09-05", expected_target_qid="Q291057",
    ) == []


def test_event_order_certificate_rejects_intervening_event_and_hash_tamper():
    certificate = build_event_order_certificate(
        boundary_event_date="2024-06-01",
        selected_event_date="2025-09-05",
        selected_target_qid="Q291057",
        candidate_events=[
            {"event_date": "2024-07-05", "target_qid": "Q534727"},
            {"event_date": "2025-09-05", "target_qid": "Q291057"},
        ],
        coverage_end="2026-08-01", source="test_fixture",
    )
    errors = event_order_certificate_errors(certificate)
    assert "selected event is not the first event after the boundary" in errors
    certificate["selected_event_date"] = "2025-09-06"
    errors = event_order_certificate_errors(certificate)
    assert "event_order_certificate source hash does not match" in errors
