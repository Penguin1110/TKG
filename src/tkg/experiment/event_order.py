"""Deterministic certificates for temporal "first/next" relation claims."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, Iterable


EVENT_ORDER_SCHEMA = "wikidata-event-order-v1"


def _strict_date(value: Any) -> str:
    text = str(value)
    parsed = date.fromisoformat(text)
    if parsed.isoformat() != text:
        raise ValueError("date must use YYYY-MM-DD")
    return text


def _source_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_event_order_certificate(
    *, boundary_event_date: str, selected_event_date: str,
    selected_target_qid: str, candidate_events: Iterable[dict[str, str]],
    coverage_end: str, source: str, source_query_sha256: str | None = None,
    complete: bool = True,
) -> dict[str, Any]:
    """Build a content-addressed ordering proof over an exhaustive candidate set."""
    normalized = sorted(
        {
            (_strict_date(row["event_date"]), str(row["target_qid"]))
            for row in candidate_events
        }
    )
    events = [
        {"event_date": event_date, "target_qid": target_qid}
        for event_date, target_qid in normalized
    ]
    payload: dict[str, Any] = {
        "boundary_event_date": _strict_date(boundary_event_date),
        "selected_event_date": _strict_date(selected_event_date),
        "selected_target_qid": str(selected_target_qid),
        "candidate_events": events,
        "coverage_end": _strict_date(coverage_end),
        "source": str(source),
        "complete": bool(complete),
    }
    if source_query_sha256:
        payload["source_query_sha256"] = str(source_query_sha256)
    return {
        "schema_version": EVENT_ORDER_SCHEMA,
        **payload,
        "source_sha256": _source_hash(payload),
    }


def event_order_certificate_errors(
    certificate: Any, *, expected_boundary: str | None = None,
    expected_event_date: str | None = None,
    expected_target_qid: str | None = None,
) -> list[str]:
    """Verify that the selected edge is the unique earliest event after boundary."""
    if not isinstance(certificate, dict):
        return ["missing event_order_certificate"]
    if certificate.get("schema_version") != EVENT_ORDER_SCHEMA:
        return ["unsupported event_order_certificate schema"]
    errors: list[str] = []
    try:
        boundary = _strict_date(certificate.get("boundary_event_date"))
        selected_date = _strict_date(certificate.get("selected_event_date"))
        coverage_end = _strict_date(certificate.get("coverage_end"))
    except (TypeError, ValueError):
        return ["event_order_certificate has an invalid date"]
    selected_qid = str(certificate.get("selected_target_qid") or "")
    source = str(certificate.get("source") or "")
    if certificate.get("complete") is not True:
        errors.append("event_order_certificate candidate coverage is incomplete")
    if not source:
        errors.append("event_order_certificate is missing its source")
    if not boundary < selected_date <= coverage_end:
        errors.append("event_order_certificate selected event is outside its boundary")
    if expected_boundary is not None and boundary != expected_boundary:
        errors.append("event_order_certificate boundary does not match the prior event")
    if expected_event_date is not None and selected_date != expected_event_date:
        errors.append("event_order_certificate date does not match the selected edge")
    if expected_target_qid is not None and selected_qid != expected_target_qid:
        errors.append("event_order_certificate target does not match the selected edge")

    raw_events = certificate.get("candidate_events")
    if not isinstance(raw_events, list) or not raw_events:
        errors.append("event_order_certificate has no candidate events")
        events: list[dict[str, str]] = []
    else:
        events = []
        for row in raw_events:
            try:
                event_date = _strict_date(row["event_date"])
                target_qid = str(row["target_qid"])
            except (KeyError, TypeError, ValueError):
                errors.append("event_order_certificate has an invalid candidate event")
                continue
            if not boundary < event_date <= coverage_end or not target_qid:
                errors.append("event_order_certificate candidate is outside coverage")
            events.append({"event_date": event_date, "target_qid": target_qid})
        if events != sorted(events, key=lambda row: (row["event_date"], row["target_qid"])):
            errors.append("event_order_certificate candidates are not sorted")
        if len({(row["event_date"], row["target_qid"]) for row in events}) != len(events):
            errors.append("event_order_certificate contains duplicate candidates")
        if events:
            earliest_date = min(row["event_date"] for row in events)
            earliest_qids = {
                row["target_qid"] for row in events
                if row["event_date"] == earliest_date
            }
            if selected_date != earliest_date:
                errors.append("selected event is not the first event after the boundary")
            if earliest_qids != {selected_qid}:
                errors.append("first event after the boundary is ambiguous or mismatched")

    payload = {
        key: certificate.get(key)
        for key in (
            "boundary_event_date", "selected_event_date", "selected_target_qid",
            "candidate_events", "coverage_end", "source", "complete",
        )
    }
    if certificate.get("source_query_sha256"):
        payload["source_query_sha256"] = certificate["source_query_sha256"]
    if certificate.get("source_sha256") != _source_hash(payload):
        errors.append("event_order_certificate source hash does not match")
    return errors
