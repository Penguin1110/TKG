"""Append-only v2.5 validity and bounded search-space diagnostics.

These helpers deliberately keep reference-route validity separate from global
shortest-path discovery.  They operate only on already fetched exact revisions
and never make network requests.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence


SCHEMA_VERSION = "temporal-wikipedia-validity-v2.5"
SHORTEST_STATUSES = frozenset({
    "exact", "bounded_lower_bound", "incomplete", "not_computed",
})


def _fold(value: object) -> str:
    return " ".join(str(value).casefold().split())


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def bounded_shortcut_diagnostic(
    chain: Sequence[dict[str, Any]], *, final_aliases: Sequence[str],
) -> dict[str, Any]:
    """Run cheap, gold-route-aware checks over the fetched revision bundle.

    This is an offline diagnostic.  It does not claim global completeness and
    must not be used to change admission or action scoring.
    """
    findings: list[dict[str, Any]] = []
    pages: dict[tuple[str, int], dict[str, Any]] = {}
    for hop in chain:
        for key in ("_frozen_source_snapshot", "_frozen_target_snapshot"):
            page = hop.get(key)
            if isinstance(page, dict):
                pages[(_fold(page.get("title")), int(page.get("revision_id", 0)))] = page

    if chain:
        start = chain[0].get("_frozen_source_snapshot") or {}
        content = _fold(start.get("content", ""))
        leaked = sorted(alias for alias in final_aliases if _fold(alias) in content)
        if leaked:
            findings.append({
                "kind": "start_revision_contains_final_answer",
                "page": start.get("title"), "revision_id": start.get("revision_id"),
                "matched_aliases": leaked,
            })

    # Detect direct links from a route revision to a non-adjacent later route
    # entity.  This is a bounded raw-graph shortcut, not a semantic proof that
    # the question can be answered without the bridge.
    route_targets = [_fold(hop.get("target_title")) for hop in chain]
    for index, hop in enumerate(chain):
        source = hop.get("_frozen_source_snapshot") or {}
        links = {
            _fold(link.get("target")) for link in source.get("links", [])
            if isinstance(link, dict)
        }
        for later_index in range(index + 1, len(route_targets)):
            if route_targets[later_index] in links:
                findings.append({
                    "kind": "non_adjacent_route_link",
                    "from_hop": index, "to_hop": later_index,
                    "page": source.get("title"),
                    "revision_id": source.get("revision_id"),
                    "target": chain[later_index].get("target_title"),
                })

    label = "SHORTCUT_FOUND" if findings else "NO_SHORTCUT_FOUND_WITHIN_BOUND"
    report = {
        "schema_version": SCHEMA_VERSION,
        "shortest_path_status": "bounded_lower_bound",
        "shortcut_status": label,
        "findings": findings,
        "bounded_shortest_distance": None,
        "shorter_alternative_found": bool(findings),
        "cutoff_graph_shortcut_found": bool(findings),
        "global_shortest_complete": False,
        "bfs_explored_nodes": len(pages),
        "frontier_incomplete": True,
        "audit_bound": {
            "scope": "exact reference-route revisions",
            "checks": [
                "start revision contains final-answer alias",
                "non-adjacent route entity is directly hyperlinked",
            ],
        },
        "admission_effect": "none",
        "coverage_note": (
            "No global Wikipedia shortest-path claim. Only the fetched exact "
            "reference-route revisions were inspected."
        ),
    }
    report["diagnostic_sha256"] = _canonical_sha256(report)
    return report


def validity_contract(
    *, seed_id: str, chain: Sequence[dict[str, Any]], question: str,
    deterministic_errors: Sequence[str], question_leakage_errors: Sequence[str],
    cutoff_date: str | None = None,
    critical_event_dates: Sequence[str] = (),
) -> dict[str, Any]:
    """Materialize the explicit v2.5 admission assertions."""
    hop_count = len(chain)
    route_links = hop_count > 0 and all(
        hop.get("source_revision_id") and hop.get("target_revision_id")
        for hop in chain
    )
    exact_evidence = hop_count > 0 and all(
        hop.get("evidence") and hop.get("source_content_sha256")
        for hop in chain
    )
    temporal_hops = [
        hop for index, hop in enumerate(chain)
        if index > 0 and hop.get("incoming_time_policy") == "advance_required"
    ]
    temporal_contrast = bool(temporal_hops) and all(
        hop.get("prior_snapshot_absence_verified") is True
        or hop.get("prior_relation_contrast_verified") is True
        for hop in temporal_hops
    )
    event_dates_post_cutoff = bool(critical_event_dates) and cutoff_date is not None and all(
        str(event_date)[:10] > str(cutoff_date)[:10]
        for event_date in critical_event_dates
    )
    checks = {
        "reference_route_hyperlinks_legal": bool(route_links),
        "exact_revision_evidence_exists": bool(exact_evidence),
        "critical_bridge_post_cutoff": temporal_contrast and event_dates_post_cutoff,
        "event_time_v6_valid": not any("event-order" in error for error in deterministic_errors),
        "bridge_tail_composable": hop_count >= 2,
        "question_has_no_hidden_answer_leakage": not question_leakage_errors,
        "executable_route_exists": bool(route_links and exact_evidence),
    }
    passed = all(checks.values()) and not deterministic_errors
    contract = {
        "schema_version": SCHEMA_VERSION,
        "seed_id": seed_id,
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "checks": checks,
        "passed": passed,
        "deterministic_errors": list(deterministic_errors),
        "question_leakage_errors": list(question_leakage_errors),
        "shortest_path_is_admission_requirement": False,
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    return contract
