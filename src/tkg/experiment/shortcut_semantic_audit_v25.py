"""Post-hoc semantic audit of bounded shortcut signals.

Private/reference information is used only after generation and PK admission.
Nothing emitted here may affect inference, compaction, ranking, or beam scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tkg.wikipedia.backend import WikipediaError, WikipediaPageBackend


SCHEMA_VERSION = "shortcut-semantic-audit-v2.5"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fold(value: object) -> str:
    return " ".join(str(value).casefold().split())


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _load_cases(paths: Iterable[Path]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8")).get("cases", [])
        cases.extend(dict(row) for row in rows)
    ids = [str(case["id"]) for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case IDs overlap across admitted inputs")
    return sorted(cases, key=lambda row: str(row["id"]))


def freeze_inputs(
    *, case_paths: list[Path], ledger_paths: list[Path], output_dir: Path,
    method_manifest: Path | None = None,
) -> tuple[Path, Path]:
    frozen_cases = output_dir / "frozen_pk_admitted_cases.json"
    manifest_path = output_dir / "freeze_manifest.json"
    if frozen_cases.exists() or manifest_path.exists():
        raise FileExistsError("freeze outputs already exist; never overwrite a freeze")
    cases = _load_cases(case_paths)
    _atomic(frozen_cases, {
        "schema_version": "frozen-pk-admitted-cases-v2.5",
        "frozen_at": _now(), "count": len(cases), "cases": cases,
    })
    inputs = [*case_paths, *ledger_paths]
    if method_manifest is not None:
        inputs.append(method_manifest)
    manifest = {
        "schema_version": "temporal-eval-freeze-v2.5",
        "frozen_at": _now(), "case_count": len(cases),
        "case_ids": [case["id"] for case in cases],
        "inputs": [{"path": str(path.resolve()), "sha256": _sha(path)} for path in inputs],
        "frozen_cases": {"path": str(frozen_cases.resolve()), "sha256": _sha(frozen_cases)},
        "policy": {
            "machine_validity": "v6 route/evidence/event-time",
            "shortest_is_admission_gate": False,
            "shortcut_audit_is_posthoc_only": True,
            "formal_abcd_authorized": False,
        },
    }
    _atomic(manifest_path, manifest)
    return frozen_cases, manifest_path


def _ledger_rows(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                result[str(row["seed_id"])] = row
    return result


def _node(page: dict[str, Any], *, as_of: str | None) -> dict[str, Any]:
    return {
        "page": page.get("title"), "revision_id": page.get("revision_id"),
        "revision_date": page.get("timestamp"), "as_of": as_of,
    }


def _action(kind: str, source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": kind, "from": _node(source, as_of=source.get("as_of")),
        "to": _node(target, as_of=target.get("as_of")),
    }


def _frozen_pages(case: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    pages = case.get("frozen_wikipedia_evidence", {}).get("pages", [])
    return {
        (_fold(page.get("title")), int(page.get("revision_id"))): dict(page)
        for page in pages
    }


def _path_support(
    case: dict[str, Any], pages: list[dict[str, Any]], *, cutoff_only: bool = False,
) -> dict[str, Any]:
    cutoff = str(case["knowledge_cutoff"]["cutoff_date"])[:10]
    texts = []
    for page in pages:
        if cutoff_only and str(page.get("as_of") or "9999")[:10] > cutoff:
            continue
        texts.append(_fold(page.get("content", "")))
    chain = case["reasoning_chain"]
    probes = case["prior_knowledge_contract"]["probes"]
    bridge_indices = [int(probe["hop_index"]) for probe in probes if probe.get("role") == "critical_bridge"]
    tail_indices = [int(probe["hop_index"]) for probe in probes if probe.get("role") == "tail"]
    def supported(index: int) -> bool:
        evidence = _fold(chain[index].get("evidence", ""))
        return bool(evidence) and any(evidence in text for text in texts)
    return {
        "critical_bridge_hop_indices": bridge_indices,
        "bridge_literal_support": {str(index): supported(index) for index in bridge_indices},
        "all_bridges_literal_supported": bool(bridge_indices) and all(supported(index) for index in bridge_indices),
        "tail_literal_support": {str(index): supported(index) for index in tail_indices},
        "tail_literal_supported": bool(tail_indices) and all(supported(index) for index in tail_indices),
        "support_gate": "exact_reference_evidence_substring_only",
    }


def _waypoint_page(
    waypoint: dict[str, Any], frozen: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any] | None:
    page = frozen.get((_fold(waypoint["title"]), int(waypoint["revision_id"])))
    if page is None:
        return None
    return {**page, "as_of": waypoint.get("as_of")}


def _direct_link_candidate(
    case: dict[str, Any], finding: dict[str, Any], backend: WikipediaPageBackend,
) -> dict[str, Any]:
    waypoints = case["temporal_waypoints"]
    frozen = _frozen_pages(case)
    source_index = next((
        index for index, waypoint in enumerate(waypoints)
        if int(waypoint["revision_id"]) == int(finding["revision_id"])
        and _fold(waypoint["title"]) == _fold(finding["page"])
    ), None)
    target_index = next((
        index for index, waypoint in enumerate(waypoints)
        if index > (source_index if source_index is not None else -1)
        and waypoint.get("role") == "relation_target"
        and int(waypoint.get("relation_hop", -1)) == int(finding["to_hop"])
    ), None)
    if source_index is None or target_index is None:
        return {"classification": "SEMANTIC_UNKNOWN", "reason": "route waypoint not found"}
    source = _waypoint_page(waypoints[source_index], frozen)
    target = _waypoint_page(waypoints[target_index], frozen)
    if source is None or target is None:
        return {"classification": "SEMANTIC_UNKNOWN", "reason": "frozen waypoint page missing"}
    try:
        landing_obj = backend.fetch_page(str(finding["target"]), as_of=str(source["as_of"]))
    except WikipediaError as exc:
        return {"classification": "SEMANTIC_UNKNOWN", "reason": f"landing revision unavailable: {exc}"}
    landing = {
        "title": landing_obj.title, "revision_id": landing_obj.revision_id,
        "timestamp": landing_obj.timestamp, "as_of": source["as_of"],
        "content": landing_obj.content,
    }
    if _fold(landing["title"]) != _fold(target["title"]):
        return {
            "classification": "REDIRECT_ALIAS_ARTIFACT",
            "reason": "link landing canonical title differs from proposed later entity",
            "landing": _node(landing, as_of=landing["as_of"]),
        }
    path_pages = [
        page for waypoint in waypoints[:source_index + 1]
        if (page := _waypoint_page(waypoint, frozen)) is not None
    ]
    actions = []
    for left, right in zip(path_pages, path_pages[1:]):
        kind = "SWITCH_SNAPSHOT" if _fold(left["title"]) == _fold(right["title"]) else "FOLLOW_LINK"
        actions.append(_action(kind, left, right))
    actions.append(_action("FOLLOW_LINK", source, landing))
    path_pages.append(landing)
    if int(landing["revision_id"]) != int(target["revision_id"]):
        actions.append(_action("SWITCH_SNAPSHOT", landing, target))
        path_pages.append(target)
    for waypoint in waypoints[target_index + 1:]:
        page = _waypoint_page(waypoint, frozen)
        if page is None:
            return {"classification": "SEMANTIC_UNKNOWN", "reason": "suffix frozen page missing"}
        kind = "SWITCH_SNAPSHOT" if _fold(path_pages[-1]["title"]) == _fold(page["title"]) else "FOLLOW_LINK"
        actions.append(_action(kind, path_pages[-1], page))
        path_pages.append(page)
    support = _path_support(case, path_pages)
    cutoff_support = _path_support(case, path_pages, cutoff_only=True)
    cutoff = str(case["knowledge_cutoff"]["cutoff_date"])[:10]
    postcutoff_switch = any(
        action["action"] == "SWITCH_SNAPSHOT"
        and str(action["to"].get("as_of") or "")[:10] > cutoff
        for action in actions
    )
    shorter = len(actions) < int(case["expected_navigation_distance"])
    repeated = len({(node["page"].casefold(), node["revision_id"]) for node in map(lambda p: _node(p, as_of=p.get("as_of")), path_pages)}) != len(path_pages)
    if repeated:
        classification = "REDIRECT_ALIAS_ARTIFACT"
    elif cutoff_support["all_bridges_literal_supported"]:
        classification = "PRE_CUTOFF_LEAKAGE"
    elif shorter and support["all_bridges_literal_supported"] and support["tail_literal_supported"] and postcutoff_switch:
        classification = "TEMPORAL_VALID_ALTERNATIVE"
    elif not shorter:
        classification = "NOT_A_SHORTER_EXECUTABLE_PATH"
    else:
        classification = "SEMANTIC_UNKNOWN"
    return {
        "classification": classification,
        "shortcut_path": [_node(page, as_of=page.get("as_of")) for page in path_pages],
        "actions": actions, "action_count": len(actions),
        "reference_action_count": int(case["expected_navigation_distance"]),
        "is_shorter": shorter, "post_cutoff_switch": postcutoff_switch,
        "evidence_support": support, "cutoff_evidence_support": cutoff_support,
        "structured_evaluator_status": (
            "literal_prerequisites_passed"
            if support["all_bridges_literal_supported"] and support["tail_literal_supported"]
            else "not_run_missing_literal_prerequisites"
        ),
    }


def audit_case(
    case: dict[str, Any], diagnostic: dict[str, Any], backend: WikipediaPageBackend,
) -> dict[str, Any]:
    findings = diagnostic.get("search_space_diagnostic", {}).get("findings", [])
    candidates = []
    for finding in findings:
        if finding.get("kind") == "start_revision_contains_final_answer":
            frozen = _frozen_pages(case)
            start_wp = case["temporal_waypoints"][0]
            start = _waypoint_page(start_wp, frozen)
            support = _path_support(case, [start] if start else [])
            cutoff_support = _path_support(case, [start] if start else [], cutoff_only=True)
            classification = (
                "PRE_CUTOFF_LEAKAGE" if cutoff_support["all_bridges_literal_supported"]
                else "ANSWER_ONLY_SHORTCUT"
            )
            candidates.append({
                "signal": finding, "classification": classification,
                "shortcut_path": [_node(start, as_of=start.get("as_of"))] if start else [],
                "actions": [{"action": "SUBMIT_ANSWER", "answer_aliases": finding.get("matched_aliases", [])}],
                "post_cutoff_switch": False, "evidence_support": support,
                "cutoff_evidence_support": cutoff_support,
                "structured_evaluator_status": "must_fail_without_bridge_evidence",
            })
        elif finding.get("kind") == "non_adjacent_route_link":
            candidates.append({"signal": finding, **_direct_link_candidate(case, finding, backend)})
    labels = {candidate["classification"] for candidate in candidates}
    if "PRE_CUTOFF_LEAKAGE" in labels:
        disposition = "reject"
    elif "REDIRECT_ALIAS_ARTIFACT" in labels:
        disposition = "reject_or_fix"
    elif "SEMANTIC_UNKNOWN" in labels:
        disposition = "quarantine"
    else:
        disposition = "retain"
    return {
        "schema_version": SCHEMA_VERSION, "case_id": case["id"],
        "audited_at": _now(), "original_shortcut_status": diagnostic.get("search_space_diagnostic", {}).get("shortcut_status"),
        "candidate_shortcuts": candidates,
        "classification_counts": {label: sum(c["classification"] == label for c in candidates) for label in sorted(labels)},
        "disposition": disposition,
        "human_evidence_review": "waived_not_performed",
        "inference_isolation": "posthoc_only_no_feedback",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-file", action="append", required=True)
    parser.add_argument("--ledger", action="append", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-manifest")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    case_paths = [Path(path) for path in args.case_file]
    ledger_paths = [Path(path) for path in args.ledger]
    frozen_cases, manifest = freeze_inputs(
        case_paths=case_paths, ledger_paths=ledger_paths, output_dir=output_dir,
        method_manifest=Path(args.method_manifest) if args.method_manifest else None,
    )
    cases = _load_cases([frozen_cases])
    diagnostics = _ledger_rows(ledger_paths)
    backend = WikipediaPageBackend(cache_path=args.cache_path, offline_only=True)
    try:
        audits = [audit_case(case, diagnostics[str(case["id"])], backend) for case in cases]
    finally:
        backend.close()
    audit_path = output_dir / "shortcut_semantic_audit.jsonl"
    audit_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audits), encoding="utf-8")
    dispositions = {label: sum(row["disposition"] == label for row in audits) for label in ("retain", "reject", "reject_or_fix", "quarantine")}
    labels = sorted({label for row in audits for label in row["classification_counts"]})
    summary = {
        "schema_version": SCHEMA_VERSION, "case_count": len(audits),
        "shortcut_signal_cases": sum(bool(row["candidate_shortcuts"]) for row in audits),
        "classification_counts": {label: sum(row["classification_counts"].get(label, 0) for row in audits) for label in labels},
        "dispositions": dispositions,
        "formal_abcd_authorized": dispositions["quarantine"] == 0 and dispositions["reject"] == 0 and dispositions["reject_or_fix"] == 0,
        "freeze_manifest": str(manifest), "audit_jsonl": str(audit_path),
    }
    _atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
