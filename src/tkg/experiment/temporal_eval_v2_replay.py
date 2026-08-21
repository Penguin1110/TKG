"""Append-only corrected replay of legacy development trajectories under v2."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_eval_schema_v2 import (
    EvaluationCaseV2, load_cases_v2, normalized,
)
from tkg.experiment.temporal_evaluation_v2 import (
    EVALUATION_SCHEMA_V2, evidence_id_v2, reference_route_diagnostics_v2,
)


REPLAY_SCHEMA_V2 = "open-world-temporal-development-replay-v2"


def _load_private_cases(paths: list[str]) -> dict[str, EvaluationCaseV2]:
    result: dict[str, EvaluationCaseV2] = {}
    for path in paths:
        for case in load_cases_v2(path):
            if case.case_id in result:
                raise ValueError(f"duplicate private case {case.case_id}")
            result[case.case_id] = case
    return result


def _read_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                row["_source_artifact"] = path
                row["_source_line"] = line_number
                rows.append(row)
    return rows


def _legacy_funnel(step: dict[str, Any]) -> dict[str, Any]:
    compaction = step.get("action_compaction")
    if not isinstance(compaction, dict):
        compaction = {}
    retrieved = list(compaction.get("pre_compaction_actions") or [])
    compacted = list(compaction.get("post_compaction_actions") or [])
    candidates = list(step.get("candidate_actions") or [])
    scores = {
        str(row["action_id"]): float(row["raw_ranker_score"])
        for row in candidates
        if row.get("action_id") is not None
        and isinstance(row.get("raw_ranker_score"), (int, float))
        and not isinstance(row.get("raw_ranker_score"), bool)
    }
    expected = {str(row.get("action_id")) for row in compacted}
    parent = step.get("parent_state")
    if not isinstance(parent, dict):
        parent = {}
    return {
        "schema_version": "legacy-funnel-derived-as-v2-diagnostic",
        "parent_page": parent.get("current_page"),
        "parent_revision_id": parent.get("current_revision_id"),
        "environment_legal_actions": [],
        "environment_legal_actions_status": (
            "not_reconstructable_from_legacy_trajectory"
        ),
        "solver_retrieved_actions": retrieved,
        "compacted_ranker_actions": compacted,
        "ranker_scores": scores if set(scores) == expected else {},
        "ranker_contract_valid": set(scores) == expected,
        "expanded_actions": [
            str(row["action_id"]) for row in candidates if row.get("expanded") is True
        ],
    }


def _witness_visibility(
    case: EvaluationCaseV2, evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    details = []
    for claim in case.critical_claims:
        matched = []
        for witness in claim.witnesses:
            excerpt = normalized(witness.evidence_excerpt)
            for page in evidence:
                if (
                    normalized(page.get("title")) == normalized(witness.page_title)
                    and page.get("revision_id") == witness.revision_id
                    and excerpt and excerpt in normalized(page.get("content"))
                ):
                    matched.append({
                        "page_title": witness.page_title,
                        "revision_id": witness.revision_id,
                        "evidence_id": str(page.get("evidence_id") or evidence_id_v2(page)),
                    })
        details.append({
            "claim_id": claim.claim_id,
            "any_witness_visible": bool(matched),
            "matched_witnesses": matched,
        })
    return {
        "critical_claim_results": details,
        "critical_bridge_count": len(details),
        "critical_bridges_acquired": sum(
            bool(row["any_witness_visible"]) for row in details
        ),
        "critical_bridge_acquisition_rate": (
            sum(bool(row["any_witness_visible"]) for row in details) / len(details)
            if details else 0.0
        ),
        "critical_bridge_evidence_complete": bool(details) and all(
            row["any_witness_visible"] for row in details
        ),
    }


def _summary_evidence(
    summary: dict[str, Any], evidence_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if summary.get("slot") == "external_agent_summary":
        result = summary.get("result")
        return list(result.get("evidence_pages") or []) if isinstance(result, dict) else []
    pages = []
    for row in evidence_rows:
        page = row.get("page")
        if isinstance(page, dict):
            page = dict(page)
            page.setdefault("evidence_id", row.get("evidence_id"))
            pages.append(page)
    return pages


def _action_trace(summary: dict[str, Any]) -> list[dict[str, Any]]:
    if summary.get("slot") == "external_agent_summary":
        result = summary.get("result")
        rows = list(result.get("trajectory") or []) if isinstance(result, dict) else []
        converted = []
        for row in rows:
            action = str(row.get("action") or "").upper()
            kind = {
                "FOLLOW_LINK": "FOLLOW_LINK",
                "SWITCH_SNAPSHOT": "SWITCH_SNAPSHOT",
            }.get(action)
            if kind:
                converted.append({"kind": kind, "params": dict(row.get("args") or {})})
        return converted
    final = summary.get("final_state")
    return list(final.get("action_trace") or []) if isinstance(final, dict) else []


def replay_development_artifacts_v2(
    *, case_paths: list[str], result_paths: list[str], output_path: str,
    report_path: str,
) -> list[dict[str, Any]]:
    assert_new_output_path(output_path)
    output = Path(output_path)
    report = Path(report_path)
    if output.exists() or report.exists():
        raise ValueError("v2 replay output paths must be new")
    cases = _load_private_cases(case_paths)
    rows = _read_rows(result_paths)
    evidence_by_run: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    steps_by_run: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    summaries = []
    for row in rows:
        key = (str(row.get("case_id")), str(row.get("arm")))
        if row.get("slot") == "beam_evidence":
            evidence_by_run[key].append(row)
        elif row.get("slot") == "beam_expansion":
            steps_by_run[key].append(row)
        elif row.get("slot") in {"beam_summary", "external_agent_summary"}:
            summaries.append(row)
    derived = []
    for summary in summaries:
        case_id = str(summary.get("case_id"))
        arm = str(summary.get("arm"))
        case = cases.get(case_id)
        if case is None:
            raise ValueError(f"missing private case for result {case_id}")
        evidence = _summary_evidence(summary, evidence_by_run[(case_id, arm)])
        visibility = _witness_visibility(case, evidence)
        submitted = str(
            summary.get("metrics", {}).get("submitted_answer")
            or summary.get("result", {}).get("final_answer")
            or ""
        )
        alias_match = normalized(submitted) in {
            normalized(value) for value in case.accepted_final_answer_aliases
        }
        evaluation = {
            "schema_version": EVALUATION_SCHEMA_V2,
            "case_id": case_id,
            **visibility,
            "final_answer_correct": alias_match,
            "literal_support_gate_passed": "not_evaluable_legacy_submission",
            "tail_claim_result": "not_evaluable_legacy_submission",
            "semantically_supported_submission": False,
            "temporally_valid_submission": False,
            "composition_valid": False,
            "end_to_end_validated_answer_accuracy": 0,
            "end_to_end_success": False,
            "evaluation_status": "legacy_unstructured_submission_not_formal",
        }
        funnels = [_legacy_funnel(step) for step in steps_by_run[(case_id, arm)]]
        stop_reason = str(
            summary.get("metrics", {}).get("search_stop_reason")
            or summary.get("result", {}).get("stop_reason")
            or ""
        )
        diagnostics = reference_route_diagnostics_v2(
            route=case.reference_routes[0] if case.reference_routes else None,
            funnel_steps=funnels,
            action_trace=_action_trace(summary),
            case=case,
            evaluation=evaluation,
            search_stop_reason=stop_reason,
        )
        derived.append({
            "schema_version": REPLAY_SCHEMA_V2,
            "derived_at": datetime.now(timezone.utc).isoformat(),
            "case_id": case_id,
            "arm": arm,
            "development_only": True,
            "source_artifact": summary["_source_artifact"],
            "source_line": summary["_source_line"],
            "legacy_raw_labels_preserved": True,
            "environment_solver_separation": (
                "legacy_environment_full_set_not_recorded"
            ),
            "structured_submission_available": False,
            "evaluation": evaluation,
            "reference_diagnostics": diagnostics,
        })
    with output.open("x", encoding="utf-8") as fh:
        for row in derived:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    label_counts = Counter(
        label for row in derived
        for label in row["reference_diagnostics"]["diagnostic_labels"]
    )
    lines = [
        "# Coach development replay under open-world evaluation v2",
        "",
        "This is an append-only derived audit of immutable legacy trajectories.",
        "The four cases remain development-only and no benchmark accuracy is reported.",
        "",
        f"- Derived run records: {len(derived)}",
        "- Structured v2 submissions present: 0",
        "- End-to-end validated chains: 0",
        "- Legacy full environment action sets: not recorded",
        "",
        "## Corrected diagnostic labels",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(label_counts.items()))
    lines.extend([
        "",
        "`REFERENCE_*` means only that the known feasibility route was not recovered.",
        "`NO_VALIDATED_EVIDENCE_CHAIN_FOUND` describes these saved trajectories; it",
        "does not claim that no other route exists in Wikipedia.",
        "",
        "The old `LEGAL_CANDIDATE_RECALL_FAILURE` and related fields remain untouched",
        "inside their original JSONL and are not reused as v2 primary outcomes.",
    ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return derived


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay legacy trajectories under v2")
    parser.add_argument("--cases", action="append", required=True)
    parser.add_argument("--results", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    rows = replay_development_artifacts_v2(
        case_paths=args.cases, result_paths=args.results,
        output_path=args.output, report_path=args.report,
    )
    print(f"[done] wrote {len(rows)} derived v2 records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
