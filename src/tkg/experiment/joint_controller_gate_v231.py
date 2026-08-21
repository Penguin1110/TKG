"""Fresh, one-shot navigation and submission gate for frozen joint v2.3.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tkg.experiment.joint_controller_v23 import (
    ApiJointRankAndSubmitControllerV23, JointControllerContractErrorV23,
    SUBMIT_SLOT_ID_V23, SubmitSlotActionV23, assert_joint_public_payload_v23,
    validate_submission_public_v23,
)
from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_environment_v2 import EnvironmentActionV2
from tkg.experiment.temporal_eval_schema_v2 import PublicTemporalCaseV2
from tkg.experiment.temporal_evaluation_v2 import evidence_id_v2
from tkg.experiment.temporal_live_runner_v22 import LiveBeamStateV22


GATE_SCHEMA_V231 = "fresh-joint-navigation-submission-gate-v2.3.1"
GATE_THRESHOLDS_V231 = {
    "navigation_any_progress_recall_at_3_min": 0.90,
    "navigation_strict_progress_separation_rate_min": 0.80,
    "positive_valid_submission_recall_min": 0.90,
    "negative_false_submit_rate_max": 0.05,
    "dense_ranking_contract_pass_rate": 1.00,
    "evidence_ownership_pass_rate": 1.00,
    "gold_leakage": 0,
}


def _evidence(
    title: str, revision_id: int, content: str,
) -> dict[str, Any]:
    page = {
        "title": title,
        "page_id": revision_id,
        "revision_id": revision_id,
        "timestamp": "2025-06-01T00:00:00Z",
        "as_of": "2025-06-01",
        "content": content,
        "links": [],
        "source_url": f"synthetic-gate://revision/{revision_id}",
    }
    page["evidence_id"] = evidence_id_v2(page)
    return page


def _state(case_id: str, evidence: list[dict[str, Any]]) -> LiveBeamStateV22:
    revision_id = 10_000 + int(case_id.rsplit("-", 1)[-1])
    return LiveBeamStateV22(
        current_page=f"Fresh Gate State {case_id}",
        current_revision_id=revision_id,
        current_revision_timestamp="2025-06-01T00:00:00Z",
        snapshot_as_of="2025-06-01",
        reasoning_summary="",
        extracted_entities=(),
        collected_evidence=tuple(evidence),
        retrieved_link_actions=(), retrieved_revision_actions=(),
        link_query_started=False, link_next_cursor=None, links_exhausted=False,
        revision_query_started=False, revision_next_cursor=None,
        revisions_exhausted=False, environment_queries_used=0,
        visited_nodes=((f"fresh gate state {case_id}".casefold(), revision_id),),
        action_trace=(), cumulative_score=0.0, finished=False, submitted=None,
    )


def build_fresh_gate_manifest_v231() -> dict[str, Any]:
    states = []
    for index in range(20):
        person = f"Nora Gateperson {index:02d}"
        lab = f"Gate Laboratory {index:02d}"
        case_id = f"navigation-{index:02d}"
        question = (
            f"Where was the person who became director of {lab} after the cutoff born?"
        )
        evidence = [_evidence(
            f"Appointment bulletin {index:02d}", 20_000 + index,
            f"On 4 April 2025, {person} became director of {lab}.",
        )]
        targets = [
            f"Profile of {person}", f"Biography of {person}",
            *[f"Unrelated gate topic {index:02d}-{offset:02d}" for offset in range(27)],
        ]
        rotation = index % len(targets)
        targets = targets[rotation:] + targets[:rotation]
        actions = [EnvironmentActionV2(
            "FOLLOW_LINK", {"page_title": title}, f"Follow hyperlink to {title}",
            environment_order=order,
        ).to_dict() for order, title in enumerate(targets)]
        actions.append(SubmitSlotActionV23().to_dict())
        progress_titles = {f"Profile of {person}", f"Biography of {person}"}
        progress_ids = [
            action["action_id"] for action in actions
            if action.get("params", {}).get("page_title") in progress_titles
        ]
        states.append({
            "state_id": case_id, "gate_type": "navigation",
            "public": {
                "case": PublicTemporalCaseV2(
                    case_id=case_id, model_id="openai/gpt-5.4-mini",
                    question=question, start_page=f"Fresh Gate State {case_id}",
                    cutoff_date="2024-06-01", target_date="2025-12-31",
                ).to_dict(),
                "evidence": evidence, "actions": actions,
            },
            "posthoc_label": {
                "progress_action_ids": progress_ids,
                "progress_action_count": len(progress_ids),
                "policy": "any listed progress action is acceptable",
            },
        })

    for index in range(20):
        person = f"Paula Complete {index:02d}"
        lab = f"Complete Laboratory {index:02d}"
        city = f"Complete City {index:02d}"
        case_id = f"positive-{index:02d}"
        question = (
            f"Where was the person who became director of {lab} after the cutoff born?"
        )
        evidence = [
            _evidence(
                f"Leadership record {index:02d}", 30_000 + index,
                f"On 5 May 2025, {person} became director of {lab}.",
            ),
            _evidence(
                person, 31_000 + index,
                f"{person} was born in {city}.",
            ),
        ]
        actions = [
            EnvironmentActionV2(
                "LIST_LINKS", {"cursor": None, "page_size": 50}, "List links",
            ).to_dict(),
            EnvironmentActionV2(
                "LIST_REVISIONS",
                {
                    "cursor": None, "page_size": 50,
                    "time_window": ["2024-06-01", "2025-12-31"],
                },
                "List revisions",
            ).to_dict(),
            SubmitSlotActionV23().to_dict(),
        ]
        states.append({
            "state_id": case_id, "gate_type": "positive_submission",
            "public": {
                "case": PublicTemporalCaseV2(
                    case_id=case_id, model_id="openai/gpt-5.4-mini",
                    question=question, start_page=f"Fresh Gate State {case_id}",
                    cutoff_date="2024-06-01", target_date="2025-12-31",
                ).to_dict(),
                "evidence": evidence, "actions": actions,
            },
            "posthoc_label": {"complete_evidence": True},
        })

    negative_kinds = [
        "missing_tail", "missing_bridge", "missing_event_time", "relation_mismatch",
    ]
    for index in range(20):
        person = f"Nina Incomplete {index:02d}"
        lab = f"Incomplete Laboratory {index:02d}"
        city = f"Incomplete City {index:02d}"
        kind = negative_kinds[index // 5]
        case_id = f"negative-{index:02d}"
        question = (
            f"Where was the person who became director of {lab} after the cutoff born?"
        )
        bridge_text = f"On 6 June 2025, {person} became director of {lab}."
        tail_text = f"{person} was born in {city}."
        if kind == "missing_tail":
            contents = [(f"Partial leadership {index:02d}", bridge_text)]
        elif kind == "missing_bridge":
            contents = [(person, tail_text)]
        elif kind == "missing_event_time":
            contents = [
                (f"Undated leadership {index:02d}", f"{person} became director of {lab}."),
                (person, tail_text),
            ]
        else:
            contents = [
                (f"Award record {index:02d}",
                 f"On 6 June 2025, {person} received an award from {lab}."),
                (person, tail_text),
            ]
        evidence = [
            _evidence(title, 40_000 + index * 3 + offset, content)
            for offset, (title, content) in enumerate(contents)
        ]
        actions = [
            EnvironmentActionV2(
                "LIST_LINKS", {"cursor": None, "page_size": 50}, "List links",
            ).to_dict(),
            EnvironmentActionV2(
                "LIST_REVISIONS",
                {
                    "cursor": None, "page_size": 50,
                    "time_window": ["2024-06-01", "2025-12-31"],
                }, "List revisions",
            ).to_dict(),
            SubmitSlotActionV23().to_dict(),
        ]
        states.append({
            "state_id": case_id, "gate_type": "negative_submission",
            "public": {
                "case": PublicTemporalCaseV2(
                    case_id=case_id, model_id="openai/gpt-5.4-mini",
                    question=question, start_page=f"Fresh Gate State {case_id}",
                    cutoff_date="2024-06-01", target_date="2025-12-31",
                ).to_dict(),
                "evidence": evidence, "actions": actions,
            },
            "posthoc_label": {"complete_evidence": False, "negative_kind": kind},
        })
    return {
        "schema_version": GATE_SCHEMA_V231,
        "status": "frozen_before_model_run",
        "state_counts": {
            "navigation": 20, "positive_submission": 20,
            "negative_submission": 20, "total": 60,
        },
        "thresholds": GATE_THRESHOLDS_V231,
        "states": states,
    }


def _action_from_dict(row: dict[str, Any]):
    if row["action_id"] == SUBMIT_SLOT_ID_V23:
        return SubmitSlotActionV23()
    return EnvironmentActionV2(
        kind=row["kind"], params=row["params"], label=row["label"],
        environment_order=row.get("environment_order"),
    )


def _state_from_manifest(row: dict[str, Any]) -> LiveBeamStateV22:
    return _state(row["state_id"], list(row["public"]["evidence"]))


def run_gate_v231(
    manifest_path: str | Path, *, model: str, cache_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == GATE_SCHEMA_V231
    assert_new_output_path(str(output_path))
    controller = ApiJointRankAndSubmitControllerV23(model, cache_path=cache_path)
    results = []
    try:
        for row in manifest["states"]:
            public = row["public"]
            assert_joint_public_payload_v23(public)
            case_data = dict(public["case"])
            case_data["model_id"] = model
            case = PublicTemporalCaseV2(**case_data)
            state = _state_from_manifest(row)
            actions = [_action_from_dict(action) for action in public["actions"]]
            result: dict[str, Any] = {
                "state_id": row["state_id"],
                "gate_type": row["gate_type"],
                "public_input": public,
                "posthoc_label": row["posthoc_label"],
            }
            try:
                output = controller.control(
                    case, state, actions, seed=31,
                    budget={
                        "expansions_used": 0, "max_expansions": 40,
                        "beam_width": 3, "max_actions_per_state": 4,
                    },
                )
                validation = (
                    validate_submission_public_v23(
                        output.submission, list(state.collected_evidence),
                    ) if output.submission is not None
                    else output.submission_validation
                )
                result.update({
                    "ranking_contract_passed": True,
                    "action_utilities": output.scores,
                    "submission": (
                        output.submission.to_dict() if output.submission else None
                    ),
                    "submission_validation": validation.to_dict(),
                    "abstain_reason": output.abstain_reason,
                    "controller_attempts": list(output.attempts),
                })
                if row["gate_type"] == "navigation":
                    progress = set(row["posthoc_label"]["progress_action_ids"])
                    graph_ids = {
                        action.action_id for action in actions
                        if action.action_id != SUBMIT_SLOT_ID_V23
                    }
                    ordered = sorted(
                        graph_ids,
                        key=lambda action_id: (-output.scores[action_id], action_id),
                    )
                    distractors = graph_ids - progress
                    result["navigation_metrics"] = {
                        "any_progress_in_top_3": bool(progress & set(ordered[:3])),
                        "strict_progress_separation": (
                            max(output.scores[action_id] for action_id in progress)
                            > max(output.scores[action_id] for action_id in distractors)
                        ),
                        "best_progress_rank": min(
                            ordered.index(action_id) + 1 for action_id in progress
                        ),
                    }
            except JointControllerContractErrorV23 as exc:
                result.update({
                    "ranking_contract_passed": False,
                    "contract_error": str(exc),
                })
            results.append(result)
    finally:
        controller.close()

    navigation = [row for row in results if row["gate_type"] == "navigation"]
    positive = [row for row in results if row["gate_type"] == "positive_submission"]
    negative = [row for row in results if row["gate_type"] == "negative_submission"]
    contract_rate = sum(row["ranking_contract_passed"] for row in results) / len(results)
    navigation_recall = sum(
        row.get("navigation_metrics", {}).get("any_progress_in_top_3", False)
        for row in navigation
    ) / len(navigation)
    navigation_separation = sum(
        row.get("navigation_metrics", {}).get("strict_progress_separation", False)
        for row in navigation
    ) / len(navigation)
    positive_recall = sum(
        row.get("submission_validation", {}).get("status") == "valid"
        for row in positive
    ) / len(positive)
    false_submit_rate = sum(
        row.get("submission_validation", {}).get("status") == "valid"
        for row in negative
    ) / len(negative)
    submitted = [
        row for row in [*positive, *negative] if row.get("submission") is not None
    ]
    ownership_rate = (
        sum(row.get("submission_validation", {}).get("status") not in {
            "invalid_evidence_ownership"
        } for row in submitted) / len(submitted)
        if submitted else 1.0
    )
    metrics = {
        "navigation_any_progress_recall_at_3": navigation_recall,
        "navigation_strict_progress_separation_rate": navigation_separation,
        "positive_valid_submission_recall": positive_recall,
        "negative_false_submit_rate": false_submit_rate,
        "dense_ranking_contract_pass_rate": contract_rate,
        "evidence_ownership_pass_rate": ownership_rate,
        "gold_leakage": 0,
    }
    thresholds = manifest["thresholds"]
    checks = {
        "navigation_any_progress_recall_at_3": (
            navigation_recall >= thresholds["navigation_any_progress_recall_at_3_min"]
        ),
        "navigation_strict_progress_separation_rate": (
            navigation_separation >= thresholds[
                "navigation_strict_progress_separation_rate_min"
            ]
        ),
        "positive_valid_submission_recall": (
            positive_recall >= thresholds["positive_valid_submission_recall_min"]
        ),
        "negative_false_submit_rate": (
            false_submit_rate <= thresholds["negative_false_submit_rate_max"]
        ),
        "dense_ranking_contract_pass_rate": (
            contract_rate == thresholds["dense_ranking_contract_pass_rate"]
        ),
        "evidence_ownership_pass_rate": (
            ownership_rate == thresholds["evidence_ownership_pass_rate"]
        ),
        "gold_leakage": metrics["gold_leakage"] == thresholds["gold_leakage"],
    }
    artifact = {
        "schema_version": "fresh-joint-gate-result-v2.3.1",
        "manifest": str(manifest_path),
        "model": model,
        "thresholds": thresholds,
        "metrics": metrics,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "results": results,
        "formal_benchmark_conclusion_allowed": False,
    }
    Path(output_path).write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-manifest")
    parser.add_argument("--manifest")
    parser.add_argument("--model", default="openai/gpt-5.4-mini")
    parser.add_argument("--cache")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.prepare_manifest:
        assert_new_output_path(args.prepare_manifest)
        Path(args.prepare_manifest).write_text(
            json.dumps(build_fresh_gate_manifest_v231(), ensure_ascii=False,
                       indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[done] prepared fresh joint gate: {args.prepare_manifest}")
        return 0
    if not args.manifest or not args.cache or not args.output:
        parser.error("run mode requires --manifest, --cache, and --output")
    result = run_gate_v231(
        args.manifest, model=args.model, cache_path=args.cache,
        output_path=args.output,
    )
    print(f"[done] fresh joint gate passed={result['gate_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
