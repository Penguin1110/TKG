"""Frozen independent gate for the hierarchical open-weight controller v2.4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tkg.experiment.compact_joint_controller_v24 import CompactSubmitSlotActionV24
from tkg.experiment.open_weight_action_scorer_v24 import (
    HuggingFaceCausalLMBackendV24, OpenWeightConditionalActionScorerV24,
)
from tkg.experiment.open_weight_answer_generator_v24 import (
    EvidenceConditionedAnswerGeneratorV24,
)
from tkg.experiment.open_weight_live_controller_v24 import (
    HierarchicalOpenWeightLiveControllerV24, MODE_VERBALIZER_ROTATIONS_V24,
)
from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_environment_v2 import EnvironmentActionV2
from tkg.experiment.temporal_eval_schema_v2 import PublicTemporalCaseV2


GATE_SCHEMA_V24 = "hierarchical-open-weight-independent-gate-v2.4"
FREEZE_SCHEMA_V24 = "hierarchical-open-weight-freeze-v2.4"
FROZEN_BEAM_POLICY_V24 = {
    "beam_widths_authorized_if_passed": [1, 3, 5],
    "max_actions_per_state": 3,
    "dense_action_limit": 30,
    "tie_break": "descending_score_then_action_id",
    "score": "length_normalized_conditional_logprob",
    "factorization": "P(mode|state)*P(graph_action|continue,state)",
}
PREREGISTERED_THRESHOLDS_V24 = {
    "navigation_progress_recall_at_3_min": 0.8,
    "complete_submit_timing_rate_min": 0.8,
    "incomplete_false_submit_timing_rate_max": 0.2,
    "complete_answer_generation_rate_min": 0.8,
    "incomplete_false_answer_generation_rate_max": 0.2,
    "exact_action_contract_rate": 1.0,
}
FROZEN_SOURCE_FILES_V24 = (
    "src/tkg/experiment/open_weight_action_scorer_v24.py",
    "src/tkg/experiment/open_weight_answer_generator_v24.py",
    "src/tkg/experiment/open_weight_live_controller_v24.py",
    "src/tkg/experiment/temporal_live_runner_v24.py",
    "src/tkg/experiment/open_weight_independent_gate_v24.py",
)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _file_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state(evidence: list[dict[str, Any]], *, page: str) -> Any:
    return SimpleNamespace(
        current_page=page, current_revision_id=9001,
        reasoning_summary="Independent frozen gate state.",
        collected_evidence=tuple(evidence),
    )


def _case(case_id: str, question: str) -> PublicTemporalCaseV2:
    return PublicTemporalCaseV2(
        case_id=case_id, model_id="independent-gate",
        question=question, start_page="Gate Index", cutoff_date="2024-06-01",
        target_date="2026-01-01",
    )


def gate_dataset_v24() -> dict[str, Any]:
    navigation = []
    for index in range(10):
        person = f"Navora Kel-{index}"
        venue = f"Orchid Theatre {index}"
        question = f"Where was the person who became artistic director of {venue} born?"
        evidence = [{
            "evidence_id": f"nav_bridge_{index}", "title": venue,
            "revision_id": 10_000 + index, "timestamp": "2025-08-14T00:00:00Z",
            "content": f"On 14 August 2025, {person} became artistic director of {venue}.",
        }]
        progress_titles = [person, f"Biography of {person}"]
        titles = progress_titles + [f"Catalogue {index}-{offset}" for offset in range(9)]
        rotation = index % len(titles)
        titles = titles[rotation:] + titles[:rotation]
        actions = [{
            "kind": "FOLLOW_LINK", "page_title": title,
            "label": f"Follow hyperlink to {title}",
        } for title in titles]
        actions.append({"kind": "LIST_REVISIONS", "label": "List revisions"})
        navigation.append({
            "state_id": f"independent-navigation-{index}", "question": question,
            "evidence": evidence, "actions": actions,
            "progress_titles": progress_titles,
        })
    complete = []
    incomplete = []
    relations = (
        ("born", "was born in", "Cobalt Haven"),
        ("educated", "studied at", "Juniper College"),
        ("employed", "works for", "Marble Observatory"),
        ("married", "is married to", "Tarin Vale"),
        ("born", "was born in", "Saffron Point"),
    )
    for index, (kind, tail_phrase, answer) in enumerate(relations):
        person = f"Evidara Quill-{index}"
        institution = f"Silver Archive {index}"
        if kind == "born":
            question = f"Where was the person appointed director of {institution} born?"
        elif kind == "educated":
            question = f"Where did the person appointed director of {institution} study?"
        elif kind == "employed":
            question = f"Which organization employs the person appointed director of {institution}?"
        else:
            question = f"Who is the person appointed director of {institution} married to?"
        bridge = {
            "evidence_id": f"complete_bridge_{index}", "title": institution,
            "revision_id": 20_000 + index, "timestamp": "2025-09-03T00:00:00Z",
            "content": f"On 3 September 2025, {person} was appointed director of {institution}.",
        }
        tail = {
            "evidence_id": f"complete_tail_{index}", "title": person,
            "revision_id": 21_000 + index, "timestamp": "2025-09-04T00:00:00Z",
            "content": f"{person} {tail_phrase} {answer}.",
        }
        complete.append({
            "state_id": f"independent-complete-{index}", "question": question,
            "evidence": [bridge, tail], "expected_answer": answer,
        })
        incomplete.append({
            "state_id": f"independent-incomplete-{index}", "question": question,
            "evidence": [bridge], "expected_answer": None,
        })
    return {
        "dataset_id": "independent-gate-states-2026-08-16-v1",
        "prompt_development_overlap": False,
        "navigation": navigation, "complete": complete, "incomplete": incomplete,
    }


def create_freeze_manifest_v24(output: str) -> dict[str, Any]:
    assert_new_output_path(output)
    dataset = gate_dataset_v24()
    manifest = {
        "schema_version": FREEZE_SCHEMA_V24,
        "status": "frozen_before_model_run",
        "source_sha256": {path: _file_hash(path) for path in FROZEN_SOURCE_FILES_V24},
        "gate_dataset_sha256": _canonical_hash(dataset),
        "gate_dataset_counts": {
            key: len(dataset[key]) for key in ("navigation", "complete", "incomplete")
        },
        "verbalizer_rotations": MODE_VERBALIZER_ROTATIONS_V24,
        "beam_policy": FROZEN_BEAM_POLICY_V24,
        "thresholds": PREREGISTERED_THRESHOLDS_V24,
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    Path(output).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def _verify_freeze(path: str) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded_hash = manifest.pop("manifest_sha256", None)
    if recorded_hash != _canonical_hash(manifest):
        raise ValueError("freeze manifest hash mismatch")
    manifest["manifest_sha256"] = recorded_hash
    if manifest.get("gate_dataset_sha256") != _canonical_hash(gate_dataset_v24()):
        raise ValueError("independent gate dataset changed after freeze")
    for path_name, expected in manifest.get("source_sha256", {}).items():
        if _file_hash(path_name) != expected:
            raise ValueError(f"frozen source changed: {path_name}")
    return manifest


def _actions(rows: list[dict[str, Any]]) -> list[Any]:
    actions: list[Any] = []
    for order, row in enumerate(rows):
        params = ({"page_title": row["page_title"]}
                  if row["kind"] == "FOLLOW_LINK" else {"cursor": None})
        actions.append(EnvironmentActionV2(
            row["kind"], params, row["label"], environment_order=order,
        ))
    actions.append(CompactSubmitSlotActionV24())
    return actions


def run_independent_gate_v24(
    *, freeze_manifest: str, model: str, device: str, dtype: str,
) -> dict[str, Any]:
    frozen = _verify_freeze(freeze_manifest)
    backend = HuggingFaceCausalLMBackendV24(model, device=device, dtype=dtype)
    controller = HierarchicalOpenWeightLiveControllerV24(
        OpenWeightConditionalActionScorerV24(backend),
        compact_payload_proposer=EvidenceConditionedAnswerGeneratorV24(backend),
        payload_proposer_name="open_weight_evidence_conditioned_answer_generator_v2.4",
    )
    dataset = gate_dataset_v24()
    nav_rows = []
    contract_values: list[bool] = []
    for row in dataset["navigation"]:
        case = _case(row["state_id"], row["question"])
        state = _state(row["evidence"], page="Navigation Gate")
        actions = _actions(row["actions"])
        output = controller.control(
            case, state, actions, seed=47, budget={"max_expansions": 40},
        )
        ranked = sorted(output.scores, key=lambda key: (-output.scores[key], key))
        progress_ids = {
            action.action_id for action in actions
            if action.to_dict().get("params", {}).get("page_title")
            in set(row["progress_titles"])
        }
        rank = min(ranked.index(action_id) + 1 for action_id in progress_ids)
        valid = set(output.scores) == {action.action_id for action in actions}
        contract_values.append(valid)
        nav_rows.append({
            "state_id": row["state_id"], "best_progress_rank": rank,
            "progress_in_top_3": rank <= 3, "ranking_contract_valid": valid,
            "ranked_action_ids": ranked, "scores": output.scores,
        })
    submit_rows = []
    for completeness in ("complete", "incomplete"):
        for row in dataset[completeness]:
            case = _case(row["state_id"], row["question"])
            state = _state(row["evidence"], page="Submission Gate")
            actions = _actions([
                {"kind": "LIST_REVISIONS", "label": "List revisions"},
                {"kind": "LIST_LINKS", "label": "List hyperlinks"},
            ])
            output = controller.control(
                case, state, actions, seed=47, budget={"max_expansions": 40},
            )
            attempt = output.attempts[0]
            modes = attempt["mode_log_probabilities"]
            submit_selected = modes["SUBMIT"] > modes["CONTINUE"]
            generated = output.submission
            answer_correct = (
                generated is not None
                and generated.answer.casefold() == str(row["expected_answer"]).casefold()
            ) if completeness == "complete" else generated is None
            valid = set(output.scores) == {action.action_id for action in actions}
            contract_values.append(valid)
            submit_rows.append({
                "state_id": row["state_id"], "complete": completeness == "complete",
                "mode_log_probabilities": modes, "submit_selected": submit_selected,
                "generated_submission": generated.to_dict() if generated else None,
                "generation_status": attempt["payload_proposal"]["status"],
                "answer_gate_passed": answer_correct,
                "ranking_contract_valid": valid,
            })
    mean = lambda values: sum(values) / len(values)  # noqa: E731
    complete = [row for row in submit_rows if row["complete"]]
    incomplete = [row for row in submit_rows if not row["complete"]]
    metrics = {
        "navigation_progress_recall_at_3": mean([
            row["progress_in_top_3"] for row in nav_rows
        ]),
        "complete_submit_timing_rate": mean([
            row["submit_selected"] for row in complete
        ]),
        "incomplete_false_submit_timing_rate": mean([
            row["submit_selected"] for row in incomplete
        ]),
        "complete_answer_generation_rate": mean([
            row["answer_gate_passed"] for row in complete
        ]),
        "incomplete_false_answer_generation_rate": mean([
            not row["answer_gate_passed"] for row in incomplete
        ]),
        "exact_action_contract_rate": mean(contract_values),
    }
    t = PREREGISTERED_THRESHOLDS_V24
    checks = {
        "navigation": metrics["navigation_progress_recall_at_3"] >= t["navigation_progress_recall_at_3_min"],
        "positive_submit_timing": metrics["complete_submit_timing_rate"] >= t["complete_submit_timing_rate_min"],
        "negative_submit_timing": metrics["incomplete_false_submit_timing_rate"] <= t["incomplete_false_submit_timing_rate_max"],
        "positive_answer_generation": metrics["complete_answer_generation_rate"] >= t["complete_answer_generation_rate_min"],
        "negative_answer_generation": metrics["incomplete_false_answer_generation_rate"] <= t["incomplete_false_answer_generation_rate_max"],
        "contract": metrics["exact_action_contract_rate"] == t["exact_action_contract_rate"],
    }
    passed = all(checks.values())
    return {
        "schema_version": GATE_SCHEMA_V24, "model": model,
        "freeze_manifest_sha256": frozen["manifest_sha256"],
        "metrics": metrics, "checks": checks, "passed": passed,
        "method_abcd_authorized": passed,
        "navigation_rows": nav_rows, "submission_rows": submit_rows,
        "formal_benchmark_result": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--freeze-manifest", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--dtype", default="float16")
    run.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = create_freeze_manifest_v24(args.output)
        print(json.dumps({"manifest_sha256": result["manifest_sha256"]}))
        return 0
    assert_new_output_path(args.output)
    result = run_independent_gate_v24(
        freeze_manifest=args.freeze_manifest, model=args.model,
        device=args.device, dtype=args.dtype,
    )
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"passed": result["passed"], "metrics": result["metrics"]}))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
