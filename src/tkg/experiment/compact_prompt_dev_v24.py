"""Independent, synthetic prompt-development set for compact controller v2.4.

This set is intentionally separate from the frozen v2.3.1 gate. Results are
development diagnostics only and cannot unlock A/B/C/D.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tkg.experiment.compact_joint_controller_v24 import (
    ApiCompactJointControllerV24, CompactSubmitSlotActionV24,
)
from tkg.experiment.compact_submission_v24 import validate_compact_submission_public_v24
from tkg.experiment.joint_controller_gate_v231 import _evidence, _state
from tkg.experiment.joint_controller_v23 import SUBMIT_SLOT_ID_V23
from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_environment_v2 import EnvironmentActionV2
from tkg.experiment.temporal_eval_schema_v2 import PublicTemporalCaseV2


PROMPT_DEV_SCHEMA_V24 = "compact-controller-prompt-development-v2.4"


def build_compact_prompt_dev_manifest_v24(
    model_id: str = "openai/gpt-5.4-mini",
) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    for index in range(4):
        person = f"Iris Wayfinder {index}"
        institute = f"Wayfinder Institute {index}"
        case_id = f"v24-dev-navigation-{index}"
        evidence = [_evidence(
            f"Wayfinder bulletin {index}", 61_000 + index,
            f"On 14 March 2025, {person} became director of {institute}.",
        )]
        targets = [
            f"Archive topic {index}-{offset}" for offset in range(6)
        ] + [f"Biography of {person}", f"Profile of {person}"]
        targets = targets[index:] + targets[:index]
        actions = [EnvironmentActionV2(
            "FOLLOW_LINK", {"page_title": title}, f"Follow hyperlink to {title}",
            environment_order=order,
        ).to_dict() for order, title in enumerate(targets)]
        actions.append(CompactSubmitSlotActionV24().to_dict())
        progress = [
            action["action_id"] for action in actions
            if action.get("params", {}).get("page_title") in {
                f"Biography of {person}", f"Profile of {person}",
            }
        ]
        states.append(_row(
            case_id, model_id,
            f"Where was the person who became director of {institute} after the cutoff born?",
            evidence, actions, "navigation", {"progress_action_ids": progress},
        ))

    for index in range(4):
        person = f"Mara Complete {index}"
        institute = f"Complete Observatory {index}"
        city = f"Copper City {index}"
        case_id = f"v24-dev-positive-{index}"
        evidence = [
            _evidence(
                f"Observatory bulletin {index}", 62_000 + index,
                f"On 8 April 2025, {person} became director of {institute}.",
            ),
            _evidence(person, 63_000 + index, f"{person} was born in {city}."),
        ]
        states.append(_row(
            case_id, model_id,
            f"Where was the person who became director of {institute} after the cutoff born?",
            evidence, _control_actions(), "positive_submission",
            {"expected_answer": city},
        ))

    for index in range(4):
        person = f"Nadia Partial {index}"
        institute = f"Partial Museum {index}"
        city = f"Silver City {index}"
        case_id = f"v24-dev-negative-{index}"
        if index % 2 == 0:
            evidence = [_evidence(
                f"Museum bulletin {index}", 64_000 + index,
                f"On 9 May 2025, {person} became director of {institute}.",
            )]
            missing = "tail"
        else:
            evidence = [_evidence(person, 65_000 + index, f"{person} was born in {city}.")]
            missing = "bridge"
        states.append(_row(
            case_id, model_id,
            f"Where was the person who became director of {institute} after the cutoff born?",
            evidence, _control_actions(), "negative_submission", {"missing": missing},
        ))
    return {
        "schema_version": PROMPT_DEV_SCHEMA_V24,
        "status": "development_only_not_a_gate",
        "independent_from": "fresh-joint-navigation-submission-gate-v2.3.1",
        "may_unlock_abcd": False,
        "state_counts": {"navigation": 4, "positive_submission": 4,
                         "negative_submission": 4, "total": 12},
        "states": states,
    }


def _control_actions() -> list[dict[str, Any]]:
    return [
        EnvironmentActionV2(
            "LIST_LINKS", {"cursor": None, "page_size": 50}, "List links",
        ).to_dict(),
        CompactSubmitSlotActionV24().to_dict(),
    ]


def _row(
    case_id: str, model_id: str, question: str, evidence: list[dict[str, Any]],
    actions: list[dict[str, Any]], kind: str, label: dict[str, Any],
) -> dict[str, Any]:
    return {
        "state_id": case_id, "development_type": kind,
        "public": {
            "case": PublicTemporalCaseV2(
                case_id=case_id, model_id=model_id, question=question,
                start_page=f"Prompt Development {case_id}",
                cutoff_date="2024-06-01", target_date="2025-12-31",
            ).to_dict(),
            "evidence": evidence, "actions": actions,
        },
        "posthoc_development_label": label,
    }


def _action_from_dict(row: dict[str, Any]):
    if row["action_id"] == SUBMIT_SLOT_ID_V23:
        return CompactSubmitSlotActionV24()
    return EnvironmentActionV2(
        kind=row["kind"], params=row["params"], label=row["label"],
        environment_order=row.get("environment_order"),
    )


def run_compact_prompt_dev_v24(
    manifest_path: str | Path, *, model: str, cache_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PROMPT_DEV_SCHEMA_V24:
        raise ValueError("prompt-development manifest schema mismatch")
    assert_new_output_path(str(output_path))
    controller = ApiCompactJointControllerV24(model, cache_path=cache_path)
    rows = []
    try:
        for source in manifest["states"]:
            public = source["public"]
            case_data = dict(public["case"])
            case_data["model_id"] = model
            case = PublicTemporalCaseV2(**case_data)
            state = _state(source["state_id"], list(public["evidence"]))
            actions = [_action_from_dict(action) for action in public["actions"]]
            output = controller.control(
                case, state, actions, seed=43,
                budget={"expansions_used": 0, "max_expansions": 40,
                        "beam_width": 3, "max_actions_per_state": 4},
            )
            validation = (
                validate_compact_submission_public_v24(
                    output.submission, list(state.collected_evidence),
                ) if output.submission is not None
                else output.submission_schema_validation
            )
            ordered_graph = sorted(
                (action.action_id for action in actions
                 if action.action_id != SUBMIT_SLOT_ID_V23),
                key=lambda action_id: (-output.scores[action_id], action_id),
            )
            progress = set(source["posthoc_development_label"].get(
                "progress_action_ids", []
            ))
            rows.append({
                "state_id": source["state_id"],
                "development_type": source["development_type"],
                "ranking_contract_passed": True,
                "scores": output.scores,
                "progress_in_top_3": bool(progress & set(ordered_graph[:3]))
                if progress else None,
                "submission": output.submission.to_dict()
                if output.submission else None,
                "submission_validation": validation.to_dict(),
                "attempts": list(output.attempts),
            })
    finally:
        controller.close()
    result = {
        "schema_version": PROMPT_DEV_SCHEMA_V24,
        "status": "development_only_not_a_gate",
        "may_unlock_abcd": False,
        "model": model,
        "rows": rows,
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--model", default="openai/gpt-5.4-mini")
    parser.add_argument("--cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--build-manifest", action="store_true")
    args = parser.parse_args()
    if args.build_manifest:
        assert_new_output_path(args.output)
        Path(args.output).write_text(json.dumps(
            build_compact_prompt_dev_manifest_v24(args.model),
            ensure_ascii=False, indent=2,
        ) + "\n", encoding="utf-8")
        print(json.dumps({"output": args.output, "states": 12}))
        return 0
    if not args.manifest or not args.cache:
        parser.error("--manifest and --cache are required for a live development run")
    result = run_compact_prompt_dev_v24(
        args.manifest, model=args.model, cache_path=args.cache,
        output_path=args.output,
    )
    print(json.dumps({"output": args.output, "states": len(result["rows"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
