"""Held-out comparability diagnostics for open-weight action scores v2.4."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

from tkg.experiment.compact_joint_controller_v24 import CompactSubmitSlotActionV24
from tkg.experiment.open_weight_action_scorer_v24 import (
    HuggingFaceCausalLMBackendV24, OpenWeightConditionalActionScorerV24,
)
from tkg.experiment.open_weight_live_controller_v24 import (
    HierarchicalOpenWeightLiveControllerV24,
)
from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_environment_v2 import EnvironmentActionV2
from tkg.experiment.temporal_eval_schema_v2 import PublicTemporalCaseV2


DIAGNOSTIC_SCHEMA_V24 = "open-weight-action-comparability-diagnostic-v2.4"


@dataclass(frozen=True)
class _RenamedAction:
    action_id: str
    kind: str
    label: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id, "kind": self.kind,
            "label": self.label, "params": self.params,
            "environment_order": None,
        }


def _navigation_state(index: int) -> tuple[str, list[Any], set[str]]:
    person = f"Heldout Navigator {index}"
    institution = f"Heldout Institute {index}"
    prompt = (
        f"Question: Where was the person appointed director of {institution} "
        "after the cutoff born?\n"
        f"Visible evidence: On 7 July 2025, {person} became director of "
        f"{institution}.\nChoose the single best action to make progress."
    )
    titles = [
        f"Biography of {person}", f"Profile of {person}",
        *[f"Unrelated Archive {index}-{offset}" for offset in range(5)],
    ]
    rotation = index % len(titles)
    titles = titles[rotation:] + titles[:rotation]
    actions: list[Any] = [
        EnvironmentActionV2(
            "FOLLOW_LINK", {"page_title": title}, f"Follow hyperlink to {title}",
            environment_order=order,
        ) for order, title in enumerate(titles)
    ]
    actions.extend([
        EnvironmentActionV2(
            "LIST_REVISIONS", {"cursor": None, "page_size": 50},
            "List revisions of the current page",
        ),
        CompactSubmitSlotActionV24(),
    ])
    progress = {
        action.action_id for action in actions
        if action.to_dict().get("params", {}).get("page_title") in {
            f"Biography of {person}", f"Profile of {person}",
        }
    }
    return prompt, actions, progress


def _prompt_with_actions(prompt: str, actions: list[Any]) -> str:
    public_actions = [{
        "kind": action.kind, "label": action.label,
        "params": action.to_dict().get("params", {}),
    } for action in actions]
    public_actions.sort(key=lambda row: json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ))
    return prompt + "\nCandidate actions:\n" + json.dumps(
        public_actions, ensure_ascii=False,
    )


def run_diagnostics_v24(
    *, model: str, device: str = "cuda", dtype: str = "float16",
) -> dict[str, Any]:
    backend = HuggingFaceCausalLMBackendV24(model, device=device, dtype=dtype)
    scorer = OpenWeightConditionalActionScorerV24(backend)
    hierarchical = HierarchicalOpenWeightLiveControllerV24(
        scorer, compact_payload_proposer=lambda state: None,
        payload_proposer_name="diagnostic_none",
    )
    navigation: list[dict[str, Any]] = []
    type_scores: dict[str, list[float]] = {}
    order_max_deltas = []
    id_max_deltas = []
    for index in range(8):
        prompt, actions, progress = _navigation_state(index)
        scored = scorer.score(_prompt_with_actions(prompt, actions), actions)
        reversed_actions = list(reversed(actions))
        reversed_scored = scorer.score(
            _prompt_with_actions(prompt, reversed_actions), reversed_actions,
        )
        order_max_deltas.append(max(
            abs(scored.scores[action_id] - reversed_scored.scores[action_id])
            for action_id in scored.scores
        ))
        renamed: list[Any] = [
            _RenamedAction(
                action_id=f"renamed:{index}:{offset}", kind=action.kind,
                label=action.label, params=dict(action.to_dict().get("params", {})),
            ) for offset, action in enumerate(actions)
        ]
        renamed_scores = scorer.score(_prompt_with_actions(prompt, renamed), renamed)
        id_max_deltas.append(max(
            abs(scored.scores[action.action_id] -
                renamed_scores.scores[renamed[offset].action_id])
            for offset, action in enumerate(actions)
        ))
        ordered = sorted(scored.scores, key=lambda key: (-scored.scores[key], key))
        best_rank = min(ordered.index(action_id) + 1 for action_id in progress)
        navigation.append({
            "state_id": f"heldout-navigation-{index}",
            "best_progress_rank": best_rank,
            "progress_in_top_1": best_rank <= 1,
            "progress_in_top_3": best_rank <= 3,
            "scores": scored.scores,
        })
        for action in actions:
            type_scores.setdefault(action.kind, []).append(
                scored.scores[action.action_id]
            )

    name_prompt = (
        "Question: Inspect the biography linked by the visible appointment evidence.\n"
        "Choose the best next hyperlink."
    )
    name_actions: list[Any] = [
        EnvironmentActionV2(
            "FOLLOW_LINK", {"page_title": name}, f"Follow hyperlink to {name}",
        ) for name in (
            "John Smith", "Eirik Horneland", "Xylophanes Quenby-Zoric",
        )
    ]
    name_result = scorer.score(name_prompt, name_actions)

    submit_rows: list[dict[str, Any]] = []
    for complete in (False, True):
        for index in range(4):
            person = f"Submit Person {index}"
            city = f"Submit City {index}"
            bridge = (
                f"On 3 March 2025, {person} became director of Submit Lab {index}."
            )
            tail = f"{person} was born in {city}."
            evidence = bridge + ("\n" + tail if complete else "")
            prompt = (
                "Question: Where was the person appointed director after the cutoff born?\n"
                f"Visible evidence:\n{evidence}\nChoose the best next action."
            )
            actions = [
                EnvironmentActionV2(
                    "LIST_LINKS", {"cursor": None}, "List more hyperlinks",
                ),
                EnvironmentActionV2(
                    "LIST_REVISIONS", {"cursor": None}, "List more revisions",
                ),
                CompactSubmitSlotActionV24(),
            ]
            scores = scorer.score(prompt, actions).scores
            ordered = sorted(scores, key=lambda key: (-scores[key], key))
            mode_case = PublicTemporalCaseV2(
                case_id=f"submit-diagnostic-{complete}-{index}",
                model_id=model,
                question="Where was the person appointed director after the cutoff born?",
                start_page="Diagnostic", cutoff_date="2024-06-01",
                target_date="2025-12-31",
            )
            evidence_pages = [{
                "evidence_id": f"ev_{complete}_{index}",
                "title": "Diagnostic evidence", "content": evidence,
            }]
            mode_scores, mode_records = hierarchical._mode_scores(
                mode_case, SimpleNamespace(collected_evidence=evidence_pages),
            )
            submit_rows.append({
                "complete_evidence": complete, "index": index,
                "submit_rank": ordered.index("submit_slot:v1") + 1,
                "submit_top_1": ordered[0] == "submit_slot:v1",
                "scores": scores,
                "hierarchical_mode_log_probabilities": mode_scores,
                "hierarchical_submit_selected": (
                    mode_scores["SUBMIT"] > mode_scores["CONTINUE"]
                ),
                "hierarchical_mode_records": mode_records,
            })

    return {
        "schema_version": DIAGNOSTIC_SCHEMA_V24,
        "model": model, "device": device, "dtype": dtype,
        "score_kind": "length_normalized_conditional_logprob",
        "action_id_in_scored_text": False,
        "order_invariance": {
            "max_absolute_score_delta": max(order_max_deltas),
            "passed_exact": max(order_max_deltas) == 0,
        },
        "action_id_invariance": {
            "max_absolute_score_delta": max(id_max_deltas),
            "passed_exact": max(id_max_deltas) == 0,
        },
        "action_type_scores": {
            kind: {"mean": mean(values), "count": len(values)}
            for kind, values in sorted(type_scores.items())
        },
        "name_commonness_scores": {
            action.to_dict()["params"]["page_title"]: name_result.scores[action.action_id]
            for action in name_actions
        },
        "heldout_navigation": {
            "states": len(navigation),
            "recall_at_1": mean(
                bool(row["progress_in_top_1"]) for row in navigation
            ),
            "recall_at_3": mean(
                bool(row["progress_in_top_3"]) for row in navigation
            ),
            "rows": navigation,
        },
        "submit_decision": {
            "complete_top_1_rate": mean(
                bool(row["submit_top_1"]) for row in submit_rows
                if row["complete_evidence"]
            ),
            "incomplete_top_1_rate": mean(
                bool(row["submit_top_1"]) for row in submit_rows
                if not row["complete_evidence"]
            ),
            "rows": submit_rows,
        },
        "hierarchical_submit_decision": {
            "complete_submit_rate": mean(
                bool(row["hierarchical_submit_selected"]) for row in submit_rows
                if row["complete_evidence"]
            ),
            "incomplete_false_submit_rate": mean(
                bool(row["hierarchical_submit_selected"]) for row in submit_rows
                if not row["complete_evidence"]
            ),
        },
        "formal_conclusion_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert_new_output_path(args.output)
    result = run_diagnostics_v24(
        model=args.model, device=args.device, dtype=args.dtype,
    )
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "output": args.output,
        "recall_at_3": result["heldout_navigation"]["recall_at_3"],
        "complete_submit_top_1": result["submit_decision"]["complete_top_1_rate"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
