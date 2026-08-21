"""Post-freeze API smoke for the fresh joint-controller v2.3 fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from tkg.experiment.joint_controller_v23 import ApiJointRankAndSubmitControllerV23
from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_environment_v2 import TemporalWikipediaEnvironmentV2
from tkg.experiment.temporal_eval_schema_v2 import (
    PublicTemporalCaseV2, load_cases_v2, structured_submission_from_dict,
)
from tkg.experiment.temporal_evaluation_v2 import (
    reference_route_diagnostics_v2, validate_structured_submission_v2,
)
from tkg.experiment.temporal_live_runner_v23 import (
    LiveSearchConfigV23, run_live_temporal_search_v23,
)
from tkg.experiment.temporal_live_v23_synthetic import (
    SYNTHETIC_CASE_V23, SyntheticJointBackendV23,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-5.4-mini")
    parser.add_argument("--case", default=str(SYNTHETIC_CASE_V23))
    parser.add_argument("--controller-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert_new_output_path(args.output)

    row = json.loads(Path(args.case).read_text(encoding="utf-8"))["cases"][0]
    public_case = PublicTemporalCaseV2(
        case_id=row["case_id"], model_id=args.model, question=row["question"],
        start_page=row["start_page"], cutoff_date=row["cutoff_date"],
        target_date=row["target_date"],
    )
    backend = SyntheticJointBackendV23()
    controller = ApiJointRankAndSubmitControllerV23(
        args.model, cache_path=args.controller_cache,
    )
    search = None
    search_error = ""
    try:
        try:
            search = run_live_temporal_search_v23(
                public_case=public_case, backend=backend,
                environment=TemporalWikipediaEnvironmentV2(backend),
                controller=controller,
                config=LiveSearchConfigV23(
                    beam_width=1, max_expansions=16, max_actions_per_state=1,
                    dense_action_limit=30, seed=23,
                ),
            )
        except RuntimeError as exc:
            search_error = str(exc)
    finally:
        controller.close()

    if search is None:
        connection = sqlite3.connect(args.controller_cache)
        try:
            cached = [
                {"cache_key": key, "raw_response": response}
                for key, response in connection.execute(
                    "SELECT cache_key, response FROM ranker_cache ORDER BY cache_key"
                )
            ]
        finally:
            connection.close()
        payload = {
            "schema_version": "joint-live-api-smoke-v2.3",
            "freeze_manifest": (
                "docs/OPEN_WORLD_LIVE_RUNNER_V2_3_FREEZE_2026-08-16.json"
            ),
            "model": args.model,
            "search": None,
            "posthoc_private_evaluation": None,
            "posthoc_reference_diagnostics": None,
            "status": {
                "runner_completed": False,
                "failure": "NO_RETAINED_STATE_RUNTIME_ERROR",
                "error": search_error,
                "cached_joint_response_count": len(cached),
            },
            "cached_joint_responses": cached,
            "controller_cache_sha256": hashlib.sha256(
                Path(args.controller_cache).read_bytes()
            ).hexdigest(),
            "private_case_loaded_only_post_search": True,
            "formal_conclusion_allowed": False,
        }
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[done] joint API smoke failed closed: {search_error}")
        return 0

    # Private objects are first created after the frozen runner has returned.
    private_case = load_cases_v2(args.case)[0]
    evaluation = None
    if search.final_state.submitted is not None:
        evaluation = validate_structured_submission_v2(
            case=private_case,
            submission=structured_submission_from_dict(search.final_state.submitted),
            trajectory_evidence=list(search.final_state.collected_evidence),
            trajectory_actions_valid=all(
                action["result"] == "ok"
                for action in search.final_state.action_trace
                if action["action"]["kind"] in {"FOLLOW_LINK", "SWITCH_SNAPSHOT"}
            ),
        )
    diagnostics = reference_route_diagnostics_v2(
        route=private_case.reference_routes[0],
        funnel_steps=[step["action_funnel"] for step in search.audit_steps
                      if "action_funnel" in step],
        action_trace=list(search.final_state.action_trace), case=private_case,
        evaluation=evaluation or {"end_to_end_success": False},
        search_stop_reason=search.stop_reason,
    )
    payload = {
        "schema_version": "joint-live-api-smoke-v2.3",
        "freeze_manifest": "docs/OPEN_WORLD_LIVE_RUNNER_V2_3_FREEZE_2026-08-16.json",
        "model": args.model,
        "search": search.to_dict(),
        "posthoc_private_evaluation": evaluation,
        "posthoc_reference_diagnostics": diagnostics,
        "status": {
            "navigation_reached_bridge_page": any(
                page["title"] == "Alternative Route" and page["revision_id"] == 111
                for page in search.final_state.collected_evidence
            ),
            "navigation_reached_tail_page": any(
                page["title"] == "Scientist Z" and page["revision_id"] == 120
                for page in search.final_state.collected_evidence
            ),
            "dense_ranking_contract_passed": all(
                step.get("action_funnel", {}).get("ranker_contract_valid") is True
                for step in search.audit_steps
            ),
            "structured_submission_passed": bool(
                search.final_state.submitted is not None
            ),
            "end_to_end_synthetic_success": bool(
                evaluation and evaluation["end_to_end_success"]
            ),
        },
        "private_case_loaded_only_post_search": True,
        "formal_conclusion_allowed": False,
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[done] joint API smoke: stop={search.stop_reason} "
        f"submitted={search.final_state.submitted is not None} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
