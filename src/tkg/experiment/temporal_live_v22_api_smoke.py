"""Post-freeze API-model smoke over the synthetic live v2.2 environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_environment_v2 import TemporalWikipediaEnvironmentV2
from tkg.experiment.temporal_eval_schema_v2 import (
    PublicTemporalCaseV2, load_cases_v2, structured_submission_from_dict,
)
from tkg.experiment.temporal_evaluation_v2 import (
    reference_route_diagnostics_v2, validate_structured_submission_v2,
)
from tkg.experiment.temporal_live_ranker_v22 import ApiLiveActionRankerV22
from tkg.experiment.temporal_live_runner_v22 import (
    LiveSearchConfigV22, run_live_temporal_search_v22,
)
from tkg.experiment.temporal_live_v22_synthetic import (
    SYNTHETIC_CASE_PATH, SyntheticLiveBackendV22,
)
from tkg.experiment.temporal_submission_v2 import StructuredSubmissionProposerV2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-5.4-mini")
    parser.add_argument("--case", default=str(SYNTHETIC_CASE_PATH))
    parser.add_argument("--ranker-cache", required=True)
    parser.add_argument("--submission-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-expansions", type=int, default=12)
    args = parser.parse_args()
    for path in (args.ranker_cache, args.submission_cache, args.output):
        assert_new_output_path(path)

    row = json.loads(Path(args.case).read_text(encoding="utf-8"))["cases"][0]
    public_case = PublicTemporalCaseV2(
        case_id=row["case_id"], model_id=args.model, question=row["question"],
        start_page=row["start_page"], cutoff_date=row["cutoff_date"],
        target_date=row["target_date"],
    )
    backend = SyntheticLiveBackendV22()
    ranker = ApiLiveActionRankerV22(args.model, cache_path=args.ranker_cache)
    proposer = StructuredSubmissionProposerV2(
        args.model, cache_path=args.submission_cache,
    )
    try:
        search = run_live_temporal_search_v22(
            public_case=public_case,
            backend=backend,
            environment=TemporalWikipediaEnvironmentV2(backend),
            ranker=ranker,
            submission_proposer=proposer,
            config=LiveSearchConfigV22(
                beam_width=1, max_expansions=args.max_expansions,
                max_actions_per_state=1, dense_action_limit=30, seed=17,
            ),
        )
    finally:
        proposer.close()
        ranker.close()

    evaluation = None
    diagnostics = None
    # The private object is first instantiated here, after search has returned.
    private_case = load_cases_v2(args.case)[0]
    if search.final_state.submitted is not None:
        submission = structured_submission_from_dict(search.final_state.submitted)
        transitions = [
            action for action in search.final_state.action_trace
            if action["action"]["kind"] in {"FOLLOW_LINK", "SWITCH_SNAPSHOT"}
        ]
        actions_valid = all(
            action["result"] == "ok"
            and (
                action["hyperlink_valid"] is True
                or action["revision_valid"] is True
            )
            for action in transitions
        )
        evaluation = validate_structured_submission_v2(
            case=private_case, submission=submission,
            trajectory_evidence=list(search.final_state.collected_evidence),
            trajectory_actions_valid=actions_valid,
        )
    evaluation_for_diagnostics = evaluation or {"end_to_end_success": False}
    diagnostics = reference_route_diagnostics_v2(
        route=private_case.reference_routes[0],
        funnel_steps=[
            step["action_funnel"] for step in search.audit_steps
            if "action_funnel" in step
        ],
        action_trace=list(search.final_state.action_trace),
        case=private_case,
        evaluation=evaluation_for_diagnostics,
        search_stop_reason=search.stop_reason,
    )
    payload = {
        "schema_version": "open-world-temporal-live-api-smoke-v2.2",
        "model": args.model,
        "search": search.to_dict(),
        "posthoc_private_evaluation": evaluation,
        "posthoc_reference_diagnostics": diagnostics,
        "private_case_loaded_only_post_search": True,
        "formal_conclusion_allowed": False,
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[done] api live smoke: stop={search.stop_reason} "
        f"submitted={search.final_state.submitted is not None} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
