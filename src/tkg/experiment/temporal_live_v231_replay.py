"""Zero-network replay of the five frozen v2.3 joint-controller responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tkg.experiment.joint_controller_v23 import ApiJointRankAndSubmitControllerV23
from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_environment_v2 import TemporalWikipediaEnvironmentV2
from tkg.experiment.temporal_eval_schema_v2 import PublicTemporalCaseV2
from tkg.experiment.temporal_live_runner_v23 import LiveSearchConfigV23
from tkg.experiment.temporal_live_runner_v231 import run_live_temporal_search_v231
from tkg.experiment.temporal_live_v23_synthetic import (
    SYNTHETIC_CASE_V23, SyntheticJointBackendV23,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="openai/gpt-5.4-mini")
    args = parser.parse_args()
    assert_new_output_path(args.output)
    row = json.loads(SYNTHETIC_CASE_V23.read_text(encoding="utf-8"))["cases"][0]
    case = PublicTemporalCaseV2(
        case_id=row["case_id"], model_id=args.model, question=row["question"],
        start_page=row["start_page"], cutoff_date=row["cutoff_date"],
        target_date=row["target_date"],
    )

    def forbid_new_call(*unused_args, **unused_kwargs):
        del unused_args, unused_kwargs
        raise AssertionError("v2.3.1 replay attempted a new model call")

    controller = ApiJointRankAndSubmitControllerV23(
        args.model, cache_path=args.cache, call_model_fn=forbid_new_call,
    )
    backend = SyntheticJointBackendV23()
    try:
        result = run_live_temporal_search_v231(
            public_case=case, backend=backend,
            environment=TemporalWikipediaEnvironmentV2(backend),
            controller=controller,
            config=LiveSearchConfigV23(
                beam_width=1, max_expansions=16, max_actions_per_state=1,
                dense_action_limit=30, seed=23,
            ),
        )
    finally:
        controller.close()
    payload = {
        "schema_version": "terminal-only-cache-replay-v2.3.1",
        "source_cache": args.cache,
        "new_model_calls": 0,
        "success": False,
        "terminal_status": result.final_state.stop_reason,
        "complete_trajectory_available": True,
        "submitted_answer": result.final_state.submitted,
        "cached_controller_responses_replayed": result.controller_calls,
        "search": result.to_dict(),
        "formal_conclusion_allowed": False,
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[done] v2.3.1 replay: terminal={result.final_state.stop_reason} "
        f"calls={result.controller_calls} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
