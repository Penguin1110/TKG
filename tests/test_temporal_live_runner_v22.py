from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tkg.experiment.temporal_beam import RankerContractError
from tkg.experiment.temporal_environment_v2 import (
    EnvironmentActionV2, TemporalWikipediaEnvironmentV2,
)
from tkg.experiment.temporal_eval_schema_v2 import PublicTemporalCaseV2
from tkg.experiment.temporal_live_ranker_v22 import (
    ApiLiveActionRankerV22, CallableLiveActionRankerV22,
)
from tkg.experiment.temporal_live_runner_v22 import (
    LiveSearchConfigV22, run_live_temporal_search_v22,
)
from tkg.experiment.temporal_live_v22_synthetic import (
    SYNTHETIC_CASE_PATH, SyntheticLiveBackendV22,
    SyntheticSubmissionProposerV22, run_synthetic_live_smoke_v22,
    synthetic_score_v22,
)


def _public_case() -> PublicTemporalCaseV2:
    row = json.loads(SYNTHETIC_CASE_PATH.read_text(encoding="utf-8"))["cases"][0]
    return PublicTemporalCaseV2(
        case_id=row["case_id"],
        model_id=row["model_id"],
        question=row["question"],
        start_page=row["start_page"],
        cutoff_date=row["cutoff_date"],
        target_date=row["target_date"],
    )


def _search(max_expansions: int = 12):
    backend = SyntheticLiveBackendV22()
    return run_live_temporal_search_v22(
        public_case=_public_case(),
        backend=backend,
        environment=TemporalWikipediaEnvironmentV2(backend),
        ranker=CallableLiveActionRankerV22(synthetic_score_v22),
        submission_proposer=SyntheticSubmissionProposerV22(),
        config=LiveSearchConfigV22(
            beam_width=1,
            max_expansions=max_expansions,
            max_actions_per_state=1,
            dense_action_limit=30,
            seed=17,
        ),
    )


def test_synthetic_live_search_starts_at_start_and_finds_alternative_route() -> None:
    artifact = run_synthetic_live_smoke_v22()
    search = artifact["search"]
    trace = search["final_state"]["action_trace"]
    assert [row["action"]["kind"] for row in trace] == [
        "LIST_LINKS", "FOLLOW_LINK", "LIST_REVISIONS", "SWITCH_SNAPSHOT",
        "LIST_LINKS", "FOLLOW_LINK", "SUBMIT_ANSWER",
    ]
    assert trace[1]["action"]["params"]["page_title"] == "Route B"
    assert trace[3]["action"]["params"]["revision_id"] == 20
    assert search["final_state"]["submitted"]["answer"] == "Answer City"
    assert artifact["posthoc_private_evaluation"]["end_to_end_success"] is True
    assert artifact["posthoc_reference_diagnostics"]["reference_route_recalled"] is False
    assert artifact["posthoc_reference_diagnostics"]["alternative_valid_route_found"] is True
    assert all(artifact["checks"].values())


def test_compaction_keeps_route_b_and_drops_private_route_a() -> None:
    result = _search()
    start_steps = [
        step for step in result.audit_steps
        if step.get("parent_state", {}).get("current_page") == "Start"
        and step.get("action_funnel", {}).get("solver_retrieved_actions")
    ]
    funnel = start_steps[-1]["action_funnel"]
    retrieved_targets = {
        row["params"].get("page_title")
        for row in funnel["solver_retrieved_actions"]
        if row["kind"] == "FOLLOW_LINK"
    }
    compacted_targets = {
        row["params"].get("page_title")
        for row in funnel["compacted_ranker_actions"]
        if row["kind"] == "FOLLOW_LINK"
    }
    assert {"Route A", "Route B"} <= retrieved_targets
    assert "Route B" in compacted_targets
    assert "Route A" not in compacted_targets
    manifest = next(
        row for row in result.environment_manifests if row.page_title == "Start"
    )
    assert manifest.action_count == len(manifest.actions) == 32
    assert manifest.link_count == 32


def test_same_seed_reproduces_beam_and_scores_recompute() -> None:
    left = _search().to_dict()
    right = _search().to_dict()
    assert left == right
    assert left["score_recomputable"] is True


def test_equivalent_state_dedup_ignores_path_score_and_history() -> None:
    state = _search(max_expansions=1).final_state
    equivalent = replace(
        state,
        cumulative_score=state.cumulative_score - 100,
        action_trace=(*state.action_trace, {"diagnostic": "different path history"}),
    )
    assert state.dedup_key() == equivalent.dedup_key()
    assert state.state_id == equivalent.state_id


def test_max_expansions_stops_strictly_without_success() -> None:
    result = _search(max_expansions=2)
    assert result.expansions == 2
    assert result.stop_reason == "max_expansions"
    assert result.final_state.submitted is None


@pytest.mark.parametrize("mode", ["missing", "unexpected", "duplicate"])
def test_api_ranker_fails_closed_after_one_retry(tmp_path: Path, mode: str) -> None:
    action = EnvironmentActionV2("LIST_LINKS", {"cursor": None, "page_size": 50}, "x")
    calls = []

    def invalid_call(*args, **kwargs):
        del args, kwargs
        calls.append(1)
        if mode == "missing":
            scores = []
        elif mode == "unexpected":
            scores = [
                {"action_id": action.action_id, "utility": 1},
                {"action_id": "extra", "utility": 0},
            ]
        else:
            scores = [
                {"action_id": action.action_id, "utility": 1},
                {"action_id": action.action_id, "utility": 0},
            ]
        return json.dumps({"action_utilities": scores})

    ranker = ApiLiveActionRankerV22(
        "test", cache_path=tmp_path / "ranker.db", call_model_fn=invalid_call,
    )
    state = _search(max_expansions=1).final_state
    with pytest.raises(RankerContractError, match="invalid_after_retry"):
        ranker.rank(_public_case(), state, [action], seed=17)
    assert len(calls) == 2
    ranker.close()


def test_runner_api_is_public_only() -> None:
    parameters = run_live_temporal_search_v22.__annotations__
    assert "public_case" in parameters
    assert "private_case" not in parameters
