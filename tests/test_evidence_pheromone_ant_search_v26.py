from __future__ import annotations

import random
from dataclasses import replace

import pytest

from tkg.experiment.compact_joint_controller_v24 import CompactJointOutputV24
from tkg.experiment.compact_submission_v24 import CompactSubmissionValidationV24
from tkg.experiment.contracts import PageLink, PageSnapshot
from tkg.experiment.evidence_pheromone_ant_search_v26 import (
    AntSearchConfigV26, PheromoneLedgerV26, assert_public_only_v26,
    frozen_method_config_v26, replay_pheromone_history_v26,
    run_temporal_evidence_ant_search_v26, score_and_sample_action_v26,
    transition_public_progress_v26,
)
from tkg.experiment.temporal_environment_v2 import TemporalWikipediaEnvironmentV2
from tkg.experiment.temporal_eval_schema_v2 import PublicTemporalCaseV2
from tkg.experiment.temporal_live_runner_v22 import LiveBeamStateV22


def _case() -> PublicTemporalCaseV2:
    return PublicTemporalCaseV2(
        case_id="fictional_case", model_id="fictional_model",
        question=(
            "Which organization appointed the later coach, and what academy "
            "did the next coach attend?"
        ),
        start_page="Start Club", cutoff_date="2024-06-01",
        target_date="2026-01-01",
    )


def _state(**changes: object) -> LiveBeamStateV22:
    base = LiveBeamStateV22(
        current_page="Start Club", current_revision_id=1,
        current_revision_timestamp="2024-06-01T00:00:00Z",
        snapshot_as_of="2024-06-01", reasoning_summary="",
        extracted_entities=(), collected_evidence=(),
        retrieved_link_actions=(), retrieved_revision_actions=(),
        link_query_started=False, link_next_cursor=None,
        links_exhausted=False, revision_query_started=False,
        revision_next_cursor=None, revisions_exhausted=False,
        environment_queries_used=0, visited_nodes=(("start club", 1),),
        action_trace=(), cumulative_score=0.0, finished=False,
        submitted=None,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def test_frozen_method_parameters_are_distinct_and_stochastic_has_no_pheromone() -> None:
    stochastic = frozen_method_config_v26("STOCHASTIC_LM", 11)
    structural = frozen_method_config_v26("STRUCTURAL_ACO", 11)
    evidence = frozen_method_config_v26("EVIDENCE_ACO", 11)
    assert (stochastic.alpha, stochastic.gamma) == (0.0, 0.0)
    assert structural.alpha == evidence.alpha == 0.8
    assert evidence.gamma > structural.gamma > 0.0


def test_private_key_leakage_guard_fails_closed() -> None:
    assert_public_only_v26(_case().to_dict())
    with pytest.raises(ValueError, match="private-key leakage"):
        assert_public_only_v26({"question": "public", "reference_routes": []})
    with pytest.raises(ValueError, match="private-key leakage"):
        assert_public_only_v26({"nested": {"accepted_final_answer_aliases": ["x"]}})


def test_seeded_sampling_is_reproducible() -> None:
    state = _state()
    candidates = [
        {"action_id": "a", "kind": "LIST_LINKS", "params": {},
         "label": "List links", "action_score": -1.0},
        {"action_id": "b", "kind": "LIST_REVISIONS", "params": {},
         "label": "List revisions", "action_score": -1.2},
    ]
    config = frozen_method_config_v26("STOCHASTIC_LM", 37)
    first = score_and_sample_action_v26(
        case=_case(), state=state, candidates=candidates, pheromones={},
        config=config, rng=random.Random(37),
    )
    second = score_and_sample_action_v26(
        case=_case(), state=state, candidates=candidates, pheromones={},
        config=config, rng=random.Random(37),
    )
    assert first == second


def test_delayed_evidence_reinforces_low_lm_ancestor_and_replays() -> None:
    config = frozen_method_config_v26("EVIDENCE_ACO", 11)
    ledger = PheromoneLedgerV26.create(config)
    record = ledger.update(
        selected_edge="later_evidence", ant_path=["low_lm_edge", "later_evidence"],
        reward=2.0, global_step=1,
    )
    assert ledger.values["low_lm_edge"] > config.pheromone_initial
    assert ledger.values["later_evidence"] > ledger.values["low_lm_edge"]
    assert replay_pheromone_history_v26(config, [record]) == ledger.values


def test_pheromone_evaporates_and_is_bounded() -> None:
    config = frozen_method_config_v26("STRUCTURAL_ACO", 23)
    ledger = PheromoneLedgerV26.create(config)
    ledger.values["old"] = 2.0
    ledger.update(
        selected_edge="new", ant_path=["new"], reward=1.0, global_step=1,
    )
    assert ledger.values["old"] == pytest.approx(2.0 * (1.0 - config.evaporation))
    for step in range(2, 100):
        ledger.update(
            selected_edge="new", ant_path=["new"], reward=-100.0,
            global_step=step,
        )
    assert ledger.values["new"] == config.pheromone_min


def test_incomplete_evidence_never_gets_completion_reward() -> None:
    parent = _state()
    evidence = ({
        "evidence_id": "ev_fictional", "title": "Coach Example",
        "revision_id": 2, "timestamp": "2025-01-01T00:00:00Z",
        "content": "The coach joined an organization.",
    },)
    child = replace(
        parent, current_page="Coach Example", current_revision_id=2,
        collected_evidence=evidence,
        visited_nodes=(*parent.visited_nodes, ("coach example", 2)),
    )
    progress = transition_public_progress_v26(
        _case(), parent, child,
        {"kind": "FOLLOW_LINK", "params": {"page_title": "Coach Example"}},
    )
    assert progress.new_evidence_count == 1
    assert not progress.public_submission_complete
    assert progress.evidence_reward < 2.0


def test_cycle_and_zero_progress_are_penalized() -> None:
    parent = _state()
    cycle = transition_public_progress_v26(
        _case(), parent, parent,
        {"kind": "FOLLOW_LINK", "params": {"page_title": "Start Club"}},
    )
    assert cycle.repeated_node
    assert cycle.structural_reward < 0.0
    idle = transition_public_progress_v26(
        _case(), parent, parent,
        {"kind": "LIST_LINKS", "params": {}},
    )
    assert idle.zero_progress
    assert idle.structural_reward < 0.0


class _SyntheticBackend:
    def __init__(self) -> None:
        self.pages = {
            1: PageSnapshot(
                title="Start Club", page_id=1, revision_id=1,
                timestamp="2024-06-01T00:00:00Z", as_of="2024-06-01",
                content="Start club page", links=[PageLink("Coach Page", "coach")],
            ),
            2: PageSnapshot(
                title="Coach Page", page_id=2, revision_id=2,
                timestamp="2024-06-01T00:00:00Z", as_of="2024-06-01",
                content="Coach page", links=[PageLink("Start Club", "club")],
            ),
        }

    def fetch_page(self, title: str, as_of: str | None = None) -> PageSnapshot:
        del as_of
        return next(page for page in self.pages.values() if page.title == title)

    def fetch_revision(self, revision_id: int) -> PageSnapshot:
        return self.pages[revision_id]

    def list_revision_metadata_page(
        self, title: str, from_date: str, to_date: str, *,
        cursor: str | None = None, page_size: int = 50,
    ) -> dict[str, object]:
        del from_date, to_date, cursor, page_size
        return {"title": title, "revisions": [], "next_cursor": None}


class _SyntheticController:
    controller_name = "fictional_dense_controller"

    def control(
        self, case: PublicTemporalCaseV2, state: LiveBeamStateV22,
        actions: list[object], *, seed: int, budget: dict[str, int],
    ) -> CompactJointOutputV24:
        del case, state, seed, budget
        scores = {}
        for action in actions:
            kind = str(getattr(action, "kind"))
            scores[str(getattr(action, "action_id"))] = (
                0.0 if kind in {"FOLLOW_LINK", "LIST_LINKS"} else -100.0
            )
        return CompactJointOutputV24(
            scores=scores, reasoning_summary="fictional", extracted_entities=(),
            evidence_notes=(), submission=None,
            submission_schema_validation=CompactSubmissionValidationV24(
                "abstained", False, "fictional incomplete evidence",
            ),
            abstain_reason="fictional incomplete evidence",
            attempts=({"fictional": True},),
            score_kind="length_normalized_conditional_logprob",
        )


def test_synthetic_runner_stops_at_exact_budget_and_is_reproducible() -> None:
    backend = _SyntheticBackend()
    environment = TemporalWikipediaEnvironmentV2(backend)
    config = AntSearchConfigV26(
        method="STOCHASTIC_LM", seed=53, max_expansions=7,
        ant_count=2, max_steps_per_ant=4, alpha=0.0, gamma=0.0,
    )
    first = run_temporal_evidence_ant_search_v26(
        public_case=_case(), backend=backend, environment=environment,
        controller=_SyntheticController(), config=config,
    )
    second = run_temporal_evidence_ant_search_v26(
        public_case=_case(), backend=backend, environment=environment,
        controller=_SyntheticController(), config=config,
    )
    assert first.expansions == second.expansions == 7
    assert first.stop_reason == "max_expansions"
    assert [row["selected_action"]["action_id"] for row in first.audit_steps] == [
        row["selected_action"]["action_id"] for row in second.audit_steps
    ]
    assert first.to_dict()["pheromone_replay_valid"]
