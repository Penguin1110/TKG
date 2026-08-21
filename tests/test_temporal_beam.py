"""Engineering contracts for temporal graph-constrained beam search."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from tkg.experiment.contracts import PageLink, PageSnapshot
from tkg.experiment.temporal_beam import (
    AnswerProposal, BeamSearchConfig, RankerOutput, TemporalAction,
    TemporalSearchRequest,
    _score_actions, evidence_id, initial_beam_state, legal_actions,
    legal_actions_with_compaction,
    recompute_cumulative_score,
    run_temporal_beam_search,
    run_temporal_greedy_search,
)
from tkg.experiment.temporal_beam_runner import (
    _collect_evidence, _compact_audit_value, _posthoc_private_action_funnel,
    external_engineering_metrics,
    load_public_cases, public_request_from_record,
)
from tkg.experiment.temporal_beam_ranker import (
    ApiUtilityRanker, RankerContractError, _extract_json,
)


def _page(title, revision, as_of, links=(), content=None):
    return PageSnapshot(
        title=title,
        page_id=sum(ord(char) for char in title),
        revision_id=revision,
        timestamp=f"{as_of}T12:00:00Z",
        as_of=as_of,
        content=content or f"Visible content for {title} at {as_of}.",
        links=[PageLink(target=target, anchor=anchor) for target, anchor in links],
        source_url=f"https://example.invalid/?oldid={revision}",
    )


class _Backend:
    def __init__(self):
        self.pages = {
            ("a", "2024-06-01"): _page(
                "A", 1, "2024-06-01",
                (("B", "B link"), ("Bee", "B alias"), ("C", "C link")),
            ),
            ("a", "2025-01-01"): _page(
                "A", 2, "2025-01-01", (("B", "new B link"),),
            ),
            ("b", "2024-06-01"): _page(
                "B", 3, "2024-06-01", (("A", "back"),),
            ),
            ("b", "2025-01-01"): _page(
                "B", 4, "2025-01-01", (("A", "back"),),
            ),
            ("c", "2024-06-01"): _page("C", 5, "2024-06-01"),
            ("c", "2025-01-01"): _page("C", 6, "2025-01-01"),
        }

    def fetch_page(self, title, as_of=None):
        canonical = "B" if title.casefold() == "bee" else title
        selected = "2024-06-01" if as_of < "2025-01-01" else "2025-01-01"
        page = self.pages[(canonical.casefold(), selected)]
        return replace(page, as_of=as_of)

    def list_revision_dates(self, title, from_date, to_date, limit):
        del title, from_date, to_date, limit
        return ["2024-06-01", "2025-01-01"]


@dataclass
class _Ranker:
    submit_empty: bool = False
    ranker_name: str = "deterministic_test_ranker"

    def propose_answer(self, request, state, visible_evidence, *, seed):
        del request, visible_evidence, seed
        if self.submit_empty or state.current_page != "B":
            return AnswerProposal(answer="")
        page = state.collected_evidence[-1]
        return AnswerProposal(
            answer="B", supporting_evidence_ids=(evidence_id(page),),
        )

    def rank(self, request, state, visible_evidence, actions, *, seed):
        del request, visible_evidence, seed
        scores = {}
        for action in actions:
            score = -20.0
            if action.kind == "LIST_REVISIONS":
                score = 10.0
            elif action.kind == "SWITCH_SNAPSHOT" and action.params["revision_id"] == 2:
                score = 9.0
            elif action.kind == "FOLLOW_LINK" and action.params["page_title"] == "B":
                score = 8.0
            elif action.kind == "SUBMIT_ANSWER" and state.current_page == "B":
                score = 7.0
            scores[action.action_id] = score
        return RankerOutput(
            scores=scores,
            score_kind="api_fallback_utility_softmax_log_score",
            reasoning_summary=f"At {state.current_page} revision {state.current_revision_id}",
            extracted_entities=(state.current_page,),
        )


def _request():
    return TemporalSearchRequest(
        case_id="public-case",
        question="Which visible entity is the answer?",
        start_page="A",
        cutoff_date="2024-06-01",
        target_date="2025-01-01",
    )


def _config(**changes):
    values = {
        "beam_width": 1,
        "max_expansions": 12,
        "max_actions_per_state": 2,
        "max_links": 10,
        "revision_limit": 2,
        "seed": 17,
    }
    values.update(changes)
    return BeamSearchConfig(**values)


def test_same_input_and_seed_reproduce_the_same_beam():
    left = run_temporal_beam_search(_request(), _Backend(), _Ranker(), _config())
    right = run_temporal_beam_search(_request(), _Backend(), _Ranker(), _config())
    assert left.final_state.to_dict() == right.final_state.to_dict()
    assert left.audit_steps == right.audit_steps


def test_follow_and_switch_actions_obey_revision_graph_semantics():
    result = run_temporal_beam_search(_request(), _Backend(), _Ranker(), _config())
    trace = result.final_state.action_trace
    switches = [row for row in trace if row["action"]["kind"] == "SWITCH_SNAPSHOT"]
    follows = [row for row in trace if row["action"]["kind"] == "FOLLOW_LINK"]
    assert switches and follows
    assert all(row["from_node"][0] == row["to_node"][0] for row in switches)
    assert all(row["revision_valid"] is True for row in switches)
    assert all(row["hyperlink_valid"] is True for row in follows)


def test_beam_width_one_matches_greedy_reference():
    beam = run_temporal_beam_search(_request(), _Backend(), _Ranker(), _config())
    greedy = run_temporal_greedy_search(_request(), _Backend(), _Ranker(), _config())
    assert beam.final_state.action_trace == greedy.final_state.action_trace
    assert beam.final_state.submitted_answer == greedy.final_state.submitted_answer


def test_same_information_state_is_deduplicated():
    # B and Bee are distinct legal link labels but resolve to the same B revision.
    result = run_temporal_beam_search(
        _request(), _Backend(), _Ranker(),
        _config(beam_width=5, max_actions_per_state=5, max_expansions=10),
    )
    assert result.repeated_state_count >= 1
    reasons = [
        candidate["pruning_reason"]
        for step in result.audit_steps for candidate in step["candidate_actions"]
    ]
    assert "duplicate_state_lower_score" in reasons


def test_cumulative_score_and_pruning_are_recomputable():
    result = run_temporal_beam_search(_request(), _Backend(), _Ranker(), _config())
    assert result.final_state.cumulative_score == recompute_cumulative_score(
        result.final_state
    )
    assert result.to_dict()["score_recomputable"] is True
    assert all(
        "pruning_reason" in candidate
        for step in result.audit_steps for candidate in step["candidate_actions"]
    )


def test_max_expansions_is_a_strict_global_stop():
    result = run_temporal_beam_search(
        _request(), _Backend(), _Ranker(), _config(max_expansions=3),
    )
    assert result.expansions == 3
    assert result.stop_reason == "max_expansions"


def test_empty_answer_and_errors_are_not_successful_submission():
    result = run_temporal_beam_search(
        _request(), _Backend(), _Ranker(submit_empty=True), _config(),
    )
    assert not result.final_state.submitted_answer
    assert result.final_state.stop_reason == "exhausted_no_legal_progress"
    assert result.stop_reason != "submit_answer"


def test_submit_is_generated_separately_and_bound_to_visible_evidence():
    backend = _Backend()
    state = initial_beam_state(_request(), backend)
    graph_actions, _ = legal_actions(state, backend, _config())
    assert all(action.kind != "SUBMIT_ANSWER" for action in graph_actions)

    result = run_temporal_beam_search(_request(), backend, _Ranker(), _config())
    answer_steps = [
        step for step in result.audit_steps
        if step["answer_candidate"]["supported"]
    ]
    assert answer_steps
    assert answer_steps[0]["answer_candidate"]["literal_support_gate_passed"] is True
    assert answer_steps[0]["answer_candidate"]["semantic_relation_support"] == (
        "not_evaluated"
    )
    submits = [
        candidate for step in answer_steps for candidate in step["candidate_actions"]
        if candidate["kind"] == "SUBMIT_ANSWER"
    ]
    assert submits
    assert submits[0]["params"]["answer"] == "B"
    assert submits[0]["params"]["support_policy"] == "verbatim_visible_evidence_v1"
    assert submits[0]["params"]["supporting_evidence_ids"]


def test_document_order_compaction_records_full_and_visible_action_sets():
    backend = _Backend()
    state = initial_beam_state(_request(), backend)
    actions, _, audit = legal_actions_with_compaction(
        state, backend, _config(max_links=1),
    )
    assert [row.params["page_title"] for row in actions if row.kind == "FOLLOW_LINK"] == [
        "B"
    ]
    assert [
        row["params"]["page_title"]
        for row in audit["pre_compaction_actions"] if row["kind"] == "FOLLOW_LINK"
    ] == ["B", "Bee", "C"]
    assert audit["post_compaction_count"] < audit["pre_compaction_count"]


def test_generic_scorer_rejects_omissions_instead_of_assigning_a_floor():
    actions = [TemporalAction("FOLLOW_LINK", {"page_title": "B"}, "Follow B")]
    try:
        _score_actions(actions, RankerOutput(
            scores={}, score_kind="api_fallback_utility_softmax_log_score",
        ))
    except RankerContractError as exc:
        assert "ranker_score_coverage_mismatch" in str(exc)
    else:
        raise AssertionError("omitted action was silently assigned a score")


def test_private_route_audit_is_posthoc_and_labels_compaction_recall_failure():
    result = run_temporal_beam_search(
        _request(), _Backend(), _Ranker(), _config(max_links=1, max_expansions=1),
    )
    metrics = _posthoc_private_action_funnel(result, {
        "reasoning_chain": [{
            "index": 0,
            "source_title": "A",
            "source_revision_id": 1,
            "prior_revision_id": None,
            "target_title": "C",
        }],
        "new_answer_keywords": [],
    })
    assert metrics["legal_candidate_recall"] == 1.0
    assert metrics["post_compaction_recall@30"] == 0.0
    assert metrics["action_funnel_failure_class"] == "COMPACTION_RECALL_FAILURE"


def test_unsupported_answer_never_becomes_a_submit_action():
    result = run_temporal_beam_search(
        _request(), _Backend(), _Ranker(submit_empty=True), _config(),
    )
    assert all(
        candidate["kind"] != "SUBMIT_ANSWER"
        for step in result.audit_steps for candidate in step["candidate_actions"]
    )
    assert all(
        step["answer_candidate"]["supported"] is False
        for step in result.audit_steps
    )


def test_full_audit_contains_state_actions_evidence_and_result():
    result = run_temporal_beam_search(_request(), _Backend(), _Ranker(), _config())
    expanded = [
        (step, candidate)
        for step in result.audit_steps for candidate in step["candidate_actions"]
        if candidate["expanded"]
    ]
    assert expanded
    assert all(step["parent_state"] and step["visible_evidence"] for step, _ in expanded)
    assert all(candidate["resulting_state"] for _, candidate in expanded)
    assert all("selected_actions" in step and "retained_actions" in step
               for step in result.audit_steps)


def test_public_request_boundary_rejects_private_fields_and_qids():
    base = {
        "id": "x", "model_id": "m", "question": "A public question?",
        "start_page": "A", "cutoff_date": "2024-01-01",
        "target_date": "2025-01-01",
    }
    request, model = public_request_from_record(base)
    assert request.to_dict() == {
        "case_id": "x", "question": "A public question?", "start_page": "A",
        "cutoff_date": "2024-01-01", "target_date": "2025-01-01",
    }
    assert model == "m"
    for addition in (
        {"private_chain": []},
        {"gold_route": []},
        {"answer_aliases": ["secret"]},
    ):
        try:
            public_request_from_record({**base, **addition})
        except ValueError as exc:
            assert "forbidden" in str(exc)
        else:
            raise AssertionError("private inference field crossed the public boundary")
    try:
        public_request_from_record({**base, "question": "Follow Q123 now?"})
    except ValueError as exc:
        assert "Wikidata identifier" in str(exc)
    else:
        raise AssertionError("Wikidata QID crossed the public boundary")


def test_two_case_smoke_manifest_is_public_only():
    cases = load_public_cases("examples/temporal_beam_smoke/public_cases.json")
    assert len(cases) == 2
    serialized = str([request.to_dict() for request, _ in cases])
    for hidden in ("David Lammy", "Yvette Cooper", "Ed Balls", "Inverness", "Q"):
        assert hidden not in serialized


def test_api_fallback_freezes_only_complete_dense_scores(tmp_path):
    request = TemporalSearchRequest(
        case_id="SECRET_CASE_ID_MUST_NOT_ENTER_PROMPT",
        question="Public question?", start_page="A",
        cutoff_date="2024-06-01", target_date="2025-01-01",
    )
    backend = _Backend()
    config = _config()
    state = initial_beam_state(request, backend)
    actions, page = legal_actions(state, backend, config)
    calls = []

    def fake_call(model, messages, temperature):
        del model, temperature
        prompt = messages[-1]["content"]
        calls.append(prompt)
        return json.dumps({
            "reasoning_summary": "brief",
            "extracted_entities": ["A"],
            "evidence_notes": [],
            "action_utilities": {
                action.action_id: 5 - index
                for index, action in enumerate(actions)
            },
        })

    ranker = ApiUtilityRanker(
        "test-model", cache_path=tmp_path / "ranker.db", call_model_fn=fake_call,
    )
    try:
        first = ranker.rank(request, state, [page.to_dict()], actions, seed=17)
        second = ranker.rank(request, state, [page.to_dict()], actions, seed=17)
    finally:
        ranker.close()
    assert first.scores == second.scores
    assert set(first.scores) == {action.action_id for action in actions}
    assert len(calls) == 1
    assert request.case_id not in calls[0]
    assert first.raw and first.raw["action_id_coverage_complete"] is True
    assert first.raw["expected_action_count"] == len(actions)
    assert first.raw["returned_action_count"] == len(actions)


def test_api_ranker_retries_once_then_requires_exact_id_coverage(tmp_path):
    request = _request()
    backend = _Backend()
    state = initial_beam_state(request, backend)
    actions, page = legal_actions(state, backend, _config())
    calls = []

    def fake_call(model, messages, temperature):
        del model, temperature
        calls.append(messages[-1]["content"])
        selected = actions[:1] if len(calls) == 1 else actions
        return json.dumps({
            "action_utilities": {action.action_id: 1 for action in selected},
        })

    ranker = ApiUtilityRanker(
        "test-model", cache_path=tmp_path / "retry.db", call_model_fn=fake_call,
    )
    try:
        output = ranker.rank(request, state, [page.to_dict()], actions, seed=17)
    finally:
        ranker.close()
    assert len(calls) == 2
    assert "CORRECTIVE RETRY" in calls[1]
    assert output.raw and output.raw["contract_attempt"] == 2
    assert len(output.raw["contract_failures_before_success"]) == 1


def test_api_ranker_invalid_after_one_retry_is_not_scored(tmp_path):
    request = _request()
    backend = _Backend()
    state = initial_beam_state(request, backend)
    actions, page = legal_actions(state, backend, _config())
    calls = []

    def fake_call(model, messages, temperature):
        del model, messages, temperature
        calls.append(1)
        return json.dumps({
            "action_utilities": {actions[0].action_id: 1},
        })

    ranker = ApiUtilityRanker(
        "test-model", cache_path=tmp_path / "invalid.db", call_model_fn=fake_call,
    )
    try:
        try:
            ranker.rank(request, state, [page.to_dict()], actions, seed=17)
        except RankerContractError as exc:
            assert "ranker_call_invalid_after_retry" in str(exc)
        else:
            raise AssertionError("incomplete ranker output was accepted")
    finally:
        ranker.close()
    assert len(calls) == 2


def test_api_dense_ranker_fails_closed_above_action_limit(tmp_path):
    ranker = ApiUtilityRanker(
        "test-model", cache_path=tmp_path / "limit.db",
        call_model_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("oversized dense prompt must not call the API")
        ),
    )
    actions = [
        TemporalAction("FOLLOW_LINK", {"page_title": f"Page {index}"}, f"Page {index}")
        for index in range(31)
    ]
    state = initial_beam_state(_request(), _Backend())
    try:
        try:
            ranker.rank(_request(), state, [], actions, seed=17)
        except RankerContractError as exc:
            assert "candidate_count_exceeds_dense_limit" in str(exc)
        else:
            raise AssertionError("oversized dense action set was accepted")
    finally:
        ranker.close()


def test_ranker_json_rejects_duplicate_action_ids():
    try:
        _extract_json('{"action_utilities":{"same":1,"same":2}}')
    except ValueError as exc:
        assert "duplicate JSON key: same" in str(exc)
    else:
        raise AssertionError("duplicate action ID was silently overwritten")


def test_api_answer_candidate_is_a_separate_cached_call(tmp_path):
    request = _request()
    state = initial_beam_state(request, _Backend())
    page = state.collected_evidence[-1]
    calls = []

    def fake_call(model, messages, temperature):
        del model, temperature
        calls.append(messages[-1]["content"])
        return json.dumps({
            "answer": "A",
            "supporting_evidence_ids": [evidence_id(page)],
        })

    ranker = ApiUtilityRanker(
        "test-model", cache_path=tmp_path / "answers.db", call_model_fn=fake_call,
    )
    try:
        first = ranker.propose_answer(request, state, [page], seed=17)
        second = ranker.propose_answer(request, state, [page], seed=17)
    finally:
        ranker.close()
    assert first.answer == second.answer == "A"
    assert first.supporting_evidence_ids == second.supporting_evidence_ids
    assert first.supporting_evidence_ids == (evidence_id(page),)
    assert len(calls) == 1
    assert "separate from graph-action ranking" in calls[0]


def test_external_metrics_do_not_call_no_answer_success():
    metrics = external_engineering_metrics({
        "trajectory": [{
            "action": "follow_link", "from_title": "A", "to_title": "A",
            "revision_id": 1, "result": "Error: invalid link",
        }],
        "evidence_pages": [], "final_answer": "", "stop_reason": "max_steps",
    }, "2025-01-01")
    assert metrics["answer_status"] == "no_answer"
    assert metrics["tool_api_cache_error"] is True
    assert metrics["formal_success"] == "not_evaluated_engineering_smoke"


def test_jsonl_compaction_keeps_one_hash_bound_evidence_record():
    page = _page("A", 1, "2024-06-01").to_dict()
    value = {"visible_evidence": [page], "state": {"collected_evidence": [page]}}
    found = {}
    _collect_evidence(value, found)
    compact = _compact_audit_value(value)
    assert len(found) == 1
    left = compact["visible_evidence"][0]
    right = compact["state"]["collected_evidence"][0]
    assert left["evidence_id"] == right["evidence_id"] == next(iter(found))
    assert "content" not in left and len(left["content_sha256"]) == 64
