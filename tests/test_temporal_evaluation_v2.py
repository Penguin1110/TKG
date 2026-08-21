"""Open-world multi-path evaluation v2 contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from tkg.experiment.contracts import PageLink, PageSnapshot
from tkg.experiment.temporal_environment_v2 import (
    EnvironmentActionV2, TemporalWikipediaEnvironmentV2,
    action_funnel_record_v2, compact_solver_actions_v2,
    execute_environment_query_v2, initial_environment_queries_v2,
)
from tkg.experiment.temporal_eval_schema_v2 import (
    ClaimWitnessV2, EvaluationCaseV2, PrivateReferenceRouteV2,
    ReferenceActionV2, RequiredClaimV2, StructuredSubmissionV2,
    SubmittedClaimV2, load_cases_v2, validate_evaluation_case_v2,
)
from tkg.experiment.temporal_evaluation_v2 import (
    SemanticDecisionV2, assert_no_gold_leak_v2, evidence_id_v2,
    reference_route_diagnostics_v2,
    validate_structured_submission_v2,
)
from tkg.experiment.temporal_submission_v2 import (
    StructuredSubmissionProposerV2, structured_submit_action_v2,
)
from tkg.experiment.temporal_semantic_judge_v2 import LLMSemanticClaimJudgeV2


def _page(title, revision, content, links=(), timestamp="2025-01-01T00:00:00Z"):
    return PageSnapshot(
        title=title,
        page_id=revision,
        revision_id=revision,
        timestamp=timestamp,
        as_of=timestamp,
        content=content,
        links=[PageLink(target=target, anchor=anchor) for target, anchor in links],
        source_url=f"https://example.invalid/?oldid={revision}",
    )


def _witness(title, revision, excerpt):
    return ClaimWitnessV2(
        page_title=title,
        revision_id=revision,
        evidence_excerpt=excerpt,
        evidence_hash=hashlib.sha256(excerpt.encode()).hexdigest(),
    )


def _case():
    bridge_excerpt_a = "Club appointed Person X as head coach on 1 January 2025."
    bridge_excerpt_b = "On 1 January 2025, Person X became the Club head coach."
    tail_excerpt = "Person X was born in Answer City."
    bridge = RequiredClaimV2(
        claim_id="bridge_1",
        subject="Club",
        relation="appointed head coach",
        object="Person X",
        event_time="2025-01-01",
        witnesses=(
            _witness("Route A", 10, bridge_excerpt_a),
            _witness("Route B", 20, bridge_excerpt_b),
        ),
    )
    tail = RequiredClaimV2(
        claim_id="tail",
        subject="Person X",
        relation="place of birth",
        object="Answer City",
        event_time=None,
        witnesses=(_witness("Person X", 30, tail_excerpt),),
    )
    reference = PrivateReferenceRouteV2(
        route_id="route_a",
        actions=(
            ReferenceActionV2(
                kind="FOLLOW_LINK", parent_page="Start", parent_revision_id=1,
                page_title="Route A",
            ),
            ReferenceActionV2(
                kind="FOLLOW_LINK", parent_page="Route A", parent_revision_id=10,
                page_title="Person X",
            ),
        ),
        distance=2,
    )
    return EvaluationCaseV2(
        case_id="synthetic-multiroute",
        model_id="test-model",
        question="Who was appointed, and where was that person born?",
        start_page="Start",
        cutoff_date="2024-01-01",
        target_date="2025-12-31",
        accepted_final_answer_aliases=("Answer City",),
        critical_claims=(bridge,),
        tail_relation=tail,
        validated_evidence_requirements={"semantic": True},
        event_time_constraints={
            "bridge_1": {"operator": "equals", "event_time": "2025-01-01"},
        },
        reference_routes=(reference,),
    )


def _alternative_evidence():
    route_b = _page(
        "Route B", 20,
        "On 1 January 2025, Person X became the Club head coach.",
        (("Person X", "Person X"),),
    ).to_dict()
    person = _page(
        "Person X", 30, "Person X was born in Answer City.",
        (("Answer City", "Answer City"),),
    ).to_dict()
    for page in (route_b, person):
        page["evidence_id"] = evidence_id_v2(page)
    return route_b, person


def _submission(route_b, person):
    return StructuredSubmissionV2(
        answer="Answer City",
        critical_claims=(SubmittedClaimV2(
            claim_id="bridge_1",
            subject="Club",
            relation="appointed head coach",
            object="Person X",
            event_time="2025-01-01",
            supporting_evidence_ids=(route_b["evidence_id"],),
        ),),
        tail_claim=SubmittedClaimV2(
            claim_id="tail",
            subject="Person X",
            relation="place of birth",
            object="Answer City",
            event_time=None,
            supporting_evidence_ids=(person["evidence_id"],),
        ),
    )


def test_alternative_route_is_accepted_when_reference_route_is_missing():
    case = _case()
    route_b, person = _alternative_evidence()
    evaluation = validate_structured_submission_v2(
        case=case,
        submission=_submission(route_b, person),
        trajectory_evidence=[route_b, person],
        trajectory_actions_valid=True,
    )
    diagnostics = reference_route_diagnostics_v2(
        route=case.reference_routes[0],
        funnel_steps=[],
        action_trace=[
            {"kind": "FOLLOW_LINK", "params": {"page_title": "Route B"}},
            {"kind": "FOLLOW_LINK", "params": {"page_title": "Person X"}},
        ],
        case=case,
        evaluation=evaluation,
        search_stop_reason="submit_answer",
    )
    assert evaluation["end_to_end_success"] is True
    assert diagnostics["reference_route_recalled"] is False
    assert diagnostics["alternative_valid_route_found"] is True
    assert "REFERENCE_ROUTE_NOT_COMPLETED" in diagnostics["diagnostic_labels"]
    assert "REFERENCE_ACTION_NOT_RANKED" not in diagnostics["diagnostic_labels"]
    assert "NO_VALIDATED_EVIDENCE_CHAIN_FOUND" not in diagnostics["diagnostic_labels"]


def test_alternative_witness_revision_is_accepted():
    case = _case()
    route_b, person = _alternative_evidence()
    result = validate_structured_submission_v2(
        case=case,
        submission=_submission(route_b, person),
        trajectory_evidence=[route_b, person],
        trajectory_actions_valid=True,
    )
    support = result["critical_claim_results"][0]["support"]
    assert support["matched_witnesses"][0]["revision_id"] == 20
    assert result["critical_bridge_evidence_complete"] is True


def test_reference_route_missing_does_not_override_primary_success():
    case = _case()
    route_b, person = _alternative_evidence()
    evaluation = validate_structured_submission_v2(
        case=case, submission=_submission(route_b, person),
        trajectory_evidence=[route_b, person], trajectory_actions_valid=True,
    )
    diagnostics = reference_route_diagnostics_v2(
        route=case.reference_routes[0], funnel_steps=[], action_trace=[],
        case=case, evaluation=evaluation, search_stop_reason="submit_answer",
    )
    assert evaluation["end_to_end_validated_answer_accuracy"] == 1
    assert diagnostics["reference_route_completion_rate"] == 0.0


def test_reference_action_outside_compaction_has_no_fake_rank():
    case = _case()
    route_b, person = _alternative_evidence()
    evaluation = validate_structured_submission_v2(
        case=case, submission=_submission(route_b, person),
        trajectory_evidence=[route_b, person], trajectory_actions_valid=True,
    )
    reference = case.reference_routes[0].actions[0]
    action = EnvironmentActionV2(
        kind="FOLLOW_LINK", params={"page_title": reference.page_title},
        label="reference",
    ).to_dict()
    diagnostics = reference_route_diagnostics_v2(
        route=case.reference_routes[0],
        funnel_steps=[{
            "parent_page": "Start", "parent_revision_id": 1,
            "environment_legal_actions": [action],
            "solver_retrieved_actions": [action],
            "compacted_ranker_actions": [], "ranker_scores": {},
            "ranker_contract_valid": True, "expanded_actions": [],
        }],
        action_trace=[], case=case, evaluation=evaluation,
        search_stop_reason="submit_answer",
    )
    first = diagnostics["reference_action_details"][0]
    assert first["rank_of_reference_action"] == "not_evaluable"
    assert "REFERENCE_ACTION_NOT_RANKED" in diagnostics["diagnostic_labels"]


def test_correct_alias_without_bridge_evidence_is_not_success():
    case = _case()
    route_b, person = _alternative_evidence()
    submission = replace(_submission(route_b, person), critical_claims=())
    result = validate_structured_submission_v2(
        case=case, submission=submission, trajectory_evidence=[person],
        trajectory_actions_valid=True,
    )
    assert result["final_answer_correct"] is True
    assert result["critical_bridge_evidence_complete"] is False
    assert result["end_to_end_success"] is False


def test_evidence_not_seen_by_trajectory_is_rejected():
    case = _case()
    route_b, person = _alternative_evidence()
    result = validate_structured_submission_v2(
        case=case, submission=_submission(route_b, person),
        trajectory_evidence=[person], trajectory_actions_valid=True,
    )
    assert result["critical_claim_results"][0][
        "evidence_ids_from_trajectory"
    ] is False
    assert result["end_to_end_success"] is False


def test_non_witness_semantic_route_preserves_full_judge_record():
    case = _case()
    route_b, person = _alternative_evidence()
    only_reference_witness = replace(
        case.critical_claims[0], witnesses=(case.critical_claims[0].witnesses[0],),
    )
    case = replace(case, critical_claims=(only_reference_witness,))

    class _Judge:
        def judge(self, required, submitted, evidence):
            judge_input = {
                "required_claim": required.to_dict(),
                "submitted_claim": submitted.to_dict(),
                "evidence": evidence,
            }
            return SemanticDecisionV2(
                supported=True,
                confidence=0.97,
                reason="The sentence states the requested appointment relation.",
                model="semantic-test-model",
                version="2026-08-16",
                judge_input=judge_input,
                judge_output={"decision": "supported"},
                deterministic_guards={"evidence_owned": True},
            )

    result = validate_structured_submission_v2(
        case=case, submission=_submission(route_b, person),
        trajectory_evidence=[route_b, person], trajectory_actions_valid=True,
        semantic_judge=_Judge(),
    )
    support = result["critical_claim_results"][0]["support"]
    assert result["end_to_end_success"] is True
    assert support["judge_used"] is True
    assert support["decision"]["model"] == "semantic-test-model"
    assert support["decision"]["review_status"] == (
        "machine_pass_human_review_required"
    )


def test_posthoc_llm_semantic_judge_is_cached_and_auditable(tmp_path):
    case = _case()
    route_b, _ = _alternative_evidence()
    calls = []

    def fake_call(model, messages, temperature):
        calls.append((model, messages, temperature))
        return json.dumps({
            "decision": "supported", "confidence": 0.91,
            "reason": "The cited sentence states the appointment and date.",
        })

    judge = LLMSemanticClaimJudgeV2(
        "openai/gpt-4.1", version="frozen-test", cache_path=tmp_path / "judge.db",
        call_model_fn=fake_call,
    )
    submitted = _submission(route_b, _alternative_evidence()[1]).critical_claims[0]
    try:
        first = judge.judge(case.critical_claims[0], submitted, [route_b])
        second = judge.judge(case.critical_claims[0], submitted, [route_b])
    finally:
        judge.close()
    assert first.supported is second.supported is True
    assert len(calls) == 1
    assert first.judge_input["cited_evidence"][0]["revision_id"] == 20
    assert first.judge_output["raw_response"]
    assert first.deterministic_guards["judge_is_posthoc_only"] is True


def test_private_material_cannot_enter_public_inference_payload():
    case = _case()
    assert_no_gold_leak_v2(case.public_view().to_dict(), private_case=case)
    with pytest.raises(AssertionError, match="private inference key"):
        assert_no_gold_leak_v2({
            **case.public_view().to_dict(),
            "reference_route": case.reference_routes[0].to_dict(),
        }, private_case=case)


def test_structured_submission_prompt_contains_no_private_gold(tmp_path):
    case = _case()
    seen = []

    def fake_call(model, messages, temperature):
        del model, temperature
        seen.append(messages[-1]["content"])
        return json.dumps({
            "schema_version": "structured-temporal-evidence-submission-v2",
            "answer": "",
            "critical_claims": [],
            "tail_claim": {
                "claim_id": "tail", "subject": "", "relation": "",
                "object": "", "event_time": None, "supporting_evidence_ids": [],
            },
        })

    proposer = StructuredSubmissionProposerV2(
        "test-model", cache_path=tmp_path / "submit.db", call_model_fn=fake_call,
    )
    try:
        proposer.propose(case.public_view(), [], seed=17)
    finally:
        proposer.close()
    assert len(seen) == 1
    for secret in ("Answer City", "Route A", "Route B", "revision 10"):
        assert secret not in seen[0]
    assert "accepted_final_answer_aliases" not in seen[0]
    with pytest.raises(AssertionError, match="private inference value"):
        assert_no_gold_leak_v2({
            **case.public_view().to_dict(), "hint": "Answer City",
        }, private_case=case)
    with pytest.raises(AssertionError, match="Wikidata"):
        assert_no_gold_leak_v2({
            **case.public_view().to_dict(), "hint": "Q123",
        }, private_case=case)


class _EnvironmentBackend:
    def __init__(self):
        self.pages = {
            1: _page(
                "Start", 1, "start",
                tuple((f"Page {index}", f"link {index}") for index in range(130)),
            ),
            2: _page("Start", 2, "later", timestamp="2025-02-01T00:00:00Z"),
            3: _page("Start", 3, "latest", timestamp="2025-03-01T00:00:00Z"),
        }
        self.revisions = [
            {"revision_id": 1, "timestamp": "2025-01-01T00:00:00Z"},
            {"revision_id": 2, "timestamp": "2025-02-01T00:00:00Z"},
            {"revision_id": 3, "timestamp": "2025-03-01T00:00:00Z"},
        ]

    def fetch_revision(self, revision_id):
        return self.pages[revision_id]

    def list_revision_metadata_page(
        self, title, from_date, to_date, *, cursor=None, page_size=50,
    ):
        del from_date, to_date
        offset = int(cursor or 0)
        rows = self.revisions[offset:offset + page_size]
        next_offset = offset + len(rows)
        return {
            "title": title,
            "revisions": rows,
            "next_cursor": str(next_offset) if next_offset < len(self.revisions) else None,
        }


def test_environment_full_list_is_separate_from_solver_compaction():
    backend = _EnvironmentBackend()
    env = TemporalWikipediaEnvironmentV2(backend)
    page = env.list_links(backend.pages[1], page_size=50)
    compacted, audit = compact_solver_actions_v2([page], dense_limit=30)
    assert page.full_count == 130
    assert len(page.items) == 50
    assert len(audit["solver_retrieved_actions"]) == 51  # 50 links + pagination
    assert len(compacted) == 30
    assert audit["compacted_ranker_actions"] != audit["solver_retrieved_actions"]


def test_link_pagination_can_retrieve_document_position_120():
    backend = _EnvironmentBackend()
    env = TemporalWikipediaEnvironmentV2(backend)
    first = env.list_links(backend.pages[1], page_size=50)
    second = env.list_links(
        backend.pages[1], cursor=first.next_cursor, page_size=50,
    )
    third = env.list_links(
        backend.pages[1], cursor=second.next_cursor, page_size=50,
    )
    assert third.items[20].params["page_title"] == "Page 120"
    assert third.items[20].environment_order == 120


def test_revision_pagination_is_complete_and_switch_stays_on_page():
    backend = _EnvironmentBackend()
    env = TemporalWikipediaEnvironmentV2(backend)
    actions, manifest = env.enumerate_revisions(
        backend.pages[1], from_date="2025-01-01", to_date="2025-12-31",
        page_size=1,
    )
    assert manifest["complete"] is True
    assert manifest["pages"] == 3
    assert {row.params["revision_id"] for row in actions} == {2, 3}
    assert all(env.verify_action(backend.pages[1], row) for row in actions)


def test_forced_and_end_to_end_modes_share_environment_query_builder():
    backend = _EnvironmentBackend()
    env = TemporalWikipediaEnvironmentV2(backend)
    queries = initial_environment_queries_v2(
        link_page_size=50, revision_page_size=2,
        from_date="2025-01-01", to_date="2025-12-31",
    )
    forced_page = execute_environment_query_v2(env, backend.pages[1], queries[0])
    end_to_end_page = execute_environment_query_v2(env, backend.pages[1], queries[0])
    assert forced_page.to_dict() == end_to_end_page.to_dict()
    assert forced_page.full_count == 130


def test_action_funnel_only_scores_conditional_on_exact_contract():
    backend = _EnvironmentBackend()
    env = TemporalWikipediaEnvironmentV2(backend)
    page = env.list_links(backend.pages[1], page_size=2)
    compacted, _ = compact_solver_actions_v2([page], dense_limit=3)
    complete = action_funnel_record_v2(
        environment_actions=list(page.items), retrieved_pages=[page], dense_limit=3,
        ranker_scores={row.action_id: 1.0 for row in compacted},
        expanded_action_ids=[compacted[0].action_id],
    )
    omitted = action_funnel_record_v2(
        environment_actions=list(page.items), retrieved_pages=[page], dense_limit=3,
        ranker_scores={compacted[0].action_id: 1.0}, expanded_action_ids=[],
    )
    assert complete["ranker_contract_valid"] is True
    assert omitted["ranker_contract_valid"] is False
    assert omitted["ranker_scores"] == {}


def test_structured_submit_action_contains_claims_and_evidence():
    route_b, person = _alternative_evidence()
    submission = _submission(route_b, person)
    action = structured_submit_action_v2(submission)
    assert action is not None and action.kind == "SUBMIT_ANSWER"
    assert action.params["critical_claims"][0]["supporting_evidence_ids"]
    assert action.params["tail_claim"]["object"] == "Answer City"


def test_legacy_case_loader_is_read_only_and_adds_witness_sets(tmp_path):
    source = "examples/temporal_beam_new_questions_v1/rare_coach_recovery_cases_v1.json"
    before = hashlib.sha256(open(source, "rb").read()).hexdigest()
    cases = load_cases_v2(source)
    after = hashlib.sha256(open(source, "rb").read()).hexdigest()
    assert before == after
    assert cases and cases[0].source_case_schema.endswith("multihop-v6")
    assert cases[0].critical_claims[0].witnesses
    assert len(cases[0].reference_routes) == 1
    assert validate_evaluation_case_v2(
        cases[0], require_formal_witness_records=True,
    )


def test_legacy_frozen_manifest_hashes_are_unchanged():
    manifest = json.load(open(
        "docs/SEARCH_CONTRACT_FREEZE_2026-08-15.json", encoding="utf-8",
    ))
    for path, expected in manifest["sha256"].items():
        actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert actual == expected, path
