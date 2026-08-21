"""Candidate-only 100-question generation contracts."""

import json

from tkg.experiment.candidate_question_batch import (
    EventRow, QUESTION_TEMPLATE_VERSION, _event_first_query,
    _event_query, _is_specific_office, _profile_event_seeds, _query,
    _discover_mixed_spines, _p39_event_skeletons, _questions_for_spines,
    _questions_for_staged_spines, _refresh_question_texts,
    _resolve_candidate_relations, _select_diverse_candidates, _select_spines,
)
from tkg.experiment.candidate_topology_registry import (
    load_candidate_topology_registry,
)
from tkg.experiment.temporal_relation_registry import load_temporal_relation_registry


def _entity(title, claims=None):
    return {
        "labels": {"en": {"value": title}},
        "sitelinks": {"enwiki": {"title": title}},
        "claims": claims or {},
    }


def _claim(qid):
    return {"rank": "normal", "mainsnak": {
        "datavalue": {"value": {"id": qid}},
    }}


def _interval_claim(qid, start, end=None):
    qualifiers = {
        "P580": [{"datavalue": {"value": {"time": f"+{start}T00:00:00Z"}}}],
    }
    if end:
        qualifiers["P582"] = [
            {"datavalue": {"value": {"time": f"+{end}T00:00:00Z"}}},
        ]
    return {**_claim(qid), "qualifiers": qualifiers}


def test_spine_selection_uses_locally_earliest_unambiguous_events():
    rows = [
        {
            "anchor_qid": "Q1", "person_0_qid": "Q2",
            "later_anchor_qid": "Q3", "person_2_qid": "Q4",
            "start_0": "2024-01-01", "end_0": "2024-07-01",
            "start_1": "2025-01-01", "start_2": "2025-06-01",
        },
        {
            "anchor_qid": "Q1", "person_0_qid": "Q2",
            "later_anchor_qid": "Q5", "person_2_qid": "Q6",
            "start_0": "2024-01-01", "end_0": "2024-07-01",
            "start_1": "2025-02-01", "start_2": "2025-07-01",
        },
    ]
    selected = _select_spines(rows, 10)
    assert len(selected) == 1
    assert selected[0]["later_anchor_qid"] == "Q3"
    assert selected[0]["person_2_qid"] == "Q4"


def test_candidate_questions_keep_answers_private_and_status_provisional():
    registry = load_temporal_relation_registry()
    spec = next(row for row in registry.relations if row.property_id == "P286")
    spine = {
        "anchor_qid": "Q1", "person_0_qid": "Q2",
        "later_anchor_qid": "Q3", "person_2_qid": "Q4",
        "start_0": "2024-01-01", "end_0": "2024-07-01",
        "start_1": "2025-01-01", "start_2": "2025-06-01",
    }
    entities = {
        "Q1": _entity("Original Team"), "Q2": _entity("First Coach"),
        "Q3": _entity("Later Team"),
        "Q4": _entity("Second Coach", {"P19": [_claim("Q5")]}),
        "Q5": _entity("Answer City"),
    }
    questions, missing = _questions_for_spines(
        {"P286": [spine]}, {"P286": spec}, entities,
        model_id="openai/gpt-4.1-mini", cutoff="2024-06-01",
        until="2026-08-14", max_questions=10, max_per_anchor=2,
        query_truncated={"P286": False},
    )
    assert missing == {"Q5"}
    assert len(questions) == 1
    question = questions[0]
    assert question["status"] == "pending_wikipedia_validation"
    assert question["expected_answer"] == "Answer City"
    assert "Answer City" not in question["benchmark_question"]
    assert "2025-01-01" not in question["benchmark_question"]
    assert "2025-01-01" in question["audit_question_with_dates"]
    assert question["hop_count"] == 4


def test_wording_refresh_is_offline_and_removes_selection_claim():
    registry = load_temporal_relation_registry()
    spec = next(row for row in registry.relations if row.property_id == "P286")
    raw = {
        "id": "q", "relation_property": "P286", "tail_property": "P19",
        "public_anchor": "Original Team", "knowledge_cutoff": "2024-06-01",
        "expected_answer_aliases": ["Answer City"],
        "private_chain": [
            {}, {"event_date": "2025-01-01"}, {"event_date": "2025-06-01"}, {},
        ],
    }
    refreshed = _refresh_question_texts([raw], {"P286": spec})
    assert len(refreshed) == 1
    question = refreshed[0]["benchmark_question"]
    assert "selected" not in question
    assert "first begin serving as head coach" in question
    assert refreshed[0]["question_template_version"] == QUESTION_TEMPLATE_VERSION


def test_inverted_p39_join_uses_first_position_and_unique_next_holder():
    events = [
        EventRow("Q1", "Q10", "2025-01-01"),
        EventRow("Q2", "Q10", "2025-03-01"),
        EventRow("Q1", "Q11", "2025-04-01"),
        EventRow("Q3", "Q11", "2025-06-01"),
    ]
    rows = _p39_event_skeletons(events, 10)
    assert rows == [{
        "person_0_qid": "Q1", "later_anchor_qid": "Q10",
        "person_2_qid": "Q2", "start_1": "2025-01-01",
        "start_2": "2025-03-01",
    }]


def test_event_query_keeps_endpoint_join_small_and_bounded():
    p39 = _event_query("P39", "2024-06-01", "2025-01-01", 100)
    assert "?subject p:P39 ?statement" in p39
    assert '2024-06-01T23:59:59Z' in p39
    assert "LIMIT 100" in p39
    assert "P1308" not in p39


def test_event_first_query_uses_profiled_events_instead_of_global_join(tmp_path):
    registry = load_temporal_relation_registry()
    spec = next(row for row in registry.relations if row.property_id == "P286")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"profiles": [{
        "relation": {"property_id": "P286"},
        "samples": [
            {"source_qid": "Q10", "target_qid": "Q20",
             "start": "2025-02-03"},
            {"source_qid": "Q11", "target_qid": "Q21",
             "start": "2024-01-01"},
        ],
    }]}))
    seeds = _profile_event_seeds(
        [str(profile)], property_id="P286", cutoff="2024-06-01",
        until="2026-08-16",
    )
    assert seeds == [{
        "source_qid": "Q10", "target_qid": "Q20", "start": "2025-02-03",
    }]
    query = _event_first_query(
        spec, cutoff="2024-06-01", until="2026-08-16",
        events=seeds, limit=50,
    )
    assert "VALUES (?laterAnchor ?p0 ?start1)" in query
    assert 'wd:Q10 wd:Q20 "2025-02-03T00:00:00Z"' in query
    assert "?laterAnchor p:P286 ?s1" not in query
    assert "?laterAnchor p:P286 ?s2" in query


def test_pair_novelty_filter_is_opt_in_and_checks_both_temporal_pairs():
    registry = load_temporal_relation_registry()
    spec = next(row for row in registry.relations if row.property_id == "P286")
    ordinary = _query(spec, "2024-06-01", "2026-08-16", 50)
    filtered = _query(
        spec, "2024-06-01", "2026-08-16", 50,
        require_pair_novelty=True,
    )
    assert "?priorStart1" not in ordinary
    assert "?priorStart2" not in ordinary
    assert "FILTER(?priorStart1 < ?start1)" in filtered
    assert "FILTER(?priorStart2 < ?start2)" in filtered
    targeted = _event_first_query(
        spec, cutoff="2024-06-01", until="2026-08-16",
        events=[{"source_qid": "Q10", "target_qid": "Q20", "start": "2025-01-01"}],
        limit=50, require_pair_novelty=True,
    )
    assert "FILTER(?priorStart1 < ?start1)" in targeted
    assert "FILTER(?priorStart2 < ?start2)" in targeted


def test_specific_office_gate_rejects_generic_multi_holder_position():
    assert _is_specific_office(_entity(
        "Prime Minister of Exampleland", {
            "P1001": [_claim("Q99")], "P1308": [_claim("Q98")],
        },
    ))
    assert not _is_specific_office(_entity("president"))
    assert not _is_specific_office(_entity("Member of the European Parliament"))


def test_staged_p39_question_preserves_linear_chain_and_hidden_dates():
    registry = load_temporal_relation_registry()
    specs = {row.property_id: row for row in registry.relations}
    spine = {
        "anchor_qid": "Q1", "person_0_qid": "Q2",
        "later_anchor_qid": "Q3", "person_2_qid": "Q4",
        "start_0": "2024-01-01", "end_0": "2024-07-01",
        "start_1": "2025-01-01", "start_2": "2025-06-01",
        "topology_id": "p39-to-p1308-office-succession",
        "domain_family": "career", "edge_0_property": "P39",
        "edge_1_property": "P39", "edge_2_property": "P1308",
        "specific_office_gate": "heuristic_pass_formal_singleton_audit_pending",
        "query_truncated": False,
    }
    entities = {
        "Q1": _entity("Office A"), "Q2": _entity("First Holder"),
        "Q3": _entity("Office B"),
        "Q4": _entity("Second Holder", {"P19": [_claim("Q5")]}),
        "Q5": _entity("Answer City"),
    }
    questions, missing = _questions_for_staged_spines(
        [spine], specs, entities, model_id="openai/gpt-4.1-mini",
        cutoff="2024-06-01", until="2026-08-15",
    )
    assert missing == {"Q5"}
    question = questions[0]
    assert question["relation_properties"] == ["P39", "P39", "P1308"]
    assert [edge["direction"] for edge in question["private_chain"][:3]] == [
        "inverse_at_cutoff", "forward", "inverse",
    ]
    assert "2025-01-01" not in question["benchmark_question"]


def test_diversity_sampler_enforces_sports_and_topology_caps():
    candidates = []
    for index in range(10):
        candidates.append({
            "id": f"sports-{index}", "topology_id": "sports",
            "tail_property": "P19", "public_anchor": f"team-{index}",
            "domain_family": "sports",
        })
        candidates.append({
            "id": f"career-{index}", "topology_id": f"career-{index % 2}",
            "tail_property": "P19", "public_anchor": f"office-{index}",
            "domain_family": "career",
        })
    selected = _select_diverse_candidates(
        candidates, max_questions=10, max_per_anchor=1,
        max_sports_share=0.2, max_topology_share=0.5,
    )
    assert len(selected) == 10
    assert sum(row["domain_family"] == "sports" for row in selected) <= 2
    assert max(
        sum(other["topology_id"] == row["topology_id"] for other in selected)
        for row in selected
    ) <= 5


def test_mixed_topology_starts_from_subjects_active_specific_office():
    entities = {
        "Q1": _entity("First Person", {
            "P39": [_interval_claim("Q10", "2024-01-01", "2024-08-01")],
        }),
        "Q2": _entity("New Party", {
            "P488": [_interval_claim("Q3", "2025-06-01")],
        }),
        "Q10": _entity("Prime Minister of Exampleland", {
            "P1308": [_interval_claim("Q1", "2024-01-01", "2024-08-01")],
        }),
    }
    rows = _discover_mixed_spines(
        "P102", [EventRow("Q1", "Q2", "2025-01-01")], entities,
        cutoff="2024-06-01", max_spines=10, until="2026-08-15",
        unresolved_saturation=False,
    )
    assert len(rows) == 1
    assert rows[0]["anchor_qid"] == "Q10"
    assert rows[0]["edge_0_property"] == "P39"
    assert rows[0]["edge_1_property"] == "P102"
    assert rows[0]["edge_2_property"] == "P488"


def test_topology_registry_versions_adapters_and_tail_contracts():
    topology = load_candidate_topology_registry()
    assert topology.schema_version == "candidate-topology-registry-v1"
    assert {row.event_property_id for row in topology.staged_topologies} == {
        "P39", "P102", "P108", "P1416",
    }
    assert len(topology.tails) == 18


def test_relation_resolution_uses_admission_intersection_and_dependencies():
    registry = load_temporal_relation_registry()
    topology = load_candidate_topology_registry()

    direct, staged, leaders = _resolve_candidate_relations(
        registry, topology, {"P169"},
    )
    assert set(direct) == {"P169"}
    assert staged == {}
    assert leaders == ("P169",)

    direct, staged, leaders = _resolve_candidate_relations(
        registry, topology, {"P39", "P1308"},
    )
    assert set(direct) == {"P1308"}
    assert set(staged) == {"P39"}
    assert leaders == ("P1308",)


def test_relation_resolution_rejects_unadmitted_manual_override():
    registry = load_temporal_relation_registry()
    topology = load_candidate_topology_registry()
    try:
        _resolve_candidate_relations(
            registry, topology, {"P169"}, requested_direct=["P286"],
        )
    except ValueError as exc:
        assert "not admitted" in str(exc)
    else:
        raise AssertionError("manual property selection bypassed relation admission")
