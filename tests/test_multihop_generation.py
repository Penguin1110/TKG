"""Offline tests for cutoff-relative multi-hop question contracts."""

from __future__ import annotations

import json
import sys

import pytest

from tkg.experiment.case_validation import validate_case, validate_chain_route
from tkg.experiment.multihop_generation import (
    MultiHopQuestionJudge, MultiHopQuestionWriter, MultiHopSeed, build_case,
    compose_canonical_question, compose_relative_question,
    has_infrastructure_error, question_wording_errors, validate_chain,
    validate_shortest_arena,
)
from tkg.experiment.model_cutoffs import get_model_cutoff, model_matches_cutoff
from tkg.experiment.temporal_runner import _semantic_waypoint_metrics
from tkg.experiment.temporal_candidate_discovery import (
    P286Candidate, attribute_quota_capacity, candidate_popularity, p286_query,
    verify_p286_candidate,
)
from tkg.experiment.attribute_tail_generation import (
    generate_attribute_tail_variants, select_claim, select_diverse_variants,
)
from tkg.experiment.relation_catalog import RELATION_BY_PROPERTY
from tkg.experiment.contracts import PageLink, PageSnapshot
from tkg.wikipedia.snapshot import page_version_key
from tkg.wikipedia.backend import WikipediaError
from tkg.wikipedia.pageviews import (
    PopularityRecord, PopularitySnapshot, fetch_top_pageviews, last_complete_month,
)


CUTOFF = "2024-06-01"
MIDDLE = "2025-01-01"
TARGET = "2026-01-01"


def _page(title, revision, as_of, content, links=()):
    return PageSnapshot(
        title=title, page_id=revision, revision_id=revision,
        timestamp=f"{as_of}T00:00:00Z", as_of=as_of, content=content,
        links=[PageLink(target=target, anchor=anchor) for target, anchor in links],
        source_url=f"https://example.invalid/?oldid={revision}",
    )


class FakeBackend:
    def __init__(self):
        self.pages = {
            ("Example Corp", CUTOFF): _page(
                "Example Corp", 1, CUTOFF,
                "Example Corp was led by [Alice Stone -> Alice Stone] as CEO.",
                [("Alice Stone", "Alice Stone")],
            ),
            ("Alice Stone", CUTOFF): _page(
                "Alice Stone", 2, CUTOFF, "Alice Stone was the CEO.",
            ),
            ("Alice Stone", MIDDLE): _page(
                "Alice Stone", 3, MIDDLE,
                "Her second successor as CEO was [Bob Reed -> Bob Reed].",
                [("Bob Reed", "Bob Reed")],
            ),
            ("Bob Reed", MIDDLE): _page(
                "Bob Reed", 4, MIDDLE, "Bob Reed biography.",
            ),
            ("Bob Reed", TARGET): _page(
                "Bob Reed", 5, TARGET,
                "Bob Reed is married to [Carol Reed -> Carol Reed].",
                [("Carol Reed", "Carol Reed")],
            ),
            ("Carol Reed", TARGET): _page(
                "Carol Reed", 6, TARGET, "Carol Reed biography.",
            ),
        }

    def fetch_page(self, title, as_of=None):
        try:
            return self.pages[(title, as_of)]
        except KeyError as exc:
            raise WikipediaError(f"missing fixture {title}@{as_of}") from exc

    def find_backlinks(self, title, as_of=None, max_results=50):
        sources = []
        for (source_title, source_as_of), page in self.pages.items():
            if source_as_of == as_of and any(
                link.target.casefold() == title.casefold() for link in page.links
            ):
                sources.append(source_title)
        return sorted(sources)[:max_results]


class AttributeBackend(FakeBackend):
    lang = "en"

    def __init__(self):
        super().__init__()
        self.pages[("Carol Reed", TARGET)] = _page(
            "Carol Reed", 6, TARGET,
            "Carol Reed was born in [Example City -> Example City].",
            [("Example City", "Example City")],
        )
        self.pages[("Example City", TARGET)] = _page(
            "Example City", 7, TARGET, "Example City biography.",
        )

    def resolve_title_qid(self, title):
        assert title == "Carol Reed"
        return "Q1"

    def get_wikidata_entities(self, qids, props="claims|labels|sitelinks"):
        entities = {
            "Q1": {
                "claims": {
                    "P19": [{
                        "rank": "normal",
                        "mainsnak": {"datavalue": {"value": {"id": "Q2"}}},
                    }],
                },
            },
            "Q2": {"sitelinks": {"enwiki": {"title": "Example City"}}},
        }
        return {qid: entities[qid] for qid in qids if qid in entities}


def _seed_dict():
    return {
        "id": "example_ceo_relative_chain",
        "model_id": "openai/gpt-4.1-mini",
        "target_as_of": TARGET,
        "anchor_label": "Example Corp",
        "category": "corporate",
        "old_answer_keywords": ["Alice Stone"],
        "hops": [
            {
                "source_title": "Example Corp", "target_title": "Alice Stone",
                "relation": "chief executive officer at cutoff", "as_of": CUTOFF,
                "relative_clause": (
                    "the person who was chief executive officer of {source} at the tested "
                    "model's registered knowledge cutoff"
                ),
                "evidence": "Example Corp was led by Alice Stone as CEO.",
                "target_aliases": ["Alice Stone"],
            },
            {
                "source_title": "Alice Stone", "target_title": "Bob Reed",
                "relation": "second successor as chief executive officer", "as_of": MIDDLE,
                "relative_clause": "the second successor as chief executive officer to {source}",
                "evidence": "Her second successor as CEO was Bob Reed.",
                "target_aliases": ["Bob Reed"],
            },
            {
                "source_title": "Bob Reed", "target_title": "Carol Reed",
                "relation": "spouse", "as_of": TARGET,
                "relative_clause": "the spouse of {source}",
                "evidence": "Bob Reed is married to Carol Reed.",
                "target_aliases": ["Carol Reed"],
            },
        ],
    }


def test_question_uses_short_ordered_steps_and_hides_entities():
    seed = MultiHopSeed.from_dict(_seed_dict())
    question = compose_relative_question(seed)
    assert "registered knowledge cutoff" in question
    assert "target snapshot" in question
    assert "second successor" in question and "spouse" in question
    assert "Alice Stone" not in question
    assert "Bob Reed" not in question
    assert "Carol Reed" not in question
    assert "after that" not in question.casefold()
    assert question.casefold().count("after") == 2
    assert "snapshot used in the previous step" not in question.casefold()
    assert "event identified in the previous step occurred" in question.casefold()
    assert len(question.split(". ")) == 4
    canonical = compose_canonical_question(seed)
    assert len(canonical.split()) > max(
        len(sentence.split()) for sentence in question.split(". ")
    )
    errors, chain = validate_chain(seed, FakeBackend())
    assert errors == []
    case = build_case(seed, chain, {"decision": "pending", "confidence": 0.0})
    assert case["_generation"]["canonical_question"] == canonical
    assert case["_generation"]["question_style"] == "explicit_event_anchored_steps_v3"


def test_public_anchor_substring_is_not_hidden_entity_leakage():
    value = _seed_dict()
    value["anchor_label"] = "Deputy Mayor of Example"
    value["hops"][0]["source_title"] = "Deputy Mayor of Example"
    value["hops"][1]["target_title"] = "Mayor of Example"
    value["hops"][2]["source_title"] = "Mayor of Example"
    seed = MultiHopSeed.from_dict(value)
    question = compose_relative_question(seed)
    assert "Deputy Mayor of Example" in question
    assert not any(
        "hidden entity name 'Mayor of Example'" in error
        for error in question_wording_errors(seed, question)
    )


def test_validator_rejects_entity_cycle_created_by_canonical_redirect():
    class RedirectCycleBackend(FakeBackend):
        def fetch_page(self, title, as_of=None):
            page = super().fetch_page(title, as_of)
            if title == "Example Corp":
                return _page(
                    "Bob Reed", page.revision_id, as_of, page.content,
                    [(link.target, link.anchor) for link in page.links],
                )
            return page

    errors, _ = validate_chain(
        MultiHopSeed.from_dict(_seed_dict()), RedirectCycleBackend(),
    )
    assert "fetched reasoning chain contains a canonical entity cycle" in errors


def test_question_writer_naturalizes_without_hidden_entities_or_oracle_dates():
    seed = MultiHopSeed.from_dict(_seed_dict())

    def fake_call(model, messages, temperature=0.0):
        private_prompt = messages[-1]["content"]
        assert '"private_source_entity": "Alice Stone"' in private_prompt
        assert '"private_target_entity": "Bob Reed"' in private_prompt
        assert '"private_target_entity": "Carol Reed"' in private_prompt
        assert '"as_of"' not in private_prompt
        return json.dumps({"steps": [
            "At the tested model's registered knowledge cutoff, who was Example Corp's chief executive officer?",
            "After the registered knowledge cutoff, who was that person's second successor as chief executive officer?",
            "At the target snapshot, after the event identified in the previous step occurred, who is that successor's spouse?",
        ]})

    generated = MultiHopQuestionWriter(
        "writer/model", call_model_fn=fake_call,
    ).write(seed)
    question = generated["question"]
    assert generated["model"] == "writer/model"
    assert generated["attempts"] == 1
    assert generated["private_entity_context_supplied"] is True
    assert "after the event identified in the previous step occurred" in question
    assert "snapshot used in the previous step" not in question
    assert "after that" not in question.casefold()
    assert "Alice Stone" not in question
    assert "Bob Reed" not in question
    assert "Carol Reed" not in question


def test_question_writer_rejects_a_leaked_canonical_entity_name():
    value = _seed_dict()
    value["hops"][1]["target_aliases"] = ["B. Reed"]
    seed = MultiHopSeed.from_dict(value)

    def leaking_call(model, messages, temperature=0.0):
        return json.dumps({"steps": [
            "At the tested model's registered knowledge cutoff, who was Example Corp's chief executive officer?",
            "After the registered knowledge cutoff, who was that person's second successor, Bob Reed?",
            "At the target snapshot, after the event identified in the previous step occurred, who is that successor's spouse?",
        ]})

    with pytest.raises(ValueError, match="hidden entity name 'Bob Reed'"):
        MultiHopQuestionWriter(
            "writer/model", call_model_fn=leaking_call,
        ).write(seed)


def test_question_writer_rejects_ambiguous_after_that_even_after_retry():
    seed = MultiHopSeed.from_dict(_seed_dict())

    def ambiguous_call(model, messages, temperature=0.0):
        return json.dumps({"steps": [
            "Start with Example Corp at the registered knowledge cutoff.",
            "Find its next leader after that.",
            "At the target snapshot, who is that person's spouse?",
        ]})

    with pytest.raises(ValueError, match="question writer failed gates"):
        MultiHopQuestionWriter(
            "writer/model", call_model_fn=ambiguous_call,
        ).write(seed)


def test_question_writer_rejects_a_solver_selected_snapshot_boundary():
    seed = MultiHopSeed.from_dict(_seed_dict())

    def browsing_dependent_call(model, messages, temperature=0.0):
        return json.dumps({"steps": [
            "At the registered knowledge cutoff, who was Example Corp's chief executive officer?",
            "After the registered knowledge cutoff, who was that person's second successor as chief executive officer?",
            "At the target snapshot, strictly later than the snapshot used in the previous step, who is that successor's spouse?",
        ]})

    with pytest.raises(ValueError, match="solver-selected snapshot"):
        MultiHopQuestionWriter(
            "writer/model", call_model_fn=browsing_dependent_call,
        ).write(seed)


def test_question_writer_rejects_an_unbound_tenure_pronoun():
    value = _seed_dict()
    value["hops"][1]["property_id"] = "P39"
    seed = MultiHopSeed.from_dict(value)

    def unbound_call(model, messages, temperature=0.0):
        return json.dumps({"steps": [
            "Who held Example Corp's chief executive office at the registered knowledge cutoff?",
            "After the registered knowledge cutoff, what position did that person first begin holding?",
            "At the target snapshot, who next began holding that position after its tenure began?",
        ]})

    with pytest.raises(ValueError, match="bind its event boundary"):
        MultiHopQuestionWriter(
            "writer/model", call_model_fn=unbound_call,
        ).write(seed)


def test_p39_fallback_requires_a_new_tenure_onset():
    value = _seed_dict()
    value["hops"][1]["property_id"] = "P39"
    value["hops"][1]["relative_clause"] = (
        "the next government position held by {source} after that"
    )
    value["hops"][1]["structured_evidence"] = {"direction": "forward"}
    question = compose_relative_question(MultiHopSeed.from_dict(value))
    assert "first government position" in question
    assert "began holding" in question
    assert "after that" not in question.casefold()


def test_p39_writer_accepts_interrogative_begin_holding_wording():
    value = _seed_dict()
    value["hops"][1]["property_id"] = "P39"
    value["hops"][1]["relative_clause"] = (
        "the next government position held by {source} after that"
    )
    value["hops"][1]["structured_evidence"] = {"direction": "forward"}
    seed = MultiHopSeed.from_dict(value)

    def fake_call(model, messages, temperature=0.0):
        return json.dumps({"steps": [
            "At the registered knowledge cutoff, who was Example Corp's chief executive officer?",
            "After the registered knowledge cutoff, which position did that person first begin holding?",
            "At the target snapshot, after the person identified two steps earlier began holding that position, who is that person's spouse?",
        ]})

    generated = MultiHopQuestionWriter(
        "writer/model", call_model_fn=fake_call,
    ).write(seed)
    assert "begin holding" in generated["question"]


def test_p39_scoped_relation_rejects_uncertified_first_claim():
    value = _seed_dict()
    value["hops"][1]["property_id"] = "P39"
    value["hops"][1]["structured_evidence"] = {
        "direction": "forward", "temporal_operator": "relation_after_boundary",
    }
    seed = MultiHopSeed.from_dict(value)

    def invented_first(model, messages, temperature=0.0):
        return json.dumps({"steps": [
            "Who led Example Corp at the registered knowledge cutoff?",
            "What was the first position that person began holding after the registered knowledge cutoff?",
            "At the target snapshot, who is the spouse after the event identified in the previous step occurred?",
        ]})

    with pytest.raises(ValueError, match="uncertified first/next"):
        MultiHopQuestionWriter(
            "writer/model", call_model_fn=invented_first,
        ).write(seed)


def test_p39_writer_rejects_a_redundant_tenure_clause():
    value = _seed_dict()
    value["hops"][1]["property_id"] = "P39"
    value["hops"][1]["relative_clause"] = (
        "the next government position held by {source} after that"
    )
    seed = MultiHopSeed.from_dict(value)

    def redundant_call(model, messages, temperature=0.0):
        return json.dumps({"steps": [
            "At the registered knowledge cutoff, who was Example Corp's chief executive officer?",
            "After the registered knowledge cutoff, what was the first position that person began holding, and had their tenure begun?",
            "At the target snapshot, after the tenure identified in the previous step began, who is that person's spouse?",
        ]})

    with pytest.raises(ValueError, match="redundantly restates"):
        MultiHopQuestionWriter(
            "writer/model", call_model_fn=redundant_call,
        ).write(seed)


def test_p39_writer_rejects_a_redundant_participial_tenure_clause():
    value = _seed_dict()
    value["hops"][1]["property_id"] = "P39"
    value["hops"][1]["structured_evidence"] = {"direction": "forward"}
    seed = MultiHopSeed.from_dict(value)

    def redundant_call(model, messages, temperature=0.0):
        return json.dumps({"steps": [
            "Who led Example Corp at the registered knowledge cutoff?",
            "After the registered knowledge cutoff, what position did that person first begin holding, with that tenure beginning?",
            "At the target snapshot, who is that person's spouse after the person identified two steps earlier began holding that position?",
        ]})

    with pytest.raises(ValueError, match="redundantly restates"):
        MultiHopQuestionWriter(
            "writer/model", call_model_fn=redundant_call,
        ).write(seed)


def test_question_writer_rejects_a_dangling_target_snapshot_fragment():
    seed = MultiHopSeed.from_dict(_seed_dict())

    def dangling_call(model, messages, temperature=0.0):
        return json.dumps({"steps": [
            "Who led Example Corp at the registered knowledge cutoff?",
            "After the registered knowledge cutoff, who was that person's second successor?",
            "Who is that successor's spouse, target snapshot?",
        ]})

    with pytest.raises(ValueError, match="attach the answer to the target snapshot"):
        MultiHopQuestionWriter(
            "writer/model", call_model_fn=dangling_call,
        ).write(seed)


def test_question_writer_rejects_non_question_step_fragments():
    seed = MultiHopSeed.from_dict(_seed_dict())

    def fragmented_call(model, messages, temperature=0.0):
        return json.dumps({"steps": [
            "Who led Example Corp at the registered knowledge cutoff,",
            "After the registered knowledge cutoff, who was that person's second successor,",
            "At the target snapshot, after the event identified in the previous step occurred, who is that person's spouse?",
        ]})

    with pytest.raises(ValueError, match="standalone question"):
        MultiHopQuestionWriter(
            "writer/model", call_model_fn=fragmented_call,
        ).write(seed)


def test_cutoff_registry_is_exact_and_cases_do_not_cross_models():
    record = get_model_cutoff("openai/gpt-4.1-mini")
    assert record.cutoff_date == CUTOFF
    assert len(record.source_commit) == 40
    with pytest.raises(ValueError, match="no pinned"):
        get_model_cutoff("provider/unknown-alias")
    case = {"knowledge_cutoff": {"model_ids": ["openai/gpt-4.1-mini"]}}
    assert model_matches_cutoff(case, "openai/gpt-4.1-mini")
    assert not model_matches_cutoff(case, "openai/gpt-4.1")


def test_chain_gate_verifies_revision_evidence_and_links():
    seed = MultiHopSeed.from_dict(_seed_dict())
    errors, chain = validate_chain(seed, FakeBackend())
    assert errors == []
    assert len(chain) == 3
    judged = {"decision": "pass", "confidence": 0.99}
    case = build_case(seed, chain, judged)
    assert case["reasoning_hop_count"] == 3
    assert case["expected_navigation_distance"] == 5
    assert case["required_temporal_switches"] == 2
    assert len(case["temporal_waypoints"]) == 6
    assert case["hide_pivot_title"] is True
    assert case["old_answer_keywords"] == []
    composed_probe = next(
        probe for probe in case["prior_knowledge_contract"]["probes"]
        if probe["role"] == "composed"
    )
    assert composed_probe["context_aliases"] == ["Alice Stone", "Bob Reed"]
    assert validate_case(case) == []


def test_chain_gate_accepts_exact_hyperlink_through_verified_redirect():
    backend = FakeBackend()
    backend.pages[("Alice Stone", CUTOFF)].title = "Alice Stone (executive)"
    backend.pages[("Alice Stone", MIDDLE)].title = "Alice Stone (executive)"

    errors, chain = validate_chain(MultiHopSeed.from_dict(_seed_dict()), backend)

    assert errors == []
    assert chain[0]["requested_target_title"] == "Alice Stone"
    assert chain[0]["target_title"] == "Alice Stone (executive)"
    assert chain[1]["requested_source_title"] == "Alice Stone"
    assert chain[1]["source_title"] == "Alice Stone (executive)"


def test_same_snapshot_attribute_tail_adds_hyperlink_without_fake_time_switch():
    value = _seed_dict()
    value["id"] = "example_with_birthplace_tail"
    value["answer_kind"] = "What"
    value["hops"].append({
        "source_title": "Carol Reed", "target_title": "Example City",
        "relation": "place of birth", "as_of": TARGET,
        "relative_clause": "the place of birth of {source}",
        "evidence": "Carol Reed was born in Example City.",
        "target_aliases": ["Example City"],
        "incoming_time_policy": "same_snapshot",
        "property_id": "P19", "relation_family": "geography",
        "structured_evidence": {
            "source": "wikidata_statement", "source_qid": "Q1",
            "target_qid": "Q2", "property_id": "P19", "selector": "single",
        },
    })
    seed = MultiHopSeed.from_dict(value)
    errors, chain = validate_chain(seed, AttributeBackend())
    assert errors == []
    case = build_case(seed, chain, {"decision": "pass", "confidence": 0.99})
    assert case["reasoning_hop_count"] == 4
    assert case["required_temporal_switches"] == 2
    assert case["expected_navigation_distance"] == 6
    assert len(case["temporal_waypoints"]) == 7
    assert case["temporal_waypoints"][-1]["title"] == "Example City"
    assert case["temporal_waypoints"][-1]["incoming_edge"] == "hyperlink"
    assert validate_case(case) == []
    report = validate_shortest_arena(seed, chain, AttributeBackend(), branch_cap=10)
    assert report["semantic_shortest_distance"] == 6
    trajectory = [
        {"action": "switch_snapshot", "from_title": "Example Corp",
         "to_title": "Example Corp", "from_revision_id": None, "revision_id": 1,
         "from_snapshot_token": None, "snapshot_token": CUTOFF,
         "navigation_step": 1, "result": "ok"},
        {"action": "follow_link", "from_title": "Example Corp",
         "to_title": "Alice Stone", "from_revision_id": 1, "revision_id": 2,
         "from_snapshot_token": CUTOFF, "snapshot_token": CUTOFF,
         "navigation_step": 2, "result": "ok"},
        {"action": "switch_snapshot", "from_title": "Alice Stone",
         "to_title": "Alice Stone", "from_revision_id": 2, "revision_id": 3,
         "from_snapshot_token": CUTOFF, "snapshot_token": MIDDLE,
         "navigation_step": 3, "result": "ok"},
        {"action": "follow_link", "from_title": "Alice Stone",
         "to_title": "Bob Reed", "from_revision_id": 3, "revision_id": 4,
         "from_snapshot_token": MIDDLE, "snapshot_token": MIDDLE,
         "navigation_step": 4, "result": "ok"},
        {"action": "switch_snapshot", "from_title": "Bob Reed",
         "to_title": "Bob Reed", "from_revision_id": 4, "revision_id": 5,
         "from_snapshot_token": MIDDLE, "snapshot_token": TARGET,
         "navigation_step": 5, "result": "ok"},
        {"action": "follow_link", "from_title": "Bob Reed",
         "to_title": "Carol Reed", "from_revision_id": 5, "revision_id": 6,
         "from_snapshot_token": TARGET, "snapshot_token": TARGET,
         "navigation_step": 6, "result": "ok"},
        {"action": "follow_link", "from_title": "Carol Reed",
         "to_title": "Example City", "from_revision_id": 6, "revision_id": 7,
         "from_snapshot_token": TARGET, "snapshot_token": TARGET,
         "navigation_step": 7, "result": "ok"},
    ]
    metrics = _semantic_waypoint_metrics({"trajectory": trajectory}, case)
    assert metrics["semantic_route_complete"] is True
    assert metrics["semantic_waypoints_completed"] == 7


def test_wikidata_next_edge_requires_event_order_certificate():
    value = _seed_dict()
    value["hops"][1]["property_id"] = "P39"
    value["hops"][1]["relative_clause"] = (
        "the next government position held by {source}"
    )
    value["hops"][1]["structured_evidence"] = {
        "source": "wikidata_time_qualified_statement",
        "direction": "forward", "temporal_operator": "next_after_boundary",
        "event_date": MIDDLE, "kg_subject_qid": "Q1",
        "kg_object_qid": "Q2",
    }
    errors, _ = validate_chain(MultiHopSeed.from_dict(value), FakeBackend())
    assert any("missing event_order_certificate" in error for error in errors)


def test_frozen_rendered_evidence_detects_content_tampering():
    seed = MultiHopSeed.from_dict(_seed_dict())
    errors, chain = validate_chain(seed, FakeBackend())
    assert errors == []
    case = build_case(seed, chain, {"decision": "pass", "confidence": 1.0})
    assert validate_case(case) == []
    case["frozen_wikipedia_evidence"]["pages"][0]["content"] += " tampered"
    errors = validate_case(case)
    assert any("content hash mismatch" in error for error in errors)


def test_invalidated_generation_v5_cannot_enter_runner_preflight():
    seed = MultiHopSeed.from_dict(_seed_dict())
    errors, chain = validate_chain(seed, FakeBackend())
    assert errors == []
    case = build_case(seed, chain, {"decision": "pass", "confidence": 1.0})
    case["_generation"]["schema_version"] = (
        "wikipedia-cutoff-relative-multihop-v5"
    )
    case["prior_knowledge_contract"]["schema_version"] = (
        "factorized-prior-knowledge-v1"
    )
    errors = validate_case(case)
    assert any("generation v5 is invalidated" in error for error in errors)


def test_attribute_tail_generator_requires_wikidata_and_revision_hyperlink():
    variants, packets = generate_attribute_tail_variants(
        _seed_dict(), backend=AttributeBackend(),
    )
    assert len(variants) == 1
    variant = variants[0]
    tail = variant["hops"][-1]
    assert tail["property_id"] == "P19"
    assert tail["relation_family"] == "geography"
    assert tail["incoming_time_policy"] == "same_snapshot"
    assert tail["structured_evidence"]["source_qid"] == "Q1"
    assert any(packet["status"] == "pass" for packet in packets)

    class RateLimitedBackend(AttributeBackend):
        def get_wikidata_entities(self, qids, props="claims|labels|sitelinks"):
            if qids == ["Q1"]:
                return super().get_wikidata_entities(qids, props=props)
            raise WikipediaError("Wikimedia API 呼叫失敗：HTTP 429")

    variants, packets = generate_attribute_tail_variants(
        _seed_dict(), backend=RateLimitedBackend(),
    )
    assert variants == []
    assert packets[0]["status"] == "infrastructure_error"


def test_claim_selectors_fail_closed_on_ambiguity_and_use_latest_qualifier():
    def claim(qid, *, point=None):
        value = {
            "rank": "normal",
            "mainsnak": {"datavalue": {"value": {"id": qid}}},
        }
        if point:
            value["qualifiers"] = {
                "P585": [{"datavalue": {"value": {"time": f"+{point}T00:00:00Z"}}}]
            }
        return value

    assert select_claim(
        [claim("Q1"), claim("Q2")], RELATION_BY_PROPERTY["P19"], TARGET
    ) is None
    selected = select_claim(
        [claim("Q1", point="2024-01-01"), claim("Q2", point="2025-01-01")],
        RELATION_BY_PROPERTY["P166"], TARGET,
    )
    assert selected is not None and selected.target_qid == "Q2"
    assert selected.qualifier_dates == {"P585": "2025-01-01"}
    assert select_claim(
        [claim("Q3")], RELATION_BY_PROPERTY["P54"], TARGET
    ) is None


def test_chain_gate_rejects_a_missing_real_hyperlink():
    backend = FakeBackend()
    backend.pages[("Bob Reed", TARGET)].links = []
    errors, _ = validate_chain(MultiHopSeed.from_dict(_seed_dict()), backend)
    assert any("no hyperlink" in error for error in errors)

    backend = FakeBackend()
    backend.pages[("Alice Stone", CUTOFF)].links.append(
        PageLink(target="Bob Reed", anchor="Bob Reed")
    )
    backend.pages[("Alice Stone", CUTOFF)].content += " Bob Reed"
    errors, _ = validate_chain(MultiHopSeed.from_dict(_seed_dict()), backend)
    assert any("before the required temporal switch" in error for error in errors)


def test_chain_gate_accepts_audited_semantic_contrast_for_old_unrelated_link():
    backend = FakeBackend()
    backend.pages[("Alice Stone", CUTOFF)].links.append(
        PageLink(target="Bob Reed", anchor="Bob Reed")
    )
    backend.pages[("Alice Stone", CUTOFF)].content += (
        " Earlier in her career, Alice Stone met [Bob Reed -> Bob Reed]."
    )
    value = _seed_dict()
    value["hops"][1]["property_id"] = "P169"
    value["hops"][1]["structured_evidence"] = {
        "prior_relation_contrast": {
            "method": "independent_llm_semantic_contrast",
            "decision": "pass", "confidence": 0.94,
            "judge_model": "independent/judge", "property_id": "P169",
            "direction": "forward", "before_revision_id": 2,
            "after_revision_id": 3,
            "before_evidence": "Earlier in her career, Alice Stone met Bob Reed.",
        },
    }
    errors, chain = validate_chain(MultiHopSeed.from_dict(value), backend)
    assert errors == []
    assert chain[1]["prior_snapshot_absence_verified"] is False
    assert chain[1]["prior_relation_contrast_verified"] is True


def test_seed_rejects_disconnected_or_cyclic_reasoning():
    value = _seed_dict()
    value["hops"][1]["source_title"] = "Somebody Else"
    with pytest.raises(ValueError, match="disconnected"):
        MultiHopSeed.from_dict(value)

    value = _seed_dict()
    value["hops"][2]["as_of"] = MIDDLE
    value["target_as_of"] = MIDDLE
    with pytest.raises(ValueError, match="must use a later snapshot"):
        MultiHopSeed.from_dict(value)


def test_independent_semantic_judge_confidence_gate():
    seed = MultiHopSeed.from_dict(_seed_dict())
    _, chain = validate_chain(seed, FakeBackend())

    def fake_call(model, messages, temperature=0.0):
        return '{"decision":"pass","confidence":0.92,"checks":{' \
               '"evidence_semantics":true,"relation_order_and_composition":true,' \
               '"cutoff_and_snapshot_anchoring":true,' \
               '"multi_hop_and_uniqueness":true,"no_entity_leakage":true,' \
               '"temporal_transition_clarity":true,' \
               '"natural_question_wording":true},' \
               '"reason":"supported","rejected_hops":[]}'

    result = MultiHopQuestionJudge("judge/model", call_model_fn=fake_call).judge(
        compose_relative_question(seed), chain
    )
    assert result["decision"] == "pass"

    def malformed_call(model, messages, temperature=0.0):
        return '{"decision":"pass","confidence":0.99,"checks":[],' \
               '"reason":"supported","rejected_hops":[]}'

    rejected = MultiHopQuestionJudge(
        "judge/model", call_model_fn=malformed_call,
    ).judge(compose_relative_question(seed), chain)
    assert rejected["decision"] == "reject"
    assert rejected["schema_gate_errors"] == ["checks_not_object"]


def test_semantic_judge_reuses_content_hash_cache(tmp_path):
    seed = MultiHopSeed.from_dict(_seed_dict())
    _, chain = validate_chain(seed, FakeBackend())
    calls = []

    def fake_call(model, messages, temperature=0.0):
        calls.append(messages)
        return '{"decision":"pass","confidence":0.92,"checks":{' \
               '"evidence_semantics":true,"relation_order_and_composition":true,' \
               '"cutoff_and_snapshot_anchoring":true,' \
               '"multi_hop_and_uniqueness":true,"no_entity_leakage":true,' \
               '"temporal_transition_clarity":true,' \
               '"natural_question_wording":true},' \
               '"reason":"supported","rejected_hops":[]}'

    cache_path = str(tmp_path / "judge.db")
    judge = MultiHopQuestionJudge(
        "judge/model", call_model_fn=fake_call, cache_path=cache_path
    )
    first = judge.judge(compose_relative_question(seed), chain)
    second = judge.judge(compose_relative_question(seed), chain)
    judge.close()

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert len(calls) == 1


def test_validated_prefix_only_fetches_the_new_attribute_tail():
    class CountingBackend(AttributeBackend):
        def __init__(self):
            super().__init__()
            self.fetches = []

        def fetch_page(self, title, as_of=None):
            self.fetches.append((title, as_of))
            return super().fetch_page(title, as_of=as_of)

    backend = CountingBackend()
    base = MultiHopSeed.from_dict(_seed_dict())
    errors, prefix = validate_chain(base, backend)
    assert errors == []
    before = len(backend.fetches)

    value = _seed_dict()
    value["id"] = "example_with_cached_prefix"
    value["answer_kind"] = "What"
    value["hops"].append({
        "source_title": "Carol Reed", "target_title": "Example City",
        "relation": "place of birth", "as_of": TARGET,
        "relative_clause": "the place of birth of {source}",
        "evidence": "Carol Reed was born in Example City.",
        "target_aliases": ["Example City"],
        "incoming_time_policy": "same_snapshot",
        "property_id": "P19", "relation_family": "geography",
        "structured_evidence": {
            "source": "wikidata_statement", "source_qid": "Q1",
            "target_qid": "Q2", "property_id": "P19", "selector": "single",
        },
    })
    errors, chain = validate_chain(
        MultiHopSeed.from_dict(value), backend, validated_prefix=prefix
    )
    assert errors == []
    assert len(chain) == 4
    assert backend.fetches[before:] == [
        ("Carol Reed", TARGET),
        ("Example City", TARGET),
    ]


def test_semantic_route_and_raw_shortest_are_separate_contracts():
    seed = MultiHopSeed.from_dict(_seed_dict())
    _, chain = validate_chain(seed, FakeBackend())
    case = build_case(seed, chain, {"decision": "pass", "confidence": 0.99})
    keys = {
        "corp": page_version_key("Example Corp", 1),
        "alice_old": page_version_key("Alice Stone", 2),
        "alice_new": page_version_key("Alice Stone", 3),
        "bob_old": page_version_key("Bob Reed", 4),
        "bob_new": page_version_key("Bob Reed", 5),
        "carol": page_version_key("Carol Reed", 6),
    }
    edges = [
        {"source": keys["corp"], "target": keys["alice_old"], "kind": "hyperlink"},
        {"source": keys["alice_old"], "target": keys["alice_new"], "kind": "temporal"},
        {"source": keys["alice_new"], "target": keys["bob_old"], "kind": "hyperlink"},
        {"source": keys["bob_old"], "target": keys["bob_new"], "kind": "temporal"},
        {"source": keys["bob_new"], "target": keys["carol"], "kind": "hyperlink"},
    ]
    navigation = {
        "arena_edges": edges, "target_key": keys["carol"],
        "distances": {keys["corp"]: 5},
    }
    contract = validate_chain_route(case, navigation)
    assert contract and contract["semantic_distance"] == 5
    assert contract["raw_distance"] == 5
    navigation["distances"][keys["corp"]] = 2
    contract = validate_chain_route(case, navigation)
    assert contract and contract["raw_distance"] == 2
    assert contract["raw_shortest_match"] is False
    with pytest.raises(ValueError, match="not shortest"):
        validate_chain_route(case, navigation, require_raw_shortest=True)


def test_generator_checks_shortest_arena_before_llm_judge():
    backend = FakeBackend()
    seed = MultiHopSeed.from_dict(_seed_dict())
    errors, chain = validate_chain(seed, backend)
    assert errors == []
    report = validate_shortest_arena(seed, chain, backend, branch_cap=10)
    assert report["passed"] is True
    assert report["semantic_shortest_distance"] == 5
    assert report["raw_shortest_distance"] == 5

    backend.pages[("Example Corp", CUTOFF)].links.append(
        PageLink(target="Carol Reed", anchor="shortcut")
    )
    backend.pages[("Carol Reed", CUTOFF)] = _page(
        "Carol Reed", 7, CUTOFF, "Earlier Carol Reed biography."
    )
    report = validate_shortest_arena(seed, chain, backend, branch_cap=10)
    assert report["raw_shortest_distance"] == 2
    assert report["semantic_shortest_distance"] == 5
    assert report["raw_shortest_matches_semantic"] is False


def test_branch_cap_cannot_evict_the_required_route_from_the_arena():
    backend = FakeBackend()
    backend.pages[("A Decoy", TARGET)] = _page(
        "A Decoy", 8, TARGET, "A decoy links to Carol Reed.",
        [("Carol Reed", "Carol Reed")],
    )
    seed = MultiHopSeed.from_dict(_seed_dict())
    errors, chain = validate_chain(seed, backend)
    assert errors == []

    # Alphabetical backlink truncation returns A Decoy before Bob Reed.  The
    # intended route must still be seeded into the bounded arena independently
    # of the cap, then checked against every real edge in that arena.
    report = validate_shortest_arena(seed, chain, backend, branch_cap=1)
    assert report["passed"] is True
    assert report["semantic_shortest_distance"] == 5
    assert report["raw_shortest_distance"] == 5


def test_arena_node_cap_preserves_required_route_and_discloses_truncation():
    backend = FakeBackend()
    backend.offline_only = True
    backend.pages[("A Decoy", TARGET)] = _page(
        "A Decoy", 8, TARGET, "A decoy links to Carol Reed.",
        [("Carol Reed", "Carol Reed")],
    )
    seed = MultiHopSeed.from_dict(_seed_dict())
    errors, chain = validate_chain(seed, backend)
    assert errors == []

    # The six required page-version waypoints consume the full arena budget.
    # Alternative discovery is therefore truncated, but the audited route is
    # retained and its exact distance inside that disclosed arena is unchanged.
    report = validate_shortest_arena(
        seed, chain, backend, branch_cap=10, node_cap=6,
    )
    assert report["passed"] is True
    assert report["semantic_shortest_distance"] == 5
    assert report["arena_node_count"] == report["arena_node_cap"] == 6
    assert report["arena_truncated"] is True
    assert report["arena_discovery_mode"] == "offline_cache"
    assert "uncached alternatives" in report["coverage_note"]
    assert "reached that cap" in report["coverage_note"]


def test_infrastructure_errors_are_not_semantic_rejections():
    assert has_infrastructure_error([
        "shortest_path: Wikimedia API 呼叫失敗：HTTP 429",
    ])
    assert not has_infrastructure_error([
        "hop 2: fetched chain is disconnected",
    ])


def test_semantic_progress_requires_exact_ordered_entity_time_edges():
    seed = MultiHopSeed.from_dict(_seed_dict())
    _, chain = validate_chain(seed, FakeBackend())
    case = build_case(seed, chain, {"decision": "pass", "confidence": 0.99})
    actions = [
        ("switch_snapshot", "Example Corp", None, 1),
        ("follow_link", "Alice Stone", 1, 2),
        ("switch_snapshot", "Alice Stone", 2, 3),
        ("follow_link", "Bob Reed", 3, 4),
        ("switch_snapshot", "Bob Reed", 4, 5),
        ("follow_link", "Carol Reed", 5, 6),
    ]
    trajectory = []
    from_title = "Example Corp"
    for navigation_step, (action, to_title, from_revision, revision) in enumerate(
        actions, start=1
    ):
        trajectory.append({
            "action": action, "from_title": from_title, "to_title": to_title,
            "from_revision_id": from_revision, "revision_id": revision,
            "navigation_step": navigation_step, "result": "ok",
        })
        from_title = to_title
    metrics = _semantic_waypoint_metrics({"trajectory": trajectory}, case)
    assert metrics["semantic_route_complete"] is True
    assert metrics["semantic_waypoints_completed"] == 6
    assert metrics["actual_required_temporal_switches"] == 2
    assert metrics["semantic_shortest_arrival"] is True

    shortcut = {"trajectory": [trajectory[0], trajectory[1], *trajectory[4:]]}
    metrics = _semantic_waypoint_metrics(shortcut, case)
    assert metrics["semantic_route_complete"] is False
    assert metrics["semantic_waypoints_completed"] == 2


def test_semantic_progress_accepts_model_selected_intermediate_date():
    seed = MultiHopSeed.from_dict(_seed_dict())
    _, chain = validate_chain(seed, FakeBackend())
    case = build_case(seed, chain, {"decision": "pass", "confidence": 0.99})
    trajectory = [
        {
            "action": "switch_snapshot", "from_title": "Example Corp",
            "to_title": "Example Corp", "from_revision_id": None,
            "revision_id": 1, "from_snapshot_token": None,
            "snapshot_token": CUTOFF, "navigation_step": 1, "result": "ok",
        },
        {
            "action": "follow_link", "from_title": "Example Corp",
            "to_title": "Alice Stone", "from_revision_id": 1,
            "revision_id": 2, "from_snapshot_token": CUTOFF,
            "snapshot_token": CUTOFF, "navigation_step": 2, "result": "ok",
        },
        {
            "action": "switch_snapshot", "from_title": "Alice Stone",
            "to_title": "Alice Stone", "from_revision_id": 2,
            "revision_id": 30, "from_snapshot_token": CUTOFF,
            "snapshot_token": "2025-05-17", "navigation_step": 3, "result": "ok",
        },
        {
            "action": "follow_link", "from_title": "Alice Stone",
            "to_title": "Bob Reed", "from_revision_id": 30,
            "revision_id": 40, "from_snapshot_token": "2025-05-17",
            "snapshot_token": "2025-05-17", "navigation_step": 4, "result": "ok",
        },
        {
            "action": "switch_snapshot", "from_title": "Bob Reed",
            "to_title": "Bob Reed", "from_revision_id": 40,
            "revision_id": 50, "from_snapshot_token": "2025-05-17",
            "snapshot_token": TARGET, "navigation_step": 5, "result": "ok",
        },
        {
            "action": "follow_link", "from_title": "Bob Reed",
            "to_title": "Carol Reed", "from_revision_id": 50,
            "revision_id": 60, "from_snapshot_token": TARGET,
            "snapshot_token": TARGET, "navigation_step": 6, "result": "ok",
        },
    ]
    metrics = _semantic_waypoint_metrics({"trajectory": trajectory}, case)
    assert metrics["semantic_route_complete"] is True
    assert metrics["semantic_waypoints_completed"] == 6
    assert metrics["reference_waypoint_revision_visits"] == 2

    # Jumping to target one entity too early leaves no strictly later date for
    # the required final temporal transition.
    premature = [dict(row) for row in trajectory]
    premature[2]["snapshot_token"] = TARGET
    premature[3]["from_snapshot_token"] = TARGET
    premature[3]["snapshot_token"] = TARGET
    metrics = _semantic_waypoint_metrics({"trajectory": premature}, case)
    assert metrics["semantic_route_complete"] is False
    assert metrics["semantic_waypoints_completed"] == 2


def test_p286_discovery_emits_a_generator_compatible_alternating_seed():
    candidate = P286Candidate(
        anchor_title="Example Corp", coach_title="Alice Stone",
        middle_title="Bob Reed", successor_title="Carol Reed",
        middle_start=MIDDLE, successor_start=TARGET,
    )
    seed, errors = verify_p286_candidate(
        candidate, model_id="openai/gpt-4.1-mini", backend=FakeBackend()
    )
    assert errors == []
    assert seed is not None
    parsed = MultiHopSeed.from_dict(seed)
    assert [hop.as_of for hop in parsed.hops] == [CUTOFF, MIDDLE, TARGET]
    assert "p:P286" in p286_query(CUTOFF, 25)
    assert "LIMIT 25" in p286_query(CUTOFF, 25)


def test_official_pageview_snapshot_ranks_candidate_entities():
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"items": [{"articles": [
                {"article": "Example_Corp", "views": 9000, "rank": 3},
                {"article": "Alice_Stone", "views": 4000, "rank": 9},
                {"article": "Main_Page", "views": 999999, "rank": 1},
            ]}]}

    calls = []

    def request_get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    snapshot = fetch_top_pageviews(
        lang="en", month="2026-07", request_get=request_get,
    )
    candidate = P286Candidate(
        anchor_title="Example Corp", coach_title="Alice Stone",
        middle_title="Bob Reed", successor_title="Carol Reed",
        middle_start=MIDDLE, successor_start=TARGET,
    )
    record = candidate_popularity(candidate, snapshot)
    assert record["score"] == 13000
    assert snapshot.views("Main Page") == 0
    assert "/2026/07/all-days" in calls[0][0]
    assert calls[0][1]["headers"]["User-Agent"]
    assert last_complete_month(__import__("datetime").date(2026, 8, 12)) == "2026-07"


def test_discovery_stops_when_diversity_quota_is_filled(tmp_path, monkeypatch):
    from tkg.experiment import temporal_candidate_discovery as discovery

    candidates = [
        P286Candidate(
            anchor_title=f"Anchor {index}", coach_title=f"Coach {index}",
            middle_title=f"Middle {index}", successor_title=f"Answer {index}",
            middle_start=MIDDLE, successor_start=TARGET,
        )
        for index in range(3)
    ]
    checked = []

    class Backend:
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

    def fake_verify(candidate, *, model_id, backend):
        checked.append(candidate.anchor_title)
        return {"id": candidate.anchor_title, "hops": []}, []

    def fake_tails(seed, *, backend, popularity, relations):
        assert all(spec.guessability != "easy" for spec in relations)
        return [{
            "id": f"{seed['id']}-P19",
            "selection_metadata": {
                "relation_family": "geography", "property_id": "P19",
                "source_entity": seed["id"], "source_pageviews": 1,
                "target_pageviews": 1,
            },
        }], [{"status": "pass"}]

    monkeypatch.setattr(
        discovery, "discover_p286_candidates", lambda cutoff, limit: candidates
    )
    monkeypatch.setattr(discovery, "WikipediaPageBackend", Backend)
    monkeypatch.setattr(discovery, "verify_p286_candidate", fake_verify)
    monkeypatch.setattr(discovery, "generate_attribute_tail_variants", fake_tails)
    output = tmp_path / "seeds.json"
    packets = tmp_path / "packets.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "tkg-discover-temporal-candidates",
        "--model-id", "openai/gpt-4.1-mini",
        "--max-accepted", "1",
        "--popular-prefilter", "3",
        "--no-popularity-ranking",
        "--output", str(output),
        "--packets-output", str(packets),
    ])

    assert discovery.main() == 0
    assert checked == ["Anchor 0"]
    assert len(json.loads(output.read_text())["seeds"]) == 1
    assert len(packets.read_text().splitlines()) == 1
    assert attribute_quota_capacity(
        20, max_per_family=4, max_per_property=1
    ) == 14


def test_diversity_sampler_caps_relation_family_and_source_entity():
    snapshot = PopularitySnapshot(
        project="en.wikipedia.org", month="2026-07",
        records={"popular": PopularityRecord("Popular", 10000, 1)},
    )
    assert snapshot.views("Popular") == 10000

    def variant(identifier, family, prop, source, views):
        return {
            "id": identifier,
            "selection_metadata": {
                "relation_family": family, "property_id": prop,
                "source_entity": source, "source_pageviews": views,
                "target_pageviews": 0,
            },
        }

    values = [
        variant("career-1", "career", "P108", "Popular", 10000),
        variant("career-2", "career", "P39", "Popular", 9000),
        variant("geo-1", "geography", "P19", "Popular", 8000),
        variant("edu-1", "education", "P69", "Another", 7000),
    ]
    selected = select_diverse_variants(
        values, max_total=4, max_per_family=1, max_per_source=2,
    )
    families = [value["selection_metadata"]["relation_family"] for value in selected]
    assert set(families) == {"career", "geography", "education"}
    assert len(selected) == 3
