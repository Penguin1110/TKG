"""Offline tests for cutoff-relative multi-hop question contracts."""

from __future__ import annotations

import pytest

from tkg.experiment.case_validation import validate_case, validate_chain_route
from tkg.experiment.multihop_generation import (
    MultiHopQuestionJudge, MultiHopSeed, build_case, compose_relative_question,
    validate_chain, validate_shortest_arena,
)
from tkg.experiment.model_cutoffs import get_model_cutoff, model_matches_cutoff
from tkg.experiment.contracts import PageLink, PageSnapshot
from tkg.wikipedia.snapshot import page_version_key
from tkg.wikipedia.backend import WikipediaError


CUTOFF = "2024-06-01"
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
            ("Alice Stone", TARGET): _page(
                "Alice Stone", 3, TARGET,
                "Her second successor as CEO was [Bob Reed -> Bob Reed].",
                [("Bob Reed", "Bob Reed")],
            ),
            ("Bob Reed", TARGET): _page(
                "Bob Reed", 4, TARGET,
                "Bob Reed is married to [Carol Reed -> Carol Reed].",
                [("Carol Reed", "Carol Reed")],
            ),
            ("Carol Reed", TARGET): _page(
                "Carol Reed", 5, TARGET, "Carol Reed biography.",
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


def _seed_dict():
    return {
        "id": "example_ceo_relative_chain",
        "model_id": "openai/gpt-4.1-mini",
        "target_as_of": TARGET,
        "anchor_label": "Example Corp",
        "category": "corporate",
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
                "relation": "second successor as chief executive officer", "as_of": TARGET,
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


def test_question_is_relative_nested_and_hides_entities():
    seed = MultiHopSeed.from_dict(_seed_dict())
    question = compose_relative_question(seed)
    assert "registered knowledge cutoff" in question
    assert "target snapshot" in question
    assert "second successor" in question and "spouse" in question
    assert "Alice Stone" not in question
    assert "Bob Reed" not in question
    assert "Carol Reed" not in question


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
    assert case["expected_navigation_distance"] == 4
    assert case["hide_pivot_title"] is True
    assert case["old_answer_keywords"] == []
    assert validate_case(case) == []


def test_chain_gate_rejects_a_missing_real_hyperlink():
    backend = FakeBackend()
    backend.pages[("Bob Reed", TARGET)].links = []
    errors, _ = validate_chain(MultiHopSeed.from_dict(_seed_dict()), backend)
    assert any("no hyperlink" in error for error in errors)


def test_seed_rejects_disconnected_or_cyclic_reasoning():
    value = _seed_dict()
    value["hops"][1]["source_title"] = "Somebody Else"
    with pytest.raises(ValueError, match="disconnected"):
        MultiHopSeed.from_dict(value)


def test_independent_semantic_judge_confidence_gate():
    seed = MultiHopSeed.from_dict(_seed_dict())
    _, chain = validate_chain(seed, FakeBackend())

    def fake_call(model, messages, temperature=0.0):
        return '{"decision":"pass","confidence":0.92,"checks":{},' \
               '"reason":"supported","rejected_hops":[]}'

    result = MultiHopQuestionJudge("judge/model", call_model_fn=fake_call).judge(
        compose_relative_question(seed), chain
    )
    assert result["decision"] == "pass"


def test_declared_chain_must_be_the_arena_shortest_path():
    seed = MultiHopSeed.from_dict(_seed_dict())
    _, chain = validate_chain(seed, FakeBackend())
    case = build_case(seed, chain, {"decision": "pass", "confidence": 0.99})
    keys = {
        "corp": page_version_key("Example Corp", 1),
        "alice_old": page_version_key("Alice Stone", 2),
        "alice_new": page_version_key("Alice Stone", 3),
        "bob": page_version_key("Bob Reed", 4),
        "carol": page_version_key("Carol Reed", 5),
    }
    edges = [
        {"source": keys["corp"], "target": keys["alice_old"], "kind": "hyperlink"},
        {"source": keys["alice_old"], "target": keys["alice_new"], "kind": "temporal"},
        {"source": keys["alice_new"], "target": keys["bob"], "kind": "hyperlink"},
        {"source": keys["bob"], "target": keys["carol"], "kind": "hyperlink"},
    ]
    navigation = {
        "arena_edges": edges, "target_key": keys["carol"],
        "distances": {keys["corp"]: 4},
    }
    contract = validate_chain_route(case, navigation)
    assert contract and contract["distance"] == 4
    navigation["distances"][keys["corp"]] = 2
    with pytest.raises(ValueError, match="not shortest"):
        validate_chain_route(case, navigation)


def test_generator_checks_shortest_arena_before_llm_judge():
    backend = FakeBackend()
    seed = MultiHopSeed.from_dict(_seed_dict())
    errors, chain = validate_chain(seed, backend)
    assert errors == []
    report = validate_shortest_arena(seed, chain, backend, branch_cap=10)
    assert report["passed"] is True
    assert report["shortest_distance"] == 4

    backend.pages[("Example Corp", CUTOFF)].links.append(
        PageLink(target="Carol Reed", anchor="shortcut")
    )
    backend.pages[("Carol Reed", CUTOFF)] = _page(
        "Carol Reed", 6, CUTOFF, "Earlier Carol Reed biography."
    )
    with pytest.raises(ValueError, match="not shortest"):
        validate_shortest_arena(seed, chain, backend, branch_cap=10)
