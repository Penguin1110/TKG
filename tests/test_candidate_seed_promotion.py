from types import SimpleNamespace

from tkg.experiment.candidate_seed_promotion import (
    _visible_target_excerpt,
    evidence_block,
    event_order_query,
    normalize_order_rows,
    renew_tail_from_wikipedia,
    select_candidates,
)


def test_prior_visibility_detects_alias_when_canonical_title_is_absent():
    page = SimpleNamespace(links=[], content="The club later hired Las Palmas staff.")
    excerpt = _visible_target_excerpt(page, "UD Las Palmas", ["Las Palmas"])
    assert excerpt is not None
    assert "Las Palmas" in excerpt


def test_event_order_query_keeps_direction_explicit():
    inverse = event_order_query(
        property_id="P1308", source_qid="Q1", direction="inverse",
        boundary="2024-06-01", coverage_end="2026-08-15",
    )
    forward = event_order_query(
        property_id="P1308", source_qid="Q2", direction="forward",
        boundary="2025-01-01", coverage_end="2026-08-15",
    )
    assert "?target p:P1308 ?statement" in inverse
    assert "ps:P1308 wd:Q1" in inverse
    assert "wd:Q2 p:P1308 ?statement" in forward
    assert "ps:P1308 ?target" in forward


def test_normalize_order_rows_deduplicates_and_sorts():
    rows = [
        {"target": {"value": "http://www.wikidata.org/entity/Q3"},
         "start": {"value": "2025-02-01T00:00:00Z"}},
        {"target": {"value": "http://www.wikidata.org/entity/Q2"},
         "start": {"value": "2025-01-01T00:00:00Z"}},
        {"target": {"value": "http://www.wikidata.org/entity/Q2"},
         "start": {"value": "2025-01-01T00:00:00Z"}},
    ]
    assert normalize_order_rows(rows) == [
        {"event_date": "2025-01-01", "target_qid": "Q2"},
        {"event_date": "2025-02-01", "target_qid": "Q3"},
    ]


def test_evidence_selection_prefers_semantics_over_short_unrelated_block():
    page = SimpleNamespace(
        links=[SimpleNamespace(target="Anke Rehlinger", anchor="Anke Rehlinger")],
        content=(
            "[Anke Rehlinger -> Anke Rehlinger] (born 1976)\n\n"
            "The current officeholder is [Anke Rehlinger -> Anke Rehlinger]."
        ),
    )
    block, aliases = evidence_block(
        page, "Anke Rehlinger", preferred_terms=("current", "officeholder"),
    )
    assert block == "The current officeholder is [Anke Rehlinger -> Anke Rehlinger]."
    assert aliases == ["Anke Rehlinger"]


def test_evidence_selection_penalizes_large_navigation_template():
    page = SimpleNamespace(
        links=[SimpleNamespace(
            target="President of the German Bundesrat", anchor="President of the Bundesrat",
        )],
        content=(
            "A politician serving as [President of the Bundesrat -> President of the German Bundesrat].\n\n"
            + "vte current presidents " * 300
            + "[President of the Bundesrat -> President of the German Bundesrat]"
        ),
    )
    block, _ = evidence_block(
        page, "President of the German Bundesrat",
        preferred_terms=("serving as", "president", "current"),
    )
    assert block is not None and block.startswith("A politician serving as")


def test_candidate_selection_uses_only_active_direct_relations():
    def row(identifier: str, prop: str, tail: str = "P19"):
        return {
            "id": identifier, "relation_property": prop,
            "topology_id": f"same-relation-forward-{prop}",
            "tail_property": tail, "public_anchor": identifier,
        }

    selected = select_candidates(
        [row("a", "P1308"), row("b", "P488"), row("c", "P1308", "P999")],
        {"P1308"}, 10,
    )
    assert [item["id"] for item in selected] == ["a"]


def test_tail_renewal_intersects_claims_with_revision_links():
    class Backend:
        def fetch_page(self, title, as_of=None):
            return SimpleNamespace(
                revision_id=7,
                links=[SimpleNamespace(target="Example City", anchor="Example City")],
            )

        def get_wikidata_entities(self, qids, props="claims|labels|sitelinks"):
            entities = {
                "Q1": {"claims": {"P19": [{
                    "rank": "normal",
                    "mainsnak": {"datavalue": {"value": {"id": "Q2"}}},
                }]}},
                "Q2": {"sitelinks": {"enwiki": {"title": "Example City"}}},
            }
            return {qid: entities[qid] for qid in qids if qid in entities}

    candidate = {
        "target_as_of": "2026-01-01", "tail_family": "career",
        "private_chain": [{}, {}, {}, {
            "source_qid": "Q1", "source_title": "Example Person",
            "target_qid": "Q3", "target_title": "Politician",
            "property_id": "P106",
        }],
    }
    renewed, audit = renew_tail_from_wikipedia(candidate, Backend())
    assert renewed["private_chain"][-1] == {
        "source_qid": "Q1", "source_title": "Example Person",
        "target_qid": "Q2", "target_title": "Example City",
        "property_id": "P19",
    }
    assert renewed["tail_family"] == "geography"
    assert audit is not None and audit["source_revision_id"] == 7
