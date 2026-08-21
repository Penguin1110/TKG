from types import SimpleNamespace

from tkg.experiment.certified_spine_tail_expansion_v27 import (
    expand_tail, synthesize_linked_tails,
)


class Backend:
    def fetch_page(self, title, as_of):
        return SimpleNamespace(
            title=title, revision_id=7, timestamp=as_of,
            content="Born in [Harbor City -> Harbor City].",
            links=[SimpleNamespace(target="Harbor City", anchor="Harbor City")],
        )

    def get_wikidata_entities(self, qids, props=None):
        rows = {
            "Q1": {"claims": {"P19": [{"mainsnak": {"datavalue": {"value": {"id": "Q2"}}}}]}},
            "Q2": {"sitelinks": {"enwiki": {"title": "Harbor City"}}},
        }
        return {qid: rows[qid] for qid in qids}


def test_tail_expansion_reuses_three_hop_certificate() -> None:
    certificate = {
        "model_id": "model",
        "hops": [{"hop": 0}, {"hop": 1}, {"hop": 2}],
        "selection_metadata": {"source_candidate_id": "kgcand_base"},
    }
    candidate = {
        "id": "kgcand_tail", "knowledge_cutoff": "2024-06-01",
        "target_as_of": "2026-01-01", "public_anchor": "Anchor",
        "domain_family": "leadership", "tail_family": "geography",
        "private_chain": [{}, {}, {}, {
            "source_qid": "Q1", "source_title": "Person",
            "target_qid": "Q2", "target_title": "Harbor City",
            "property_id": "P19",
        }],
    }
    seed, packet = expand_tail(candidate, certificate, Backend())
    assert seed is not None
    assert seed["hops"][:3] == certificate["hops"]
    assert seed["hops"][3]["target_title"] == "Harbor City"
    assert packet["bridge_review_reused"] is True
    assert packet["status"] == "promoted_pending_v6_validation"


def test_synthesized_tail_requires_kg_claim_and_exact_revision_link() -> None:
    certificate = {
        "id": "base", "model_id": "model", "cutoff_date": "2024-06-01",
        "target_as_of": "2026-01-01", "anchor_label": "Anchor",
        "category": "leadership",
        "hops": [{"hop": 0}, {"hop": 1}, {
            "target_title": "Person",
            "structured_evidence": {"kg_object_qid": "Q1"},
        }],
    }
    seeds, packets = synthesize_linked_tails(certificate, Backend())
    assert len(seeds) == 1
    assert seeds[0]["hops"][3]["property_id"] == "P19"
    assert seeds[0]["hops"][3]["target_title"] == "Harbor City"
    assert packets[0]["status"] == "promoted_pending_v6_validation"


def test_synthesized_tail_rejects_entity_cycle() -> None:
    certificate = {
        "id": "base", "model_id": "model", "cutoff_date": "2024-06-01",
        "target_as_of": "2026-01-01", "anchor_label": "Harbor City",
        "category": "leadership",
        "hops": [
            {"source_title": "Harbor City", "target_title": "A"},
            {"target_title": "B"},
            {"target_title": "Person", "structured_evidence": {"kg_object_qid": "Q1"}},
        ],
    }
    seeds, packets = synthesize_linked_tails(certificate, Backend())
    assert seeds == []
    assert packets[0]["errors"] == ["tail would create an entity cycle"]
