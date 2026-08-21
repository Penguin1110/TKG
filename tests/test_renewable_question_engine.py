"""Offline end-to-end tests for the renewable temporal question engine."""

from __future__ import annotations

import json

from tkg.experiment.contracts import PageLink, PageSnapshot
from tkg.experiment.multihop_generation import MultiHopSeed, compose_relative_question
from tkg.experiment.question_ledger import QuestionLedger, question_fingerprint
from tkg.experiment.relation_catalog import RELATION_BY_PROPERTY
from tkg.experiment.renewable_question_engine import (
    EdgeCandidate,
    RenewableQuestionEngine,
    TemporalEdgeContrastJudge,
    WikipediaSupport,
    admit_seeds,
    inverse_cutoff_anchor_query,
    inverse_transition_query,
    load_relation_admissions,
    _manifest_contract_sha256,
    query_inverse_candidates,
    select_claim_at,
    select_diverse_seeds,
)
from tkg.experiment.temporal_relation_profiler import PROFILE_SCHEMA
from tkg.experiment.temporal_relation_renewal import renew_registry
from tkg.experiment.temporal_relation_registry import load_temporal_relation_registry


CUTOFF = "2024-06-01"
T1 = "2025-01-01"
T2 = "2025-06-01"
T3 = "2026-01-01"


def _claim(target, *, start=None, end=None, point=None, preferred=False):
    qualifiers = {}
    for property_id, value in (("P580", start), ("P582", end), ("P585", point)):
        if value:
            qualifiers[property_id] = [{
                "datavalue": {"value": {
                    "time": f"+{value}T00:00:00Z", "precision": 11,
                }},
            }]
    return {
        "rank": "preferred" if preferred else "normal",
        "mainsnak": {"datavalue": {"value": {"id": target}}},
        "qualifiers": qualifiers,
    }


def _entity(title, claims=None):
    return {
        "claims": claims or {},
        "sitelinks": {"enwiki": {"title": title}},
    }


def _page(title, revision, as_of, content, links=()):
    return PageSnapshot(
        title=title, page_id=revision, revision_id=revision,
        timestamp=f"{as_of}T12:00:00Z", as_of=as_of, content=content,
        links=[PageLink(target=target, anchor=anchor) for target, anchor in links],
        source_url=f"https://example.invalid/?oldid={revision}",
    )


class FakeBackend:
    lang = "en"

    def __init__(self):
        self.entities = {
            "Q1": _entity("Alpha Corp", {"P169": [_claim("Q2")]}),
            "Q2": _entity("Alice Stone", {"P26": [_claim("Q3", start=T1)]}),
            "Q3": _entity("Bob Reed", {"P108": [_claim("Q4", start=T2)]}),
            "Q4": _entity("Beta Group", {"P488": [_claim("Q5", start=T3)]}),
            "Q5": _entity("Carol Jones"),
        }
        self.pages = {
            ("Alpha Corp", CUTOFF): _page(
                "Alpha Corp", 1, CUTOFF,
                "Chief executive officer [Alice Stone -> Alice Stone]",
                [("Alice Stone", "Alice Stone")],
            ),
            ("Alice Stone", CUTOFF): _page(
                "Alice Stone", 2, CUTOFF, "Alice Stone was not married."
            ),
            ("Alice Stone", T1): _page(
                "Alice Stone", 3, T1,
                "Spouse [Bob Reed -> Bob Reed]",
                [("Bob Reed", "Bob Reed")],
            ),
            ("Bob Reed", T1): _page(
                "Bob Reed", 4, T1, "Bob Reed worked independently."
            ),
            ("Bob Reed", T2): _page(
                "Bob Reed", 5, T2,
                "Employer [Beta Group -> Beta Group]",
                [("Beta Group", "Beta Group")],
            ),
            ("Beta Group", T2): _page(
                "Beta Group", 6, T2, "Beta Group had another chairperson."
            ),
            ("Beta Group", T3): _page(
                "Beta Group", 7, T3,
                "Chairperson [Carol Jones -> Carol Jones]",
                [("Carol Jones", "Carol Jones")],
            ),
            ("Carol Jones", T3): _page(
                "Carol Jones", 8, T3, "Carol Jones is an executive."
            ),
        }

    def get_wikidata_entities(self, qids, props="claims|labels|sitelinks"):
        return {qid: self.entities[qid] for qid in qids if qid in self.entities}

    def fetch_page(self, title, as_of=None):
        return self.pages[(title, as_of)]


def _profile(tmp_path):
    registry = load_temporal_relation_registry()
    wanted = {"P169", "P26", "P108", "P488"}
    profiles = []
    for spec in registry.relations:
        if spec.property_id not in wanted:
            continue
        profiles.append({
            "relation": spec.to_dict(),
            "query_sha256": spec.property_id.lower().ljust(64, "0"),
            "query_error": None,
            "metrics": {
                "kg_sampled": 3, "semantic_checked": 2,
                "semantic_supported": 2, "semantic_support_rate": 1.0,
            },
            "recommendation": "semantic_validated_candidate_human_review_required",
            "samples": ([{
                "property_id": "P169", "source_qid": "Q1",
                "source_title": "Alpha Corp", "target_qid": "Q9",
                "target_title": "Future CEO", "start": "2026-06-01",
                "end": None, "point": None, "event_date": "2026-06-01",
            }] if spec.property_id == "P169" else []),
            "wikipedia_checks": [{
                "property_id": spec.property_id, "source_qid": f"Q{index + 10}",
                "source_title": f"Source {index}", "status": "link_supported",
                "semantic_judge": {
                    "decision": "pass", "confidence": 0.95,
                    "expressed_relation": spec.label,
                    "checks": {
                        "direction": True, "relation_semantics": True,
                        "temporal_support": True,
                        "not_mere_cooccurrence": True,
                    },
                    "reason": "explicit relation field",
                },
            } for index in range(2)],
        })
    payload = {
        "schema_version": PROFILE_SCHEMA,
        "created_at": "2026-08-12T00:00:00+00:00",
        "registry": {
            "schema_version": registry.schema_version,
            "registry_version": registry.registry_version,
            "selected_relation_count": len(profiles),
        },
        "window": {"since": CUTOFF, "until": "2026-08-12"},
        "profiles": profiles,
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return registry, path


def test_registry_profiles_drive_a_strict_four_hop_temporal_beam(tmp_path):
    registry, profile_path = _profile(tmp_path)
    admissions = load_relation_admissions([profile_path], registry)
    assert {value.spec.property_id for value in admissions} == {
        "P169", "P26", "P108", "P488",
    }
    engine = RenewableQuestionEngine(
        model_id="openai/gpt-4.1-mini", until=T3,
        registry=registry, admissions=admissions, backend=FakeBackend(),
        inverse_lookup_fn=lambda *args, **kwargs: [],
        cutoff_anchor_lookup_fn=lambda *args, **kwargs: [],
        terminal_attribute_tails=False,
        min_hops=4, max_hops=4, min_families=3,
        beam_width=8, max_anchors=8,
    )
    seeds = engine.discover()
    assert len(seeds) == 1
    seed = MultiHopSeed.from_dict(seeds[0])
    assert [hop.property_id for hop in seed.hops] == ["P169", "P26", "P108", "P488"]
    assert [hop.as_of for hop in seed.hops] == [CUTOFF, T1, T2, T3]
    assert len({hop.relation_family for hop in seed.hops}) == 3
    assert len(set(seeds[0]["selection_metadata"]["semantic_identity"]["entity_qids"])) == 5
    question = compose_relative_question(seed)
    assert "registered knowledge cutoff" in question
    for hidden in ("Alice Stone", "Bob Reed", "Beta Group", "Carol Jones"):
        assert hidden not in question
    assert any(row["stage"] == "expand" and row["status"] == "pass" for row in engine.packets)


def test_claim_selector_and_inverse_query_are_direction_explicit():
    registry = load_temporal_relation_registry()
    by_property = {spec.property_id: spec for spec in registry.relations}
    assert select_claim_at(
        [_claim("Q1"), _claim("Q2")], by_property["P169"], CUTOFF
    ) is None
    selected = select_claim_at(
        [_claim("Q1", point="2023-01-01"), _claim("Q2", point="2024-01-01")],
        by_property["P166"], CUTOFF,
    )
    assert selected is not None and selected.target_qid == "Q2"
    historical = select_claim_at(
        [
            _claim("Q1", start="2023-01-01", end="2025-01-01"),
            _claim("Q2", start="2026-01-01", preferred=True),
        ],
        by_property["P169"], CUTOFF,
    )
    assert historical is not None and historical.target_qid == "Q1"

    query = inverse_transition_query(
        "Q2", [by_property["P286"]], after=CUTOFF, until=T3, limit=20
    )
    assert "(wd:P286 p:P286 ps:P286)" in query
    assert "ps:P286 wd:Q2" not in query  # statementProperty stays a bound variable
    assert "wd:Q2" in query and "LIMIT 20" in query
    anchor_query = inverse_cutoff_anchor_query(
        ["Q2", "Q3"], by_property["P286"], cutoff=CUTOFF, limit=30
    )
    assert "VALUES ?target { wd:Q2 wd:Q3 }" in anchor_query
    assert "pq:P582" in anchor_query and "LIMIT 30" in anchor_query

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": {"bindings": [{
                "property": {"value": "http://www.wikidata.org/entity/P286"},
                "source": {"value": "http://www.wikidata.org/entity/Q10"},
                "sourceArticle": {"value": "https://en.wikipedia.org/wiki/New_Team"},
                "start": {"value": "2025-01-01T00:00:00Z"},
            }]}}

    candidates = query_inverse_candidates(
        "Q2", (by_property["P286"],), after=CUTOFF, until=T3, limit=20,
        request_get=lambda *args, **kwargs: Response(),
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.spec == by_property["P286"]
    assert candidate.direction == "inverse" and candidate.next_qid == "Q10"
    assert candidate.next_title == "New Team" and candidate.event_date == T1
    assert candidate.qualifier_dates == {"P580": T1}
    assert candidate.kg_subject_qid == "Q10" and candidate.kg_object_qid == "Q2"
    certificate = candidate.event_order_certificate
    assert certificate is not None
    assert certificate["boundary_event_date"] == CUTOFF
    assert certificate["selected_event_date"] == T1
    assert certificate["selected_target_qid"] == "Q10"
    assert certificate["complete"] is True


def test_evidence_window_anchors_on_complete_visible_hyperlink_token():
    from tkg.experiment.renewable_question_engine import _evidence_window

    page = _page(
        "Coach", 99, T1,
        "History [CSKA Sofia -> PFC CSKA Sofia]\n\n"
        "Current team [PFC CSKA Sofia -> PFC CSKA Sofia] (manager)",
        [("PFC CSKA Sofia", "CSKA Sofia"),
         ("PFC CSKA Sofia", "PFC CSKA Sofia")],
    )
    evidence = _evidence_window(page, "PFC CSKA Sofia")
    assert evidence is not None
    excerpt, alias = evidence
    assert alias == "PFC CSKA Sofia" and alias in excerpt


def test_question_ledger_has_stable_identity_and_retry_semantics(tmp_path):
    identity = {
        "model_id": "openai/gpt-4.1-mini", "cutoff_date": CUTOFF,
        "target_as_of": T3, "entity_qids": ["Q1", "Q2"],
        "property_steps": [{"property_id": "P169", "direction": "forward"}],
        "snapshot_dates": [CUTOFF], "answer_qid": "Q2",
    }
    fingerprint = question_fingerprint(**identity)
    ledger = QuestionLedger(tmp_path / "ledger.db")
    ledger.record(
        fingerprint=fingerprint, seed_id="seed", model_id=identity["model_id"],
        cutoff_date=CUTOFF, target_as_of=T3, answer_qid="Q2", status="pending",
        semantic_identity=identity, seed={"wording": "A"}, audit={},
    )
    assert ledger.excludes(fingerprint) is False
    ledger.record(
        fingerprint=fingerprint, seed_id="seed", model_id=identity["model_id"],
        cutoff_date=CUTOFF, target_as_of=T3, answer_qid="Q2",
        status="judge_reject", semantic_identity=identity,
        seed={"wording": "B"}, audit={"reason": "ambiguous"},
    )
    assert ledger.excludes(fingerprint) is True
    assert ledger.excludes(fingerprint, retry_rejected=True) is False
    assert ledger.get(fingerprint)["attempts"] == 2
    ledger.close()


def test_renewal_contract_hash_excludes_observed_yield():
    manifest = {
        "schema_version": "engine-v1", "model_id": "tested/model",
        "cutoff_date": CUTOFF, "until": T3,
        "registry_schema": "registry-v1", "registry_version": "r1",
        "registry_sha256": "a" * 64, "profile_sha256s": ["b" * 64],
        "admitted_properties": ["P169"], "config": {"beam_width": 8},
        "counts": {"discovered": 1},
    }
    first = _manifest_contract_sha256(manifest)
    manifest["counts"] = {"discovered": 999, "machine_pass": 7}
    assert _manifest_contract_sha256(manifest) == first
    manifest["config"] = {"beam_width": 9}
    assert _manifest_contract_sha256(manifest) != first


def test_edge_contrast_judge_is_cached_and_confidence_gated(tmp_path):
    spec = {
        value.property_id: value for value in load_temporal_relation_registry().relations
    }["P286"]
    candidate = EdgeCandidate(
        spec=spec, direction="forward", next_qid="Q2",
        next_title="New Coach", event_date=T1,
        qualifier_dates={"P580": T1}, kg_subject_qid="Q1",
        kg_object_qid="Q2",
    )
    support = WikipediaSupport(
        as_of=T1, evidence="Head coach New Coach", alias="New Coach",
        source_revision_id=2, target_revision_id=3,
        prior_target_visible=True,
        prior_evidence="Former player New Coach appeared for the club.",
        prior_revision_id=1,
    )
    calls = []

    def fake_call(model, messages, temperature=0.0):
        calls.append(messages)
        return (
            '{"decision":"pass","confidence":0.93,'
            '"checks":{"after_relation_supported":true,'
            '"before_relation_absent":true,"direction_correct":true},'
            '"reason":"the earlier excerpt names a different role"}'
        )

    judge = TemporalEdgeContrastJudge(
        "judge/model", cache_path=str(tmp_path / "edge.db"),
        call_model_fn=fake_call,
    )
    first = judge.judge(
        spec, candidate, support, source_title="Example Team",
        target_title="New Coach", previous_as_of=CUTOFF,
    )
    second = judge.judge(
        spec, candidate, support, source_title="Example Team",
        target_title="New Coach", previous_as_of=CUTOFF,
    )
    judge.close()
    assert first["decision"] == "pass" and first["cache_hit"] is False
    assert second["cache_hit"] is True and len(calls) == 1

    def incomplete_call(model, messages, temperature=0.0):
        return (
            '{"decision":"pass","confidence":0.99,'
            '"checks":{"after_relation_supported":true},'
            '"reason":"incomplete"}'
        )

    rejected = TemporalEdgeContrastJudge(
        "judge/model", call_model_fn=incomplete_call,
    ).judge(
        spec, candidate, support, source_title="Example Team",
        target_title="New Coach", previous_as_of=CUTOFF,
    )
    assert rejected["decision"] == "reject"
    assert set(rejected["schema_gate_errors"]) == {
        "check_not_true:before_relation_absent",
        "check_not_true:direction_correct",
    }


def test_full_admission_records_case_and_deduplicates(tmp_path, monkeypatch):
    from tkg.experiment import renewable_question_engine as module

    registry, profile_path = _profile(tmp_path)
    admissions = load_relation_admissions([profile_path], registry)
    backend = FakeBackend()
    engine = RenewableQuestionEngine(
        model_id="openai/gpt-4.1-mini", until=T3,
        registry=registry, admissions=admissions, backend=backend,
        inverse_lookup_fn=lambda *args, **kwargs: [],
        cutoff_anchor_lookup_fn=lambda *args, **kwargs: [],
        terminal_attribute_tails=False,
        min_hops=4, max_hops=4, min_families=3,
    )
    seeds = engine.discover()
    monkeypatch.setattr(module, "validate_shortest_arena", lambda *args, **kwargs: {
        "passed": True, "semantic_shortest_distance": 7,
        "raw_shortest_distance": 7, "raw_shortest_matches_semantic": True,
    })

    class Judge:
        def judge(self, question, chain):
            return {"decision": "pass", "confidence": 0.99, "cache_hit": False}

    ledger = QuestionLedger(tmp_path / "admission.db")
    packets, cases = admit_seeds(
        seeds, backend=backend, judge=Judge(), ledger=ledger,
        judge_workers=2, backlink_branch_cap=10,
    )
    assert packets[0]["status"] == "machine_pass_human_review_required"
    assert len(cases) == 1 and cases[0]["required_temporal_switches"] == 3
    fingerprint = seeds[0]["selection_metadata"]["question_fingerprint"]
    assert ledger.excludes(fingerprint) is True
    assert ledger.get(fingerprint)["status"] == "machine_pass_human_review_required"
    ledger.close()


def test_diversity_sampler_caps_duplicate_property_sequences():
    def seed(identifier, anchor, properties, families, score):
        return {
            "id": identifier, "anchor_label": anchor,
            "selection_metadata": {
                "beam_score": score, "relation_families": families,
                "semantic_identity": {
                    "property_steps": [
                        {"property_id": value, "direction": "forward"}
                        for value in properties
                    ],
                },
            },
        }

    values = [
        seed("a", "Anchor A", ["P1", "P2"], ["leadership"], 10),
        seed("b", "Anchor B", ["P1", "P2"], ["leadership"], 9),
        seed("c", "Anchor A", ["P3", "P4"], ["family", "career"], 8),
    ]
    selected = select_diverse_seeds(
        values, max_total=3, max_per_anchor=1, max_per_property_sequence=1
    )
    assert {value["id"] for value in selected} == {"b", "c"}


def test_diversity_sampler_hard_caps_families_and_properties():
    def seed(identifier, property_id, family, score):
        return {
            "id": identifier, "anchor_label": f"Anchor {identifier}",
            "selection_metadata": {
                "beam_score": score, "relation_families": [family],
                "semantic_identity": {"property_steps": [{
                    "property_id": property_id, "direction": "forward",
                }]},
            },
        }

    selected = select_diverse_seeds(
        [
            seed("a", "P1", "leadership", 10),
            seed("b", "P2", "leadership", 9),
            seed("c", "P3", "family", 8),
            seed("d", "P1", "career", 7),
        ],
        max_total=4, max_per_anchor=1, max_per_property_sequence=2,
        max_per_family=1, max_per_property=1,
    )
    assert {value["id"] for value in selected} == {"a", "c"}
    assert RELATION_BY_PROPERTY["P27"].guessability == "easy"
    assert RELATION_BY_PROPERTY["P166"].guessability == "hard"


def test_registry_renewal_materializes_discovery_and_requires_validation(tmp_path):
    registry, profile_path = _profile(tmp_path)
    payload = json.loads(profile_path.read_text())
    payload["property_mining"] = {
        "method": "wikidata_recent_changes", "candidates": [{
            "property_id": "P999", "label": "example relation",
            "qualifiers": ["P580"], "status": "discovered",
        }],
    }
    renewed = renew_registry(
        registry, [(str(profile_path), "a" * 64, payload)],
        registry_version="test-renewal-v1", activate_properties={"P169"},
    )
    by_property = {spec.property_id: spec for spec in renewed.relations}
    assert by_property["P169"].status == "active"
    assert by_property["P26"].status == "validated"
    assert by_property["P999"].status == "discovered"
    assert by_property["P999"].relative_clause == "the example relation of {source}"
    assert renewed.provenance["promotion_policy"] == "explicit_property_review"
    active_admissions = load_relation_admissions(
        [profile_path], renewed, require_active=True,
    )
    assert [value.spec.property_id for value in active_admissions] == ["P169"]

    try:
        load_relation_admissions([profile_path], registry, require_active=True)
    except ValueError as exc:
        assert "no active semantically validated relation" in str(exc)
    else:
        raise AssertionError("formal admission accepted an inactive bootstrap relation")

    try:
        renew_registry(
            renewed, [], registry_version="test-renewal-v2",
            activate_properties={"P999"},
        )
    except ValueError as exc:
        assert "requires semantic validation" in str(exc)
    else:
        raise AssertionError("unvalidated discovered property was activated")
