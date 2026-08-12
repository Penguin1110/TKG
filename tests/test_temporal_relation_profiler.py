"""Offline tests for renewable temporal relation profiling."""

from __future__ import annotations

from tkg.experiment.contracts import PageLink, PageSnapshot
from tkg.experiment.temporal_relation_profiler import (
    RelationEvidenceJudge, build_relation_profile, discover_temporal_properties,
    discover_temporal_properties_from_recent_changes, query_relation_samples,
    query_relation_samples_stratified, relation_sample_query,
    relation_time_buckets, verify_wikipedia_binding,
)
from tkg.experiment.temporal_relation_registry import (
    TemporalRelationSpec, load_temporal_relation_registry,
)


def _value(value):
    return {"type": "uri" if str(value).startswith("http") else "literal", "value": value}


class Response:
    status_code = 200

    def __init__(self, rows):
        self.rows = rows

    def raise_for_status(self):
        return None

    def json(self):
        return {"results": {"bindings": self.rows}}


def _spec():
    return TemporalRelationSpec(
        property_id="P169", label="chief executive officer", family="leadership",
        time_mode="interval", source_kind="organization", target_kind="person",
        answer_kind="Who", relative_clause="the chief executive officer of {source}",
        status="bootstrap",
    )


def _row():
    return {
        "source": _value("http://www.wikidata.org/entity/Q1"),
        "sourceArticle": _value("https://en.wikipedia.org/wiki/Example_Corp"),
        "target": _value("http://www.wikidata.org/entity/Q2"),
        "targetArticle": _value("https://en.wikipedia.org/wiki/Alice_Stone"),
        "start": _value("2025-01-02T00:00:00Z"),
        "end": _value("2026-01-02T00:00:00Z"),
    }


def test_bootstrap_registry_is_versioned_and_data_driven():
    registry = load_temporal_relation_registry()
    assert registry.schema_version == "temporal-relation-registry-v1"
    assert len(registry.relations) == 24
    assert {relation.property_id for relation in registry.relations} >= {
        "P286", "P169", "P488", "P6", "P35", "P108", "P54", "P26",
    }
    assert len({relation.family for relation in registry.relations}) >= 10
    by_property = {relation.property_id: relation for relation in registry.relations}
    assert "Prime Minister" in by_property["P6"].semantic_guidance
    assert "Reject membership" in by_property["P108"].semantic_guidance


def test_relation_query_and_binding_parser_preserve_temporal_contract():
    query = relation_sample_query(
        _spec(), since="2024-06-01", until="2026-08-12", limit=5
    )
    assert "p:P169" in query and "ps:P169" in query
    assert "pq:P580" in query and "pq:P582" in query
    assert "2024-06-01T00:00:00Z" in query
    assert "ORDER BY DESC(?start)" in query

    calls = []

    def request_get(url, **kwargs):
        calls.append((url, kwargs))
        return Response([_row()])

    samples, query_hash, raw_count = query_relation_samples(
        _spec(), since="2024-06-01", until="2026-08-12", limit=5,
        request_get=request_get,
    )
    assert raw_count == 1 and len(samples) == 1
    assert samples[0].source_title == "Example Corp"
    assert samples[0].target_title == "Alice Stone"
    assert samples[0].event_date == "2025-01-02"
    assert len(query_hash) == 64
    assert calls[0][1]["headers"]["User-Agent"]

    ended_row = _row()
    ended_row["start"] = _value("2020-01-02T00:00:00Z")
    ended_row["end"] = _value("2025-08-02T00:00:00Z")
    ended, _, _ = query_relation_samples(
        _spec(), since="2024-06-01", until="2026-08-12", limit=5,
        request_get=lambda *args, **kwargs: Response([ended_row]),
    )
    assert ended[0].event_date == "2025-08-02"


def test_stratified_sampling_preserves_early_and_late_window_yield():
    assert relation_time_buckets("2024-01-01", "2024-01-04", 2) == [
        ("2024-01-01", "2024-01-02"),
        ("2024-01-03", "2024-01-04"),
    ]

    def dated_row(source_qid, target_qid, event_date):
        row = _row()
        row["source"] = _value(f"http://www.wikidata.org/entity/{source_qid}")
        row["sourceArticle"] = _value(
            f"https://en.wikipedia.org/wiki/Source_{source_qid}"
        )
        row["target"] = _value(f"http://www.wikidata.org/entity/{target_qid}")
        row["targetArticle"] = _value(
            f"https://en.wikipedia.org/wiki/Target_{target_qid}"
        )
        row["start"] = _value(f"{event_date}T00:00:00Z")
        row.pop("end", None)
        return row

    def request_get(url, **kwargs):
        query = kwargs["params"]["query"]
        if "2024-01-01T00:00:00Z" in query:
            return Response([dated_row("Q11", "Q21", "2024-01-02")])
        return Response([dated_row("Q12", "Q22", "2024-01-04")])

    samples, query_hash, raw_count = query_relation_samples_stratified(
        _spec(), since="2024-01-01", until="2024-01-04", limit=4,
        time_buckets=2, request_get=request_get,
    )
    assert {sample.event_date for sample in samples} == {"2024-01-02", "2024-01-04"}
    assert raw_count == 2 and len(query_hash) == 64


def test_property_miner_marks_unregistered_candidates_without_templates():
    rows = [{
        "property": _value("http://www.wikidata.org/entity/P999"),
        "propertyLabel": _value("example temporal property"),
        "qualifier": _value("http://www.wikidata.org/entity/P580"),
    }]

    def request_get(url, **kwargs):
        return Response(rows)

    values, query_hash = discover_temporal_properties(
        since="2024-06-01", until="2026-08-12", limit=10,
        request_get=request_get,
    )
    assert values == [{
        "property_id": "P999", "label": "example temporal property",
        "qualifiers": ["P580"], "status": "discovered",
    }]
    assert len(query_hash) == 64


def test_recent_change_miner_discovers_entity_valued_temporal_claims():
    class ApiResponse(Response):
        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    def api_get(url, **kwargs):
        params = kwargs["params"]
        if params.get("list") == "recentchanges":
            return ApiResponse({"query": {"recentchanges": [{"title": "Q1"}]}})
        if params.get("props") == "claims":
            return ApiResponse({"entities": {"Q1": {"claims": {"P108": [{
                "mainsnak": {"datavalue": {"value": {"id": "Q2"}}},
                "qualifiers": {"P580": [{
                    "datavalue": {"value": {"time": "+2025-01-02T00:00:00Z"}},
                }]},
            }]}}}})
        if params.get("props") == "labels":
            return ApiResponse({"entities": {"P108": {
                "labels": {"en": {"value": "employer"}},
            }}})
        raise AssertionError(params)

    values, manifest_hash = discover_temporal_properties_from_recent_changes(
        since="2024-06-01", until="2026-08-12", property_limit=10,
        recent_entity_limit=50, request_get=api_get,
    )
    assert values == [{
        "property_id": "P108", "label": "employer", "qualifiers": ["P580"],
        "recent_statement_count": 1, "sample_source_qids": ["Q1"],
        "status": "discovered",
    }]
    assert len(manifest_hash) == 64


def test_wikipedia_profile_measures_support_and_prior_day_novelty():
    sample, _, _ = query_relation_samples(
        _spec(), since="2024-06-01", until="2026-08-12", limit=5,
        request_get=lambda *args, **kwargs: Response([_row()]),
    )

    class Backend:
        def fetch_page(self, title, as_of=None):
            has_target = as_of != "2025-01-01"
            return PageSnapshot(
                title="Example Corp", page_id=1,
                revision_id=2 if has_target else 1,
                timestamp=f"{as_of}T00:00:00Z", as_of=as_of,
                content=(
                    "Example Corp appointed Alice Stone as chief executive officer."
                    if has_target else "Example Corp had another chief executive officer."
                ),
                links=(
                    [PageLink(target="Alice Stone", anchor="Alice Stone")]
                    if has_target else []
                ),
                source_url="https://example.invalid/oldid=2",
            )

    check = verify_wikipedia_binding(sample[0], Backend())
    assert check["status"] == "link_supported"
    assert check["supported_as_of"] == "2025-01-02"
    assert check["prior_day_absent"] is True
    profile = build_relation_profile(
        _spec(), samples=sample, raw_row_count=1, query_sha256="a" * 64,
        wiki_checks=[check], query_error=None, min_kg_samples=1,
        min_wiki_supported=1, min_wiki_support_rate=0.5,
    )
    assert profile["metrics"]["wiki_support_rate"] == 1.0
    assert profile["recommendation"] == (
        "wikipedia_link_candidate_semantic_review_required"
    )


def test_relation_semantic_judge_is_cached_and_controls_promotion(tmp_path):
    sample, _, _ = query_relation_samples(
        _spec(), since="2024-06-01", until="2026-08-12", limit=5,
        request_get=lambda *args, **kwargs: Response([_row()]),
    )

    class Backend:
        def fetch_page(self, title, as_of=None):
            return PageSnapshot(
                title="Example Corp", page_id=1, revision_id=2,
                timestamp=f"{as_of}T00:00:00Z", as_of=as_of,
                content="Chief executive officer Alice Stone",
                links=[PageLink(target="Alice Stone", anchor="Alice Stone")],
            )

    check = verify_wikipedia_binding(sample[0], Backend())
    calls = []

    def fake_call(model, messages, temperature=0.0):
        calls.append(messages)
        return (
            '{"decision":"pass","confidence":0.95,'
            '"expressed_relation":"chief executive officer",'
            '"checks":{"direction":true,"relation_semantics":true,'
            '"temporal_support":true,"not_mere_cooccurrence":true},'
            '"reason":"explicit CEO field"}'
        )

    judge = RelationEvidenceJudge(
        "judge/model", cache_path=str(tmp_path / "judge.db"),
        call_model_fn=fake_call,
    )
    first = judge.judge(_spec(), check)
    second = judge.judge(_spec(), check)
    judge.close()
    assert first["cache_hit"] is False and second["cache_hit"] is True
    assert len(calls) == 1

    check["semantic_judge"] = first
    profile = build_relation_profile(
        _spec(), samples=sample, raw_row_count=1, query_sha256="a" * 64,
        wiki_checks=[check], query_error=None, min_kg_samples=1,
        min_wiki_supported=1, min_wiki_support_rate=0.5,
        semantic_judge_requested=True, min_semantic_supported=1,
        min_semantic_support_rate=0.5,
    )
    assert profile["metrics"]["semantic_support_rate"] == 1.0
    assert profile["recommendation"] == (
        "semantic_validated_candidate_human_review_required"
    )

    def incomplete_call(model, messages, temperature=0.0):
        return (
            '{"decision":"pass","confidence":0.99,'
            '"expressed_relation":"chief executive officer",'
            '"checks":{"direction":true},"reason":"incomplete"}'
        )

    rejected = RelationEvidenceJudge(
        "judge/model", call_model_fn=incomplete_call,
    ).judge(_spec(), check)
    assert rejected["decision"] == "reject"
    assert set(rejected["schema_gate_errors"]) == {
        "check_not_true:relation_semantics",
        "check_not_true:temporal_support",
        "check_not_true:not_mere_cooccurrence",
    }
