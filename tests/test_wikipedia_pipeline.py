"""Offline contract tests for the refactored Wikipedia experiment."""

from __future__ import annotations

import json
import os
import random
import sqlite3
import sys
import tempfile
import csv
from types import SimpleNamespace

from legacy.wikipedia_prior_reversion_analysis import main as analyze_main
from tkg.experiment.contracts import JudgeResult, PageLink, PageSnapshot
from tkg.experiment.results import JsonlResultStore, assert_new_output_path
from legacy.wikipedia_prior_reversion_runner import run_trajectory
from tkg.experiment.temporal_runner import (
    _capability_metrics, _critical_bridge_evidence_metrics,
    _failure_mode, _navigation_metrics,
    _snapshot_range, _snapshot_values,
    run_case as run_temporal_case,
    run_pk_admission, write_scores as write_temporal_scores,
)
from legacy.wikipedia_prior_reversion_gates import answerability_for_item, evaluate_exposure
from tkg.judging.llm import LLMJudge, transition_label
from tkg.wikipedia.backend import WikipediaError, WikipediaPageBackend, reverse_bfs_frontier
from tkg.wikipedia.browser import (
    run_snapshot_selection, run_temporal_browsing, run_wikipedia_browsing,
)
from tkg.wikipedia.snapshot import outgoing_bfs, temporal_reverse_bfs
from tkg.visualization.exporter import build_visualization_data
from tkg.visualization.server import build_site


class _Response:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self.payload


class _FakeWikiAPI:
    def __init__(self):
        self.calls = []

    def __call__(self, url, params, headers, timeout):
        self.calls.append((url, dict(params)))
        if params.get("list") == "backlinks":
            return _Response({"query": {"backlinks": [{"title": "Source Page"}]}})
        if params.get("action") == "query" and params.get("prop") == "revisions":
            title = params["titles"]
            values = {
                "Pivot Page": (1, 10, "2025-01-01T10:00:00Z"),
                "Source Page": (2, 20, "2024-12-31T10:00:00Z"),
            }
            page_id, rev_id, timestamp = values[title]
            return _Response({"query": {"pages": [{
                "pageid": page_id, "title": title,
                "revisions": [{"revid": rev_id, "timestamp": timestamp}],
            }]}})
        if params.get("action") == "parse":
            oldid = int(params["oldid"])
            if oldid == 10:
                return _Response({"parse": {
                    "title": "Pivot Page",
                    "text": '<div><table><tr><th>Leader</th><td>New Person</td></tr></table>'
                            '<p>The updated leader is <b>New Person</b>.</p></div>',
                    "links": [],
                }})
            return _Response({"parse": {
                "title": "Source Page",
                "text": '<div><p>Read the <a href="./Pivot_Page">target article</a> next.</p>'
                        '<sup>citation noise</sup></div>',
                "links": [{"ns": 0, "title": "Pivot Page"}],
            }})
        raise AssertionError(f"unexpected API request: {url} {params}")


class _RevisionDateAPI:
    def __init__(self):
        self.calls = []

    def __call__(self, url, params, headers, timeout):
        self.calls.append(dict(params))
        return _Response({"query": {"pages": [{
            "pageid": 1, "title": "Busy Page", "revisions": [
                {"timestamp": f"2024-0{month}-01T12:00:00Z"}
                for month in range(1, 6)
            ],
        }]}})


class _ExactRevisionAndPaginationAPI:
    def __init__(self):
        self.calls = []

    def __call__(self, url, params, headers, timeout):
        del url, headers, timeout
        self.calls.append(dict(params))
        if params.get("revids") == "77":
            return _Response({"query": {"pages": [{
                "pageid": 7,
                "title": "Exact Page",
                "revisions": [{
                    "revid": 77, "parentid": 76,
                    "timestamp": "2025-02-03T04:05:06Z",
                }],
            }]}})
        if params.get("action") == "parse" and params.get("oldid") == 77:
            return _Response({"parse": {
                "title": "Exact Page",
                "text": '<p>Exact content links <a href="./Target">Target</a>.</p>',
                "links": [{"ns": 0, "title": "Target"}],
            }})
        if params.get("prop") == "revisions" and params.get("titles") == "Busy Page":
            offset = 2 if params.get("rvcontinue") == "next-2" else 0
            revisions = [
                {"revid": 100 + index, "parentid": 99 + index,
                 "timestamp": f"2025-01-0{index + 1}T00:00:00Z"}
                for index in range(offset, min(offset + 2, 3))
            ]
            result = {"query": {"pages": [{
                "pageid": 8, "title": "Busy Page", "revisions": revisions,
            }]}}
            if offset == 0:
                result["continue"] = {"rvcontinue": "next-2", "continue": "||"}
            return _Response(result)
        raise AssertionError(f"unexpected exact/pagination request: {params}")


def _page(title, revision, content, links=()):
    return PageSnapshot(
        title=title, page_id=revision, revision_id=revision,
        timestamp="2025-01-01T00:00:00Z", as_of="2025-01-01",
        content=content, links=[PageLink(target=t, anchor=a) for t, a in links],
    )


def test_backend_revision_dates_are_sampled_cached_and_content_free():
    api = _RevisionDateAPI()
    with tempfile.TemporaryDirectory() as tmp:
        backend = WikipediaPageBackend(
            cache_path=os.path.join(tmp, "wiki.db"), request_get=api,
            min_request_interval=0,
        )
        first = backend.list_revision_dates(
            "Busy Page", "2024-01-01", "2024-05-31", limit=3
        )
        second = backend.list_revision_dates(
            "Busy Page", "2024-01-01", "2024-05-31", limit=2
        )
        backend.close()
    assert first == ["2024-01-01", "2024-03-01", "2024-05-01"]
    assert second == ["2024-01-01", "2024-05-01"]
    assert len(api.calls) == 1
    assert api.calls[0]["rvprop"] == "timestamp"
    assert "content" not in json.dumps(first)


def test_backend_exact_revision_is_cached_for_offline_environment():
    api = _ExactRevisionAndPaginationAPI()
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "wiki.db")
        backend = WikipediaPageBackend(
            cache_path=cache, request_get=api, min_request_interval=0,
        )
        page = backend.fetch_revision(77)
        backend.close()
        offline = WikipediaPageBackend(cache_path=cache, offline_only=True)
        cached = offline.fetch_revision(77)
        offline.close()
    assert page.revision_id == cached.revision_id == 77
    assert page.title == cached.title == "Exact Page"
    assert page.links == [PageLink(target="Target", anchor="Target")]
    assert sum(call.get("revids") == "77" for call in api.calls) == 1
    assert sum(call.get("action") == "parse" for call in api.calls) == 1


def test_backend_revision_metadata_pagination_is_unsampled_and_cached():
    api = _ExactRevisionAndPaginationAPI()
    with tempfile.TemporaryDirectory() as tmp:
        backend = WikipediaPageBackend(
            cache_path=os.path.join(tmp, "wiki.db"), request_get=api,
            min_request_interval=0,
        )
        first = backend.list_revision_metadata_page(
            "Busy Page", "2025-01-01", "2025-01-31", page_size=2,
        )
        second = backend.list_revision_metadata_page(
            "Busy Page", "2025-01-01", "2025-01-31",
            cursor=first["next_cursor"], page_size=2,
        )
        repeated = backend.list_revision_metadata_page(
            "Busy Page", "2025-01-01", "2025-01-31",
            cursor=first["next_cursor"], page_size=2,
        )
        backend.close()
    assert [row["revision_id"] for row in first["revisions"]] == [100, 101]
    assert [row["revision_id"] for row in second["revisions"]] == [102]
    assert repeated == second
    assert len(api.calls) == 2
    assert all(call["rvprop"] == "ids|timestamp" for call in api.calls)


class _MockBackend:
    def __init__(self):
        self.pages = {
            "Source Page": _page("Source Page", 1, "Go to [Pivot -> Pivot Page].",
                                 [("Pivot Page", "Pivot")]),
            "Pivot Page": _page("Pivot Page", 2, "The updated leader is New Person."),
        }

    def fetch_page(self, title, as_of=None):
        return self.pages[title]

    def find_backlinks(self, title, as_of=None, max_results=50):
        return ["Source Page"] if title == "Pivot Page" else []


class _ScriptedToolModel:
    def __call__(self, model, messages, tools, temperature=0.7):
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call-1", "type": "function",
            "function": {"name": "follow_link", "arguments": '{"target":"Pivot Page"}'},
        }]}


class _SnapshotSelectionModel:
    def __init__(self, choices):
        self.choices = list(choices)
        self.calls = []

    def __call__(self, model, messages, tools, temperature=0.0):
        self.calls.append({"messages": list(messages), "tools": tools})
        choice = self.choices[min(len(self.calls) - 1, len(self.choices) - 1)]
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"snapshot-{len(self.calls)}", "type": "function",
            "function": {"name": "select_snapshot", "arguments": json.dumps(choice)},
        }]}


class _TemporalBackend:
    def fetch_page(self, title, as_of=None):
        values = {
            ("Pivot Page", "2024-01-01"): _page(
                "Pivot Page", 10, "The leader is Old Person.", [("Old Page", "old link")]
            ),
            ("Pivot Page", "2025-01-01"): _page(
                "Pivot Page", 20, "The leader is New Person.", [("New Page", "new link")]
            ),
            ("Source Page", "2024-01-01"): _page(
                "Source Page", 11, "Follow the pivot.", [("Pivot Page", "pivot")]
            ),
            ("Source Page", "2025-01-01"): _page(
                "Source Page", 21, "Follow the pivot.", [("Pivot Page", "pivot")]
            ),
        }
        page = values[(title, as_of)]
        page.as_of = as_of
        return page

    def find_backlinks(self, title, as_of=None, max_results=50):
        return ["Source Page"] if title == "Pivot Page" else []


class _ScriptedTemporalModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, model, messages, tools, temperature=0.7):
        actions = [
            ("switch_snapshot", {
                "as_of": "2024-01-01", "brief_reason": "Inspect the earlier revision."
            }),
            ("switch_snapshot", {
                "as_of": "2025-01-01", "brief_reason": "Compare the later revision."
            }),
            ("submit_answer", {"answer": "New Person"}),
        ]
        name, args = actions[self.calls]
        self.calls += 1
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"temporal-{self.calls}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}


class _ScriptedDateRangeModel:
    def __init__(self, actions):
        self.actions = actions
        self.calls = []

    def __call__(self, model, messages, tools, temperature=0.7):
        self.calls.append({"messages": list(messages), "tools": tools})
        name, args = self.actions[len(self.calls) - 1]
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"range-{len(self.calls)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}


class _DateRangeBackend:
    def __init__(self):
        self.requests = []

    def fetch_page(self, title, as_of=None):
        self.requests.append((title, as_of))
        revision = 20 if as_of == "2025-01-01" else 15
        page = _page(title, revision, f"Visible at {as_of}.")
        page.as_of = as_of
        page.timestamp = (
            "2025-01-01T00:00:00Z" if revision == 20
            else "2024-05-01T00:00:00Z"
        )
        return page


class _ScriptedShortestModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, model, messages, tools, temperature=0.7):
        actions = [
            ("switch_snapshot", {
                "as_of": "2025-01-01", "brief_reason": "Choose the target-time graph."
            }),
            ("follow_link", {"target": "Pivot Page"}),
            ("submit_answer", {"answer": "New Person"}),
        ]
        name, args = actions[self.calls]
        self.calls += 1
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"shortest-{self.calls}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}


class _CycleBackend:
    def __init__(self):
        self.pages = {
            "Pivot": _page("Pivot", 1, "target"),
            "A": _page("A", 2, "links", [("Pivot", "target"), ("B", "cycle")]),
            "B": _page("B", 3, "links", [("A", "cycle")]),
        }

    def fetch_page(self, title, as_of=None):
        page = self.pages[title]
        page.as_of = as_of
        return page

    def find_backlinks(self, title, as_of=None, max_results=50):
        return {
            "Pivot": ["A"],
            "A": ["B"],
            "B": ["A"],
        }.get(title, [])[:max_results]


class _MockJudge:
    def __init__(self, visibility="visible", answer="stick_new", answerable="answerable"):
        self.visibility = visibility
        self.answer = answer
        self.answerable = answerable

    def judge_visibility(self, question, new_answers, old_answers, pages):
        return JudgeResult(self.visibility, 1.0, "mock", "New Person")

    def judge_answer(self, question, response, new_answers, old_answers, pages):
        return JudgeResult(self.answer, 1.0, "mock", "New Person", "New Person")

    def judge_answerability(self, question, accepted_answers, pages):
        return JudgeResult(self.answerable, 1.0, "mock", "New Person")

    def judge_temporal_answer(
        self, question, response, after_answers, before_answers, pages,
        *, target_snapshot_as_of=None,
    ):
        return JudgeResult("correct_after", 1.0, "mock", "New Person", "New Person")


def test_backend_revision_content_links_and_offline_cache():
    fake = _FakeWikiAPI()
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "wiki.db")
        backend = WikipediaPageBackend(
            cache_path=cache, request_get=fake, min_request_interval=0,
        )
        page = backend.fetch_page("Source Page", as_of="2025-01-01")
        assert page.revision_id == 20
        assert "[target article -> Pivot Page]" in page.content
        assert page.links == [PageLink(target="Pivot Page", anchor="target article")]
        assert "citation noise" not in page.content
        pivot = backend.fetch_page("Pivot Page", as_of="2025-01-01")
        assert "Leader" in pivot.content and "New Person" in pivot.content
        backlinks = backend.find_backlinks("Pivot Page", as_of="2025-01-01")
        assert backlinks == ["Source Page"]
        backend.close()

        # The selected revisions and reverse-link index must work without network.
        offline = WikipediaPageBackend(cache_path=cache, offline_only=True)
        cached = offline.fetch_page("Source Page", as_of="2025-01-01")
        assert cached.revision_id == 20
        assert offline.find_backlinks("Pivot Page", as_of="2025-01-01") == ["Source Page"]
        offline.close()
    revision_calls = [params for _, params in fake.calls if params.get("prop") == "revisions"]
    assert revision_calls and all(call.get("rvstart") == "2025-01-01T23:59:59Z"
                                  for call in revision_calls)


def test_backend_reuses_parsed_revision_across_requested_dates():
    fake = _FakeWikiAPI()
    with tempfile.TemporaryDirectory() as tmp:
        backend = WikipediaPageBackend(
            cache_path=os.path.join(tmp, "wiki.db"),
            request_get=fake,
            min_request_interval=0,
        )
        first = backend.fetch_page("Source Page", as_of="2025-01-01")
        second = backend.fetch_page("Source Page", as_of="2025-01-02")
        backend.close()
    assert first.revision_id == second.revision_id == 20
    revision_calls = [
        params for _, params in fake.calls if params.get("prop") == "revisions"
    ]
    parse_calls = [params for _, params in fake.calls if params.get("action") == "parse"]
    assert len(revision_calls) == 2
    assert len(parse_calls) == 1


def test_backend_caches_backlink_candidate_queries():
    fake = _FakeWikiAPI()
    with tempfile.TemporaryDirectory() as tmp:
        backend = WikipediaPageBackend(
            cache_path=os.path.join(tmp, "wiki.db"), request_get=fake,
            min_request_interval=0, backlink_verify_workers=4,
        )
        assert backend.find_backlinks(
            "Pivot Page", as_of="2025-01-01", max_results=10,
        ) == ["Source Page"]
        first_count = sum(
            params.get("list") == "backlinks" for _, params in fake.calls
        )
        assert backend.find_backlinks(
            "Pivot Page", as_of="2025-01-01", max_results=10,
        ) == ["Source Page"]
        second_count = sum(
            params.get("list") == "backlinks" for _, params in fake.calls
        )
        backend.close()
    assert first_count == second_count == 1


def test_backend_enforces_hard_api_attempt_budget_during_429_retries():
    calls = []

    def always_429(url, params, headers, timeout):
        del url, params, headers, timeout
        calls.append(1)
        return _Response({}, status=429)

    with tempfile.TemporaryDirectory() as tmp:
        backend = WikipediaPageBackend(
            cache_path=os.path.join(tmp, "wiki.db"), request_get=always_429,
            min_request_interval=0, max_api_calls=2, api_max_retries=6,
            backoff_base_seconds=0, backoff_max_seconds=0,
        )
        try:
            backend.fetch_page("Rate Limited", as_of="2025-01-01")
        except WikipediaError as exc:
            assert "API call budget exhausted" in str(exc)
        else:
            raise AssertionError("request budget was not enforced")
        assert backend.request_stats() == {
            "api_calls_used": 2,
            "api_call_budget": 2,
            "api_calls_remaining": 0,
        }
        backend.close()
    assert len(calls) == 2


def test_reverse_bfs_uses_backlinks():
    frontier = reverse_bfs_frontier(_MockBackend(), "Pivot Page", 2, as_of="2025-01-01")
    assert frontier == {0: ["Pivot Page"], 1: ["Source Page"]}


def test_temporal_snapshot_uses_outgoing_links_from_each_revision():
    frontier = outgoing_bfs(
        _MockBackend(), "Source Page", 2, as_of="2025-01-01", branch_cap=10
    )
    assert frontier == {0: ["Source Page"], 1: ["Pivot Page"]}


def test_temporal_reverse_bfs_keeps_cycles_but_assigns_minimum_distance():
    graph = temporal_reverse_bfs(
        _CycleBackend(), "Pivot", "2025-01-01", ["2025-01-01"], 3,
        branch_cap=10,
    )
    by_title = {state["title"]: state for state in graph["states"].values()}
    assert by_title["Pivot"]["distance_to_pivot"] == 0
    assert by_title["A"]["distance_to_pivot"] == 1
    assert by_title["B"]["distance_to_pivot"] == 2
    edges = {(edge["source"], edge["target"]) for edge in graph["arena_edges"]}
    assert (by_title["A"]["key"], by_title["B"]["key"]) in edges
    assert (by_title["B"]["key"], by_title["A"]["key"]) in edges
    assert len([state for state in graph["states"].values()
                if state["title"] == "Pivot"]) == 1

    result = {"trajectory": [
        {"step": 1, "action": "switch_snapshot", "result": "ok",
         "to_title": "A", "revision_id": 2, "snapshot_token": "2025-01-01"},
        {"step": 2, "action": "follow_link", "result": "ok",
         "to_title": "B", "revision_id": 3, "snapshot_token": "2025-01-01"},
        {"step": 3, "action": "follow_link", "result": "ok",
         "to_title": "A", "revision_id": 2, "snapshot_token": "2025-01-01"},
        {"step": 4, "action": "follow_link", "result": "ok",
         "to_title": "Pivot", "revision_id": 1, "snapshot_token": "2025-01-01"},
    ]}
    metrics = _navigation_metrics(result, graph, "A")
    assert metrics["shortest_navigation_steps"] == 2
    assert metrics["actual_steps_to_first_pivot"] == 4
    assert metrics["detour_steps"] == 2
    assert metrics["revisit_count"] == 1
    assert metrics["cycle_detected"] is True


def test_snapshot_selection_is_a_required_auditable_first_tool():
    model = _SnapshotSelectionModel([
        {"as_of": "not-allowed", "intent_code": "uncertain", "brief_reason": "Trying."},
        {"as_of": "2025-01-01", "intent_code": "latest_available",
         "brief_reason": "I want the newest allowed snapshot."},
    ])
    result = run_snapshot_selection(
        "tested/model", ["2024-01-01", "2025-01-01"],
        snapshot_mode="agent_selected", task_prompt="neutral", call_model_fn=model,
    )
    assert result["status"] == "selected"
    assert result["selected_as_of"] == "2025-01-01"
    assert result["intent_code"] == "latest_available"
    assert [attempt["status"] for attempt in result["attempts"]] == ["invalid", "selected"]
    assert all(call["tools"][0]["function"]["name"] == "select_snapshot"
               for call in model.calls)
    assert result["messages"][-1]["role"] == "tool"


def test_dynamic_navigation_marks_revisions_outside_reference_without_crashing():
    backend = _TemporalBackend()
    graph = temporal_reverse_bfs(
        backend, "Pivot Page", "2025-01-01",
        ["2024-01-01", "2025-01-01"], 1, branch_cap=10,
    )
    result = {"trajectory": [
        {
            "step": 1, "action": "switch_snapshot", "result": "ok",
            "to_title": "Source Page", "revision_id": 999,
            "snapshot_token": "2024-06-15",
        },
        {
            "step": 2, "action": "follow_link", "result": "ok",
            "to_title": "Pivot Page", "revision_id": 20,
            "snapshot_token": "2025-01-01",
        },
    ]}
    metrics = _navigation_metrics(result, graph, "Source Page", strict_arena=False)
    assert metrics["pivot_hit"] is True
    assert metrics["outside_reference_arena_count"] == 1
    assert metrics["reference_distance_coverage_rate"] == 0.5


def test_temporal_browser_can_switch_time_repeatedly_before_answering():
    result = run_temporal_browsing(
        "tested/model", _TemporalBackend(), "Pivot Page", "Who is the leader?",
        ["2024-01-01", "2025-01-01"], 6,
        target_title="Pivot Page",
        target_as_of="2025-01-01",
        call_model_fn=_ScriptedTemporalModel(), verbose=False,
    )
    assert result["final_answer"] == "New Person"
    assert result["stop_reason"] == "submit_answer"
    assert [row["action"] for row in result["trajectory"]] == [
        "switch_snapshot", "switch_snapshot", "submit_answer",
    ]
    assert [row["revision_id"] for row in result["trajectory"][:2]] == [10, 20]
    assert [row["snapshot_token"] for row in result["trajectory"][:2]] == [
        "2024-01-01", "2025-01-01",
    ]
    assert {page["revision_id"] for page in result["visited_versions"]} == {10, 20}
    assert "Target snapshot to answer: 2025-01-01" in result["messages"][1]["content"]
    assert result["initial_state"]["snapshot_token"] == "2024-01-01"
    assert "first action must be switch_snapshot" not in result["messages"][1]["content"]


def test_temporal_browser_lists_only_revision_dates_and_starts_at_cutoff():
    class RevisionBackend(_DateRangeBackend):
        def __init__(self):
            super().__init__()
            self.revision_queries = []

        def list_revision_dates(self, title, from_date, to_date, limit=10):
            self.revision_queries.append((title, from_date, to_date, limit))
            return ["2024-03-02", "2024-08-17"]

    model = _ScriptedDateRangeModel([
        ("list_revisions", {
            "from": "2024-01-01", "to": "2025-01-01", "limit": 5,
        }),
        ("switch_snapshot", {
            "as_of": "2025-01-01", "brief_reason": "Inspect the target date.",
        }),
        ("submit_answer", {"answer": "New Person"}),
    ])
    backend = RevisionBackend()
    result = run_temporal_browsing(
        "tested/model", backend, "Pivot Page", "Who is the leader?",
        ["2024-01-01", "2025-01-01"], 3,
        target_title="Pivot Page", target_as_of="2025-01-01",
        snapshot_date_range=("2024-01-01", "2025-01-01"),
        cutoff_reference="2024-01-01", call_model_fn=model, verbose=False,
    )
    assert result["trajectory"][0]["action"] == "list_revisions"
    assert result["trajectory"][0]["result"] == '["2024-03-02", "2024-08-17"]'
    assert backend.revision_queries == [
        ("Pivot Page", "2024-01-01", "2025-01-01", 5)
    ]
    assert backend.requests[0] == ("Pivot Page", "2024-01-01")
    revision_tool = result["tool_contract"][1]["function"]
    assert revision_tool["name"] == "list_revisions"
    assert "comments" in revision_tool["description"]
    properties = revision_tool["parameters"]["properties"]
    assert "minimum" not in properties["from"]
    assert "maximum" not in properties["to"]

    capabilities = _capability_metrics(
        result, target_title="Some other proof-route page",
        target_as_of="2025-01-01", cutoff_reference="2024-01-01",
        accepted_answers=["Visible"],
    )
    assert capabilities["target_page_seen"] is False
    assert capabilities["target_snapshot_evidence_seen"] is True
    assert _failure_mode(
        capabilities, "correct_after", result["stop_reason"]
    ) == "success_with_target_evidence"


def test_temporal_browser_lets_model_choose_any_in_range_date_without_oracle_leak():
    model = _ScriptedDateRangeModel([
        ("switch_snapshot", {
            "as_of": "2023-12-31", "brief_reason": "Probe before the range."
        }),
        ("switch_snapshot", {
            "as_of": "2024-06-15", "brief_reason": "Inspect a date I selected."
        }),
        ("switch_snapshot", {
            "as_of": "2024-07-15", "brief_reason": "Move later and compare."
        }),
        ("switch_snapshot", {
            "as_of": "2025-01-01", "brief_reason": "Read the target snapshot."
        }),
        ("submit_answer", {"answer": "New Person"}),
    ])
    backend = _DateRangeBackend()
    result = run_temporal_browsing(
        "tested/model", backend, "Pivot Page", "Who is the leader?",
        ["2024-01-01", "2024-09-01", "2025-01-01"], 6,
        target_title="Pivot Page", target_as_of="2025-01-01",
        snapshot_date_range=("2024-01-01", "2025-01-01"),
        call_model_fn=model, verbose=False,
    )
    prompt = result["messages"][1]["content"]
    switch_schema = model.calls[0]["tools"][0]["function"]["parameters"][
        "properties"
    ]["as_of"]
    assert "enum" not in switch_schema
    assert switch_schema["format"] == "date"
    assert "2024-09-01" not in prompt
    assert "2024-01-01 through 2025-01-01" in prompt
    assert result["initial_messages"] == result["messages"][:2]
    assert result["snapshot_mode"] == "agent_selected_range"
    assert result["tool_contract"][0]["function"]["name"] == "switch_snapshot"
    assert result["trajectory"][0]["result"].startswith("Error:")
    assert ("Pivot Page", "2023-12-31") not in backend.requests
    assert [row["requested_snapshot_as_of"] for row in result["trajectory"][:4]] == [
        "2023-12-31", "2024-06-15", "2024-07-15", "2025-01-01",
    ]
    # Both model-selected intermediate dates resolve to revision 15 and are one graph node.
    revision_15 = next(
        row for row in result["visited_versions"] if row["revision_id"] == 15
    )
    assert revision_15["requested_snapshot_tokens"] == [
        "2024-01-01", "2024-06-15", "2024-07-15",
    ]
    assert len(result["visited_versions"]) == 2


def test_default_snapshot_range_exposes_only_case_endpoints():
    case = {
        "id": "temporal-case", "wikipedia_before": "2024-01-01",
        "wikipedia_as_of": "2025-01-01",
        "required_snapshot_dates": ["2024-01-01", "2024-09-01", "2025-01-01"],
    }
    assert _snapshot_range(case) == ("2024-01-01", "2025-01-01")
    assert _snapshot_values(None, case) == [
        "2024-01-01", "2024-09-01", "2025-01-01",
    ]


def test_multihop_browser_hides_pivot_but_exposes_cutoff_snapshot():
    result = run_temporal_browsing(
        "tested/model", _TemporalBackend(), "Source Page",
        "At the target snapshot, who follows the cutoff leader?",
        ["2024-01-01", "2025-01-01"], 6,
        target_title="Pivot Page", target_as_of="2025-01-01",
        reveal_target_title=False, cutoff_reference="2024-01-01",
        call_model_fn=_ScriptedShortestModel(), verbose=False,
    )
    prompt = result["messages"][1]["content"]
    assert "Target pivot page title" not in prompt
    assert "target page title is deliberately hidden" in prompt
    assert "Registered knowledge-cutoff snapshot" in prompt
    assert result["target_title_revealed"] is False


def test_temporal_dates_must_include_case_before_and_target():
    case = {
        "id": "temporal-case", "wikipedia_before": "2024-01-01",
        "wikipedia_as_of": "2025-01-01",
    }
    try:
        _snapshot_values("2023-01-01,2025-01-01", case)
    except ValueError as exc:
        assert "before snapshot '2024-01-01' must be included" in str(exc)
    else:
        raise AssertionError("missing before snapshot was accepted")


def test_temporal_judge_receives_target_and_visible_evidence():
    captured = {}

    def fake_call(model, messages, temperature=0.0):
        captured["prompt"] = messages[-1]["content"]
        return json.dumps({
            "decision": "correct_after", "confidence": 0.99,
            "answer_extracted": "N. Person", "evidence": "New Person",
            "reason": "matches target revision",
        })

    earlier = _page("Earlier Page", 10, "Earlier content " + "x" * 30_000).to_dict()
    earlier["as_of"] = "2024-01-01"
    target = _page("Pivot Page", 20, "The leader is New Person.").to_dict()
    target["as_of"] = "2025-01-01"
    result = LLMJudge("independent/judge", call_model_fn=fake_call).judge_temporal_answer(
        "Who is the leader?", "N. Person", ["New Person"], ["Old Person"],
        [earlier, target],
        target_snapshot_as_of="2025-01-01",
    )
    assert result.decision == "correct_after"
    assert "Target snapshot requested by the task: 2025-01-01" in captured["prompt"]
    assert "The leader is New Person." in captured["prompt"]
    assert captured["prompt"].index("PAGE Pivot Page") < captured["prompt"].index(
        "PAGE Earlier Page"
    )
    assert "as_of 2025-01-01" in captured["prompt"]


def test_temporal_judge_cannot_invent_an_answer_for_blank_response():
    def must_not_call(*args, **kwargs):
        raise AssertionError("blank responses must fail before the LLM judge")

    result = LLMJudge(
        "independent/judge", call_model_fn=must_not_call,
    ).judge_temporal_answer(
        "Who is the leader?", "   ", ["New Person"], ["Old Person"], [],
        target_snapshot_as_of="2025-01-01",
    )
    assert result.decision == "no_answer"
    assert result.confidence == 1.0
    assert result.answer_extracted == ""
    assert result.raw == {"deterministic_gate": "blank_response"}


def test_temporal_judge_positive_answer_must_come_from_tested_response():
    def hallucinating_judge(model, messages, temperature=0.0):
        return json.dumps({
            "decision": "correct_after", "confidence": 1,
            "answer_extracted": "New Person", "evidence": "New Person",
            "reason": "matches the gold alias",
        })

    result = LLMJudge(
        "independent/judge", call_model_fn=hallucinating_judge,
    ).judge_temporal_answer(
        "Who is the leader?", "I do not know.", ["New Person"],
        ["Old Person"], [], target_snapshot_as_of="2025-01-01",
    )
    assert result.decision == "unjudgeable"
    assert result.raw["original_decision"] == "correct_after"
    assert result.raw["contract_violation"] == (
        "answer_extracted_not_in_tested_response"
    )


def test_judge_extracted_answer_allows_markdown_emphasis_in_tested_response():
    def judge_call(model, messages, temperature=0.0):
        return json.dumps({
            "decision": "stick_old", "confidence": 1,
            "answer_extracted": "The answer is Old Person.",
            "evidence": "The answer is Old Person.",
            "reason": "matches the stale alias",
        })

    result = LLMJudge(
        "independent/judge", call_model_fn=judge_call,
    ).judge_answer(
        "Who?", "The answer is **Old Person**.",
        ["New Person"], ["Old Person"], [],
    )
    assert result.decision == "stick_old"
    assert "contract_violation" not in result.raw


def test_negated_or_quoted_aliases_never_use_deterministic_positive_gate():
    calls = []

    def judge_call(model, messages, temperature=0.0):
        calls.append(messages[-1]["content"])
        return json.dumps({
            "decision": "unsupported", "confidence": 1.0,
            "answer_extracted": "New Person", "evidence": "",
            "reason": "The response explicitly rejects the alias.",
        })

    judge = LLMJudge("independent/judge", call_model_fn=judge_call)
    for response in (
        "It is not New Person.",
        "The page mentions 'New Person', but that is not the answer.",
        "New Person is incorrect.",
    ):
        result = judge.judge_answer(
            "Who?", response, ["New Person"], ["Old Person"], []
        )
        assert result.decision == "unsupported"
        assert "deterministic_gate" not in result.raw
    assert len(calls) == 3


def test_llm_cannot_emit_stale_label_when_no_stale_alias_is_configured():
    def invalid_judge(model, messages, temperature=0.0):
        return json.dumps({
            "decision": "stick_old", "confidence": 1,
            "answer_extracted": "midfielder", "evidence": "",
            "reason": "incorrectly treated an unrelated answer as stale",
        })

    result = LLMJudge(
        "independent/judge", call_model_fn=invalid_judge,
    ).judge_answer("What position?", "midfielder", ["Defender"], [], [])
    assert result.decision == "unsupported"
    assert result.raw["original_decision"] == "stick_old"
    assert result.raw["contract_violation"] == (
        "positive_label_without_configured_aliases"
    )


def test_factorized_pk_judge_cannot_mistake_context_entity_for_bridge_answer():
    def confused_judge(model, messages, temperature=0.0):
        return json.dumps({
            "decision": "known", "confidence": 1,
            "answer_extracted": "David Lammy", "evidence": "David Lammy",
            "reason": "incorrectly selected an intermediate person",
        })

    result = LLMJudge(
        "independent/judge", call_model_fn=confused_judge,
    ).judge_prior_knowledge(
        probe_role="critical_bridge",
        question=(
            "Who began holding Foreign Secretary on 2025-09-05, after "
            "David Lammy's tenure?"
        ),
        response="David Lammy held it before the later appointment.",
        answer_aliases=["Yvette Cooper"],
    )

    assert result.decision == "unjudgeable"
    assert result.raw["contract_violation"] == "known_answer_not_in_probe_aliases"


def test_factorized_composed_judge_fail_closes_on_intermediate_as_final():
    def confused_judge(model, messages, temperature=0.0):
        return json.dumps({
            "decision": "wrong", "confidence": 1,
            "answer_extracted": "David Lammy", "evidence": "David Lammy",
            "reason": "incorrectly treated the first-hop answer as final",
        })

    result = LLMJudge(
        "independent/judge", call_model_fn=confused_judge,
    ).judge_prior_knowledge(
        probe_role="composed", question="Who is the final person's spouse?",
        response="David Lammy is the first step, but I do not know the spouse.",
        answer_aliases=["Ed Balls"], context_aliases=["David Lammy"],
    )

    assert result.decision == "unjudgeable"
    assert result.raw["contract_violation"] == "wrong_answer_is_context_alias"


def test_pk_admission_uses_fresh_context_and_rejects_known_target_answer():
    case = {
        "id": "temporal-case", "temporal_question": "Who is the leader?",
        "old_answer_keywords": ["Old Person"],
        "new_answer_keywords": ["New Person"],
    }
    calls = []
    temperatures = []

    def stale_probe(model, messages, temperature=0.0):
        calls.append(list(messages))
        temperatures.append(temperature)
        return "Old Person"

    with tempfile.TemporaryDirectory() as tmp:
        output = os.path.join(tmp, "pk_results.jsonl")
        store = JsonlResultStore(output)
        admitted = run_pk_admission(
            case=case, model="tested/model", target_as_of="2025-01-01",
            judge=_MockJudge(answer="stick_old"), store=store,
            repeats=3, max_known_rate=0.0, probe_call_model_fn=stale_probe,
        )
        rejected = run_pk_admission(
            case={**case, "id": "known-case"}, model="tested/model",
            target_as_of="2025-01-01", judge=_MockJudge(answer="stick_new"),
            store=store, repeats=3, max_known_rate=0.0,
            probe_call_model_fn=lambda *args, **kwargs: "New Person",
        )
        store.close()
        rows = [json.loads(line) for line in open(output, encoding="utf-8")]
    assert admitted["passed"] is True
    assert admitted["stick_old_count"] == 3 and admitted["stick_new_count"] == 0
    assert rejected["passed"] is False
    assert rejected["reason"] == "already_knows_target_answer"
    assert len(calls) == 3
    assert temperatures == [0.0, 0.2, 0.5]
    assert all(len(messages) == 1 and messages[0]["role"] == "user" for messages in calls)
    tested_prompts = [messages[0]["content"] for messages in calls]
    assert all("Question: Who is the leader?" in prompt for prompt in tested_prompts)
    assert all("Target date" not in prompt for prompt in tested_prompts)
    assert all("knowledge-cutoff" not in prompt for prompt in tested_prompts)
    probes = [row for row in rows if row["slot"] == "pk_probe"]
    assert len(probes) == 6
    assert all(row["fresh_context"] and not row["tools_available"] for row in probes)
    assert [row["probe_temperature"] for row in probes[:3]] == [0.0, 0.2, 0.5]


def test_factorized_pk_admission_is_controlled_only_by_critical_bridge():
    case = {
        "id": "factorized-case",
        "temporal_question": "Who is the spouse reached through the later appointment?",
        "knowledge_cutoff": {"cutoff_date": "2024-06-01"},
        "prior_knowledge_contract": {
            "schema_version": "factorized-prior-knowledge-v2",
            "primary_admission_role": "critical_bridge",
            "primary_admission_probe_ids": ["bridge_2"],
            "probes": [
                {
                    "id": "bridge_2", "role": "critical_bridge",
                    "objective": "must_be_unknown",
                    "question": (
                        "Who began holding Foreign Secretary on 2025-09-05, "
                        "after David Lammy's tenure?"
                    ),
                    "answer_aliases": ["Yvette Cooper"],
                },
                {
                    "id": "tail", "role": "tail",
                    "objective": "measure_known_for_composition",
                    "question": "Who is Yvette Cooper's spouse?",
                    "answer_aliases": ["Ed Balls"],
                },
                {
                    "id": "composed", "role": "composed",
                    "objective": "diagnostic_only",
                    "question": "Who is the spouse reached through the later appointment?",
                    "answer_aliases": ["Ed Balls"],
                },
            ],
        },
    }

    class FactorizedJudge:
        def judge_prior_knowledge(
            self, *, probe_role, question, response, answer_aliases,
            context_aliases=None,
        ):
            if response.strip() in answer_aliases:
                return JudgeResult(
                    "known", 1.0, "exact", response, response,
                    {"deterministic_gate": "test_alias"},
                )
            return JudgeResult("unknown", 1.0, "explicitly unknown")

    def probe(model, messages, temperature=0.0):
        prompt = messages[0]["content"]
        if "Who is Yvette Cooper's spouse?" in prompt:
            return "Ed Balls"
        if "Who is the spouse reached through the later appointment?" in prompt:
            return "David Lammy is mentioned, but I don't know the final answer."
        return "I don't know who succeeded David Lammy."

    with tempfile.TemporaryDirectory() as tmp:
        output = os.path.join(tmp, "factorized_pk.jsonl")
        store = JsonlResultStore(output)
        gate = run_pk_admission(
            case=case, model="tested/model", target_as_of="2025-09-05",
            judge=FactorizedJudge(), store=store, repeats=2,
            max_known_rate=0.0, temperatures=[0.0, 0.2],
            probe_call_model_fn=probe,
        )
        store.close()
        rows = [json.loads(line) for line in open(output, encoding="utf-8")]

    assert gate["passed"] is True
    assert gate["reason"] == "critical_bridge_not_known"
    assert gate["critical_bridge_known_rate"] == 0.0
    assert gate["tail_known_rate"] == 1.0
    assert gate["composed_known_rate"] == 0.0
    assert gate["clean_bridge_composition_candidate"] is True
    assert gate["pk_contract_version"] == "factorized-prior-knowledge-v2"
    probes = [row for row in rows if row["slot"] == "pk_probe"]
    assert len(probes) == 6
    assert {row["probe_role"] for row in probes} == {
        "critical_bridge", "tail", "composed",
    }
    composed = [row for row in probes if row["probe_role"] == "composed"]
    assert all(row["label"] == "unknown" for row in composed)
    forbidden = (
        "Probe role", "Probe objective", "must_be_unknown", "critical_bridge",
        "Registered knowledge-cutoff", "Target date", "admission",
    )
    assert all(
        not any(value in row["prompt"] for value in forbidden)
        for row in probes
    )


def test_acquisition_evidence_requires_every_exact_critical_bridge_revision():
    case = {
        "prior_knowledge_contract": {
            "primary_admission_probe_ids": ["bridge_1", "bridge_2"],
            "probes": [
                {"id": "bridge_1", "objective": "must_be_unknown", "hop_index": 0},
                {"id": "bridge_2", "objective": "must_be_unknown", "hop_index": 1},
            ],
        },
        "reasoning_chain": [
            {"source_title": "A", "source_revision_id": 10,
             "evidence": "A appointed B."},
            {"source_title": "B", "source_revision_id": 20,
             "evidence": "B was succeeded by C."},
        ],
    }
    result = {"evidence_pages": [
        {"title": "A", "revision_id": 10, "content": "A appointed B."},
        {"title": "B", "revision_id": 19, "content": "B was succeeded by C."},
    ]}
    partial = _critical_bridge_evidence_metrics(result, case)
    assert partial["critical_bridges_evidenced"] == 1
    assert partial["critical_bridge_evidence_complete"] is False
    result["evidence_pages"][1]["revision_id"] = 20
    complete = _critical_bridge_evidence_metrics(result, case)
    assert complete["critical_bridges_evidenced"] == 2
    assert complete["critical_bridge_evidence_complete"] is True
    assert _failure_mode({}, "correct_after", "answered", False) == (
        "correct_without_critical_bridge_evidence"
    )


def test_primary_runner_has_one_question_and_no_ripple_or_control_rounds():
    case = {
        "id": "temporal-case", "wikipedia_title": "Pivot Page",
        "temporal_question": "Who is the leader?",
        "wikipedia_before": "2024-01-01",
        "wikipedia_as_of": "2025-01-01",
        "old_answer_keywords": ["Old Person"],
        "new_answer_keywords": ["New Person"],
    }
    with tempfile.TemporaryDirectory() as tmp:
        output = os.path.join(tmp, "temporal_results.jsonl")
        store = JsonlResultStore(output)
        backend = _TemporalBackend()
        navigation = temporal_reverse_bfs(
            backend, "Pivot Page", "2025-01-01",
            ["2024-01-01", "2025-01-01"], 1, branch_cap=10,
        )
        attempt_id = run_temporal_case(
            case=case, model="tested/model", repeat=0, backend=backend,
            judge=_MockJudge(), store=store,
            snapshot_dates=["2024-01-01", "2025-01-01"],
            navigation=navigation, start_distance=1,
            max_steps=6, temperature=0.7,
            browse_call_model_fn=_ScriptedShortestModel(),
        )
        store.close()
        rows = [json.loads(line) for line in open(output, encoding="utf-8")]
    assert attempt_id
    assert {row["slot"] for row in rows} == {
        "temporal_step", "temporal_summary", "final_judgment",
    }
    assert not any(row["slot"] in {
        "pk_probe", "pk_gate", "followup", "answerability_gate", "distractor",
    } for row in rows)
    judgment = next(row for row in rows if row["slot"] == "final_judgment")
    assert judgment["label"] == "correct_after"
    assert judgment["target_snapshot_as_of"] == "2025-01-01"
    summary = next(row for row in rows if row["slot"] == "temporal_summary")
    assert summary["start_title"] == "Source Page"
    assert summary["pivot_hit"] is True
    assert summary["shortest_navigation_steps"] == 2
    assert summary["actual_steps_to_first_pivot"] == 2
    assert summary["detour_steps"] == 0
    assert summary["shortest_arrival"] is True
    assert summary["agent_initial_messages"][0]["role"] == "system"
    assert summary["agent_tool_contract"][0]["function"]["name"] == "switch_snapshot"


def test_persisted_temporal_e2e_writes_jsonl_scores_and_viewer():
    case = {
        "id": "e2e-case", "wikipedia_title": "Pivot Page",
        "temporal_question": "Who is the leader?",
        "wikipedia_before": "2024-01-01", "wikipedia_as_of": "2025-01-01",
        "old_answer_keywords": ["Old Person"],
        "new_answer_keywords": ["New Person"],
    }
    with tempfile.TemporaryDirectory() as tmp:
        results = os.path.join(tmp, "e2e_results.jsonl")
        scores = os.path.join(tmp, "e2e_scores.csv")
        cache = os.path.join(tmp, "e2e_snapshot.db")
        viewer = os.path.join(tmp, "viewer")
        backend = _TemporalBackend()
        navigation = temporal_reverse_bfs(
            backend, "Pivot Page", "2025-01-01",
            ["2024-01-01", "2025-01-01"], 1, branch_cap=10,
        )
        store = JsonlResultStore(results, metadata={
            "contract_hash": "e2e-contract",
            "schema_version": "temporal-pk-relative-multihop-v8",
        })
        store.write(
            slot="pk_gate", case_id=case["id"], model="tested/model",
            arm="admission", passed=True, reason="target_answer_not_known",
            n=1, stick_new_count=0, stick_old_count=1,
        )
        attempt_id = run_temporal_case(
            case=case, model="tested/model", repeat=0, backend=backend,
            judge=_MockJudge(), store=store,
            snapshot_dates=["2024-01-01", "2025-01-01"],
            navigation=navigation, start_distance=1, max_steps=6,
            temperature=0.7, browse_call_model_fn=_ScriptedShortestModel(),
        )
        assert attempt_id is not None
        store.write(
            slot="checkpoint", case_id=case["id"], model="tested/model",
            arm="temporal", repeat=0, status="complete",
            attempt_id=attempt_id, navigation_id="e2e-nav",
        )
        store.close()
        write_temporal_scores(results, scores, "e2e-contract")

        conn = sqlite3.connect(cache)
        conn.execute(
            "CREATE TABLE page_cache (cache_key TEXT PRIMARY KEY, title TEXT, "
            "as_of TEXT, data TEXT, fetched_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE page_links (source_key TEXT, target_folded TEXT, "
            "target_title TEXT)"
        )
        for title in ("Source Page", "Pivot Page"):
            for as_of in ("2024-01-01", "2025-01-01"):
                page = backend.fetch_page(title, as_of=as_of)
                key = f"en|{title.casefold()}|{as_of}"
                conn.execute(
                    "INSERT INTO page_cache VALUES (?,?,?,?,?)",
                    (key, title, as_of, json.dumps(page.to_dict()), page.timestamp),
                )
        conn.commit()
        conn.close()
        data = build_visualization_data(cache, results)
        site = build_site(data, viewer)

        score_rows = list(csv.DictReader(open(scores, encoding="utf-8")))
        result_rows = [json.loads(line) for line in open(results, encoding="utf-8")]
        viewer_payload = json.loads((site / "data.json").read_text(encoding="utf-8"))
    assert score_rows[0]["n"] == "1"
    assert any(row["slot"] == "checkpoint" for row in result_rows)
    assert len(viewer_payload["trajectories"]) == 1
    assert viewer_payload["trajectories"][0]["outcome_reason"] == "correct_after"


def test_temporal_scores_report_shortest_path_and_cycle_diagnostics():
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "results.jsonl")
        output = os.path.join(tmp, "scores.csv")
        rows = [
            {"slot": "pk_gate", "contract_hash": "c", "model": "m", "case_id": "q",
             "n": 3, "passed": True, "reason": "target_answer_not_known",
             "stick_new_count": 0, "stick_new_rate": 0.0,
             "stick_old_count": 3, "stick_old_rate": 1.0, "other_count": 0},
            {"slot": "temporal_summary", "contract_hash": "c", "attempt_id": "a",
             "model": "m", "case_id": "q", "pivot_hit": True,
             "shortest_arrival": False, "detour_steps": 2, "cycle_detected": True,
             "semantic_route_complete": True, "semantic_completion_rate": 0.75},
            {"slot": "final_judgment", "contract_hash": "c", "attempt_id": "a",
             "model": "m", "case_id": "q", "label": "old_snapshot_answer"},
            {"slot": "checkpoint", "contract_hash": "c", "attempt_id": "a",
             "status": "complete"},
        ]
        with open(source, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        write_temporal_scores(source, output, "c")
        with open(output, encoding="utf-8") as fh:
            score = next(csv.DictReader(fh))
    assert score["pivot_hit"] == "1"
    assert score["pk_admitted"] == "True"
    assert score["pk_stick_new"] == "0"
    assert score["pk_stick_old_rate_pct"] == "100.0"
    assert score["shortest_arrival"] == "0"
    assert score["mean_detour_steps_on_hit"] == "2.0"
    assert score["cycle_detected"] == "1"
    assert score["found_but_wrong"] == "1"
    assert score["semantic_route_complete"] == "1"
    assert score["semantic_route_complete_rate_pct"] == "100.0"
    assert score["mean_semantic_completion_pct"] == "75.0"


def test_temporal_scores_fail_closed_on_legacy_judge_hallucinating_blank_answer():
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "results.jsonl")
        output = os.path.join(tmp, "scores.csv")
        rows = [
            {"slot": "pk_gate", "contract_hash": "c", "model": "m",
             "case_id": "q", "n": 3, "passed": True},
            {"slot": "temporal_summary", "contract_hash": "c",
             "attempt_id": "a", "model": "m", "case_id": "q",
             "final_answer": "", "pivot_hit": False,
             "semantic_route_complete": False, "semantic_completion_rate": 0.4},
            {"slot": "final_judgment", "contract_hash": "c",
             "attempt_id": "a", "model": "m", "case_id": "q",
             "label": "correct_after"},
            {"slot": "checkpoint", "contract_hash": "c", "attempt_id": "a",
             "status": "complete"},
        ]
        with open(source, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        write_temporal_scores(source, output, "c")
        with open(output, encoding="utf-8") as fh:
            score = next(csv.DictReader(fh))
    assert score["correct_after"] == "0"
    assert score["no_answer"] == "1"
    assert score["no_answer_rate_pct"] == "100.0"
    assert score["blank_answer_judge_overrides"] == "1"


def test_browser_page_hit_means_page_was_rendered():
    result = run_wikipedia_browsing(
        "tested/model", _MockBackend(), "Source Page", 3, "neutral",
        target_title="Pivot Page", as_of="2025-01-01",
        call_model_fn=_ScriptedToolModel(), verbose=False,
    )
    assert result["page_hit"] is True
    assert result["stop_reason"] == "target_page_opened"
    assert result["evidence_pages"][-1]["title"] == "Pivot Page"
    assert "updated leader is New Person" in result["trajectory"][-1]["result"]
    assert 'frozen at or before "2025-01-01"' in result["messages"][1]["content"]


def test_judge_evidence_excludes_hidden_truncated_text():
    backend = _MockBackend()
    backend.pages["Pivot Page"].content = "A" * 16_100 + "HIDDEN_SECRET"
    result = run_wikipedia_browsing(
        "tested/model", backend, "Source Page", 3, "neutral",
        target_title="Pivot Page", as_of="2025-01-01",
        call_model_fn=_ScriptedToolModel(), verbose=False,
    )
    assert "HIDDEN_SECRET" not in result["evidence_pages"][-1]["content"]
    assert "Page truncated" in result["evidence_pages"][-1]["content"]


def test_exposure_requires_visible_and_comprehended():
    result = run_wikipedia_browsing(
        "tested/model", _MockBackend(), "Source Page", 3, "neutral",
        target_title="Pivot Page", as_of="2025-01-01",
        call_model_fn=_ScriptedToolModel(), verbose=False,
    )
    case = {"pk_question": "Who is the leader?", "new_answer_keywords": ["New Person"],
            "old_answer_keywords": ["Old Person"]}
    gate = evaluate_exposure(
        page_hit=True, case=case, pages=result["evidence_pages"], messages=result["messages"],
        tested_model="tested/model", judge=_MockJudge(),
        call_model_fn=lambda *args, **kwargs: "New Person",
    )
    assert gate.eligible and gate.pivot_visible and gate.pivot_comprehended

    ambiguous = evaluate_exposure(
        page_hit=True, case=case, pages=result["evidence_pages"], messages=result["messages"],
        tested_model="tested/model", judge=_MockJudge(visibility="ambiguous"),
        call_model_fn=lambda *args, **kwargs: "New Person",
    )
    assert not ambiguous.eligible
    assert "pivot_ambiguous" in ambiguous.failure_reasons


def test_answerability_gate_and_llm_judge_json():
    item = {"question": "Who is the leader?", "new_keywords": ["New Person"]}
    decision = answerability_for_item(_MockJudge(), item, [_page(
        "Pivot Page", 2, "The updated leader is New Person."
    ).to_dict()], "conflict")
    assert decision.decision == "answerable"

    judge_calls = []

    def fake_call(model, messages, temperature=0.0):
        judge_calls.append(messages)
        return """```json
        {"decision":"stick_new","confidence":0.97,"answer_extracted":"New Person",
         "evidence":"updated leader","reason":"matches"}
        ```"""

    judged = LLMJudge("independent/judge", call_model_fn=fake_call).judge_answer(
        "Who?", "New Person", ["New Person"], ["Old Person"], []
    )
    assert judged.decision == "stick_new" and judged.confidence == 1.0
    assert judged.raw["deterministic_gate"] == "exact_new_alias"
    assert judge_calls == []

    low = LLMJudge(
        "independent/judge",
        call_model_fn=lambda *args, **kwargs: json.dumps({
            "decision": "stick_new", "confidence": 0.2, "reason": "weak"
        }),
    ).judge_answer("Who?", "N. Person", ["New Person"], ["Old Person"], [])
    assert low.decision == "unjudgeable"
    assert low.raw["low_confidence_decision"] == "stick_new"
    assert transition_label("stick_new", "stick_old") == "reversion"
    assert transition_label("stick_old", "stick_new") == "recovery"

    temporal = LLMJudge(
        "independent/judge",
        call_model_fn=lambda *args, **kwargs: json.dumps({
            "decision": "correct_after", "confidence": 0.96,
            "answer_extracted": "New Person", "evidence": "updated leader",
            "reason": "matches later revision",
        }),
    ).judge_temporal_answer(
        "Who?", "New Person", ["New Person"], ["Old Person"], []
    )
    assert temporal.decision == "correct_without_visible_support"
    assert temporal.raw["deterministic_gate"] == "exact_new_alias_without_target_support"


def test_result_paths_protect_existing_artifacts():
    try:
        assert_new_output_path("results.jsonl")
        raise AssertionError("protected result path should have failed")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        output = os.path.join(tmp, "wikipedia_results.jsonl")
        store = JsonlResultStore(output)
        store.write(slot="checkpoint", case_id="c", model="m", arm="conflict",
                    start_distance=1, start_title="S", repeat=0, status="complete")
        selection = {
            "selection_id": "selection-1", "status": "selected",
            "selected_as_of": "2025-01-01", "messages": [],
        }
        store.write(slot="temporal_selection", case_id="c", model="m", arm="shared",
                    contract_hash="contract", selection=selection)
        store.close()
        assert JsonlResultStore.completed(output) == {("c", "m", "conflict", 1, "S", 0)}
        assert JsonlResultStore.latest_snapshot_selection(
            output, "c", "m", "contract"
        ) == selection


def test_runner_joins_browser_gates_judge_and_followups():
    case = {
        "id": "case", "pk_question": "Who is the leader?",
        "new_answer_keywords": ["New Person"], "old_answer_keywords": ["Old Person"],
        "ripples": {"1": [{
            "question": "Who is the leader?", "paraphrases": [],
            "new_keywords": ["New Person"], "old_keywords": ["Old Person"],
        }]},
        "control": {},
    }
    args = SimpleNamespace(
        max_steps=3, temperature=0.7, distractors=["irrelevant?"],
        rounds=2, distractor_every=0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        output = os.path.join(tmp, "wikipedia_results.jsonl")
        store = JsonlResultStore(output)
        completed_attempt_id = run_trajectory(
            case=case, raw_case=case, arm="conflict", model="tested/model",
            backend=_MockBackend(), target="Pivot Page", as_of="2025-01-01",
            start_title="Source Page", distance=1, repeat=0, args=args,
            judge=_MockJudge(), store=store, rng=random.Random(0),
            browse_call_model_fn=_ScriptedToolModel(),
            plain_call_model_fn=lambda *args, **kwargs: "New Person",
        )
        store.close()
        assert isinstance(completed_attempt_id, str) and completed_attempt_id
        rows = [json.loads(line) for line in open(output, encoding="utf-8")]
        summary = next(row for row in rows if row["slot"] == "browse_summary")
        assert summary["snapshot_as_of"] == "2025-01-01"
        assert summary["target_title"] == "Pivot Page"
        assert any(row["slot"] == "exposure_gate" and row["gate"]["eligible"] for row in rows)
        followups = [row for row in rows if row["slot"] == "followup"]
        assert [row["transition"] for row in followups] == ["initial", "stable_new"]


def test_analysis_writes_gate_transition_and_control_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "input.jsonl")
        rows = [
            {"slot": "browse_summary", "model": "m", "case_id": "c", "arm": "conflict",
             "start_distance": 1, "page_hit": True, "attempt_id": "a1"},
            {"slot": "exposure_gate", "model": "m", "case_id": "c", "arm": "conflict",
             "start_distance": 1, "gate": {"eligible": True}, "attempt_id": "a1"},
            {"slot": "followup", "model": "m", "case_id": "c", "arm": "conflict",
             "distance": 1, "occurrence": 1, "label": "stick_old", "transition": "reversion",
             "attempt_id": "a1"},
            {"slot": "followup", "model": "m", "case_id": "c", "arm": "control",
             "distance": 1, "occurrence": 1, "label": "stick_new", "transition": "stable_new",
             "attempt_id": "a2"},
            # Partial failed retry: must not enter analysis without a checkpoint.
            {"slot": "followup", "model": "m", "case_id": "c", "arm": "conflict",
             "distance": 1, "occurrence": 1, "label": "stick_new", "transition": "stable_new",
             "attempt_id": "failed"},
            {"slot": "checkpoint", "model": "m", "case_id": "c", "arm": "conflict",
             "status": "complete", "attempt_id": "a1"},
            {"slot": "checkpoint", "model": "m", "case_id": "c", "arm": "control",
             "status": "complete", "attempt_id": "a2"},
        ]
        with open(source, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        outputs = [os.path.join(tmp, name) for name in
                   ("wiki-summary.csv", "wiki-diff.csv", "wiki-gates.csv", "wiki-transitions.csv")]
        old_argv = sys.argv
        sys.argv = ["analyze", "--input", source, "--summary-output", outputs[0],
                    "--diff-output", outputs[1], "--gate-output", outputs[2],
                    "--transition-output", outputs[3]]
        try:
            analyze_main()
        finally:
            sys.argv = old_argv
        assert all(os.path.exists(path) for path in outputs)
        assert "significant_reversion" in open(outputs[1], encoding="utf-8").readline()
        assert "reversion" in open(outputs[3], encoding="utf-8").readline()
        with open(outputs[0], encoding="utf-8") as fh:
            summary = list(csv.DictReader(fh))
        assert all(row["n"] == "1" for row in summary), "partial retry leaked into analysis"


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"[OK] {test.__name__}")
    print(f"All {len(tests)} Wikipedia pipeline tests passed.")


if __name__ == "__main__":
    main()
