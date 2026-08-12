"""Offline tests for the Wikipedia graph and trajectory visualizer."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile

from tkg.experiment.contracts import PageLink, PageSnapshot
from tkg.visualization.exporter import _trajectory_outcome, build_visualization_data
from tkg.visualization.server import build_site


def _snapshot(title: str, page_id: int, revision: int, links=()):
    return PageSnapshot(
        title=title,
        page_id=page_id,
        revision_id=revision,
        timestamp="2025-01-01T00:00:00Z",
        as_of="2025-01-01",
        content=f"Visible content for {title}.",
        links=[PageLink(target=target, anchor=anchor) for target, anchor in links],
        source_url=f"https://en.wikipedia.org/?oldid={revision}",
    )


def _write_cache(path: str):
    pages = [
        ("en|page a|2025-01-01", _snapshot("Page A", 1, 11, [
            ("Page B", "next page"), ("Unfetched Page", "external page")
        ])),
        ("en|page b|2025-01-01", _snapshot("Page B", 2, 22, [
            ("Page C", "second layer")
        ])),
        ("en|page c|2025-01-01", _snapshot("Page C", 3, 33)),
    ]
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE page_cache (cache_key TEXT PRIMARY KEY, title TEXT, as_of TEXT, "
        "data TEXT, fetched_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE page_links (source_key TEXT, target_folded TEXT, target_title TEXT)"
    )
    for key, page in pages:
        conn.execute(
            "INSERT INTO page_cache VALUES (?,?,?,?,?)",
            (key, page.title, page.as_of, json.dumps(page.to_dict()), page.timestamp),
        )
    conn.commit()
    conn.close()


def _write_results(path: str):
    rows = [
        {
            "slot": "temporal_selection", "model": "model/a", "case_id": "case-a",
            "arm": "shared", "selection": {
                "selection_id": "selection-1", "status": "selected",
                "snapshot_mode": "agent_selected",
                "allowed_as_of": ["2024-01-01", "2025-01-01"],
                "selected_as_of": "2025-01-01", "selection_token": "2025-01-01",
                "intent_code": "latest_available",
                "brief_reason": "I want the newest allowed snapshot.",
                "attempts": [], "messages": [],
            },
        },
        {
            "slot": "browse_step", "attempt_id": "attempt-1", "trajectory_key": "t1",
            "model": "model/a", "case_id": "case-a", "arm": "conflict",
            "step": 1, "from_title": "Page A", "action": "follow_link",
            "args": {"target": "Page B"}, "result": "Wikipedia page: Page B",
        },
        {
            "slot": "browse_summary", "attempt_id": "attempt-1", "trajectory_key": "t1",
            "model": "model/a", "case_id": "case-a", "arm": "conflict",
            "start_distance": 1, "start_title": "Page A", "repeat": 0,
            "target_title": "Page B",
            "snapshot_as_of": "2025-01-01",
            "snapshot_selection_id": "selection-1",
            "target_snapshot_as_of": "2025-01-01",
            "final_title": "Page B", "page_hit": True, "stop_reason": "target_page_opened",
            "visited_titles": ["Page A", "Page B"],
            "evidence_revisions": [
                {"title": "Page A", "revision_id": 11},
                {"title": "Page B", "revision_id": 22},
            ],
        },
        {
            "slot": "exposure_gate", "attempt_id": "attempt-1", "trajectory_key": "t1",
            "gate": {"eligible": True, "pivot_visible": True,
                     "pivot_comprehended": True, "failure_reasons": []},
        },
        {
            "slot": "followup", "attempt_id": "attempt-1", "trajectory_key": "t1",
            "label": "stick_new", "transition": "stable_new",
        },
        {
            "slot": "checkpoint", "attempt_id": "attempt-1", "trajectory_key": "t1",
            "status": "complete",
        },
    ]
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _write_temporal_cache(path: str):
    pages = [
        ("en|pivot|2024-01-01", PageSnapshot(
            title="Pivot", page_id=1, revision_id=101,
            timestamp="2024-01-01T00:00:00Z", as_of="2024-01-01",
            content="The leader is Old Person.",
            links=[PageLink(target="Old Neighbor", anchor="old")],
        )),
        ("en|pivot|2025-01-01", PageSnapshot(
            title="Pivot", page_id=1, revision_id=202,
            timestamp="2025-01-01T00:00:00Z", as_of="2025-01-01",
            content="The leader is New Person.",
            links=[PageLink(target="New Neighbor", anchor="new")],
        )),
        ("en|source|2025-01-01", PageSnapshot(
            title="Source", page_id=2, revision_id=303,
            timestamp="2025-01-01T00:00:00Z", as_of="2025-01-01",
            content="Follow the pivot.",
            links=[PageLink(target="Pivot", anchor="pivot")],
        )),
    ]
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE page_cache (cache_key TEXT PRIMARY KEY, title TEXT, as_of TEXT, "
        "data TEXT, fetched_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE page_links (source_key TEXT, target_folded TEXT, target_title TEXT)"
    )
    for key, page in pages:
        conn.execute(
            "INSERT INTO page_cache VALUES (?,?,?,?,?)",
            (key, page.title, page.as_of, json.dumps(page.to_dict()), page.timestamp),
        )
    conn.commit()
    conn.close()


def _write_temporal_results(path: str):
    rows = [
        {
            "slot": "pk_gate", "case_id": "case-t", "model": "model/t",
            "arm": "admission", "passed": True, "reason": "target_answer_not_known",
            "n": 3, "stick_new_count": 0, "stick_old_count": 3,
        },
        {
            "slot": "navigation_arena", "case_id": "case-t", "model": "__shared__",
            "arm": "temporal", "navigation_id": "nav-1", "states": [
                {"title": "Pivot", "revision_id": 202, "as_of": "2025-01-01",
                 "distance_to_pivot": 0},
                {"title": "Pivot", "revision_id": 101, "as_of": "2024-01-01",
                 "distance_to_pivot": 1},
                {"title": "Source", "revision_id": 303, "as_of": "2025-01-01",
                 "distance_to_pivot": 1},
            ],
        },
        {
            "slot": "temporal_step", "attempt_id": "temporal-1", "model": "model/t",
            "case_id": "case-t", "arm": "temporal", "step": 1,
            "from_title": "Source", "to_title": "Source", "action": "switch_snapshot",
            "args": {"as_of": "2025-01-01"}, "result": "source revision",
            "from_snapshot_token": None, "snapshot_token": "2025-01-01",
            "from_revision_id": None, "revision_id": 303,
            "navigation_step": 1, "distance_to_pivot": 1, "revisited": False,
        },
        {
            "slot": "temporal_step", "attempt_id": "temporal-1", "model": "model/t",
            "case_id": "case-t", "arm": "temporal", "step": 2,
            "from_title": "Source", "to_title": "Pivot", "action": "follow_link",
            "args": {"target": "Pivot"}, "result": "new pivot revision",
            "from_snapshot_token": "2025-01-01", "snapshot_token": "2025-01-01",
            "from_revision_id": 303, "revision_id": 202,
            "navigation_step": 2, "distance_to_pivot": 0, "revisited": False,
        },
        {
            "slot": "temporal_step", "attempt_id": "temporal-1", "model": "model/t",
            "case_id": "case-t", "arm": "temporal", "step": 3,
            "from_title": "Pivot", "to_title": "Pivot", "action": "switch_snapshot",
            "args": {"as_of": "2024-01-01"}, "result": "old revision",
            "from_snapshot_token": "2025-01-01", "snapshot_token": "2024-01-01",
            "from_revision_id": 202, "revision_id": 101,
            "navigation_step": 3, "distance_to_pivot": 1, "revisited": False,
        },
        {
            "slot": "temporal_step", "attempt_id": "temporal-1", "model": "model/t",
            "case_id": "case-t", "arm": "temporal", "step": 4,
            "from_title": "Pivot", "to_title": "Pivot", "action": "switch_snapshot",
            "args": {"as_of": "2025-01-01"}, "result": "new revision",
            "from_snapshot_token": "2024-01-01", "snapshot_token": "2025-01-01",
            "from_revision_id": 101, "revision_id": 202,
            "navigation_step": 4, "distance_to_pivot": 0, "revisited": True,
        },
        {
            "slot": "temporal_step", "attempt_id": "temporal-1", "model": "model/t",
            "case_id": "case-t", "arm": "temporal", "step": 5,
            "from_title": "Pivot", "to_title": "Pivot", "action": "submit_answer",
            "args": {"answer": "New Person"}, "result": "Final answer submitted.",
            "from_snapshot_token": "2025-01-01", "snapshot_token": "2025-01-01",
            "from_revision_id": 202, "revision_id": 202,
        },
        {
            "slot": "temporal_summary", "attempt_id": "temporal-1", "model": "model/t",
            "case_id": "case-t", "arm": "temporal", "repeat": 0,
            "schema_version": "temporal-pk-relative-multihop-v4",
            "navigation_id": "nav-1",
            "start_title": "Source", "start_distance": 1,
            "reasoning_hop_count": 3, "reasoning_chain": [{"index": 0}],
            "temporal_waypoints": [
                {"index": 0, "title": "Source", "revision_id": 303,
                 "as_of": "2025-01-01", "incoming_edge": "start"},
                {"index": 1, "title": "Pivot", "revision_id": 202,
                 "as_of": "2025-01-01", "incoming_edge": "hyperlink"},
            ],
            "knowledge_cutoff": {"cutoff_date": "2024-01-01"},
            "target_title_revealed": False,
            "target_title": "Pivot", "target_revision_id": 202, "final_title": "Pivot",
            "allowed_as_of": ["2024-01-01", "2025-01-01"],
            "target_snapshot_as_of": "2025-01-01",
            "pivot_hit": True, "shortest_navigation_steps": 2,
            "actual_steps_to_first_pivot": 2, "detour_steps": 0,
            "shortest_arrival": True, "revisit_count": 1, "cycle_detected": True,
            "raw_shortest_navigation_steps": 2,
            "semantic_shortest_navigation_steps": 2,
            "semantic_actual_steps_to_complete": 2,
            "semantic_route_complete": True,
            "semantic_waypoints_completed": 2, "semantic_waypoint_count": 2,
            "semantic_completion_rate": 1.0,
            "required_temporal_switches": 0,
            "actual_required_temporal_switches": 0,
            "stop_reason": "submit_answer", "visited_versions": [
                {"title": "Source", "revision_id": 303, "timestamp": "2025-01-01T00:00:00Z",
                 "as_of": "2025-01-01", "snapshot_token": "2025-01-01"},
                {"title": "Pivot", "revision_id": 101, "timestamp": "2024-01-01T00:00:00Z",
                 "as_of": "2024-01-01", "snapshot_token": "2024-01-01"},
                {"title": "Pivot", "revision_id": 202, "timestamp": "2025-01-01T00:00:00Z",
                 "as_of": "2025-01-01", "snapshot_token": "2025-01-01"},
            ],
        },
        {"slot": "final_judgment", "attempt_id": "temporal-1", "label": "correct_after"},
        {"slot": "checkpoint", "attempt_id": "temporal-1", "status": "complete"},
    ]
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_exporter_builds_page_graph_and_replayable_trajectory():
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "snapshot.db")
        results = os.path.join(tmp, "results.jsonl")
        _write_cache(cache)
        _write_results(results)
        data = build_visualization_data(cache, results)
    assert data["schema_version"] == "tkg-visualization-v4"
    assert len(data["graph"]["nodes"]) == 4
    assert len(data["graph"]["edges"]) == 3
    nodes = {node["title"]: node for node in data["graph"]["nodes"]}
    edge = next(edge for edge in data["graph"]["edges"]
                if edge["target"] == nodes["Page B"]["id"])
    assert edge["anchors"] == ["next page"]
    assert edge["observed_in_trajectory"] is True
    trajectory = data["trajectories"][0]
    assert trajectory["completed"] and trajectory["eligible"] and trajectory["page_hit"]
    assert trajectory["pivot_title"] == "Page B"
    assert trajectory["pivot_node_id"] == nodes["Page B"]["id"]
    assert trajectory["initial_reveal_node_ids"] == [nodes["Page B"]["id"]]
    assert trajectory["events"][0]["action"] == "select_snapshot"
    assert trajectory["events"][1]["action"] == "enter_start_page"
    assert set(trajectory["events"][1]["reveal_node_ids"]) == {
        nodes["Page A"]["id"], nodes["Page B"]["id"], nodes["Unfetched Page"]["id"],
    }
    assert len(trajectory["events"][1]["reveal_edge_ids"]) == 2
    assert nodes["Page C"]["id"] not in trajectory["events"][1]["reveal_node_ids"]
    assert trajectory["events"][2]["moved"] is True
    assert trajectory["events"][2]["from_title"] == "Page A"
    assert trajectory["events"][2]["to_title"] == "Page B"
    assert set(trajectory["events"][2]["reveal_node_ids"]) == {
        nodes["Page B"]["id"], nodes["Page C"]["id"],
    }
    assert trajectory["snapshot_selection"]["intent_code"] == "latest_available"
    assert trajectory["outcome_reason"] == "no_reversion_observed"


def test_exporter_can_include_unfetched_link_stubs():
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "snapshot.db")
        _write_cache(cache)
        data = build_visualization_data(
            cache, None, include_linked_stubs=True, max_stub_nodes=10
        )
    stub = next(node for node in data["graph"]["nodes"] if node["title"] == "Unfetched Page")
    assert not stub["cached"] and not stub["trajectory_only"]
    assert any(edge["target"] == stub["id"] for edge in data["graph"]["edges"])


def test_temporal_trajectory_reveals_only_switched_page_versions():
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "temporal.db")
        results = os.path.join(tmp, "temporal.jsonl")
        _write_temporal_cache(cache)
        _write_temporal_results(results)
        data = build_visualization_data(cache, results)
    trajectory = data["trajectories"][0]
    nodes = {node["revision_id"]: node for node in data["graph"]["nodes"]
             if node["revision_id"] is not None}
    assert trajectory["pivot_node_id"] == nodes[202]["id"]
    assert trajectory["initial_reveal_node_ids"] == [nodes[303]["id"]]
    assert trajectory["events"][0]["action"] == "switch_snapshot"
    assert nodes[303]["id"] in trajectory["events"][0]["reveal_node_ids"]
    assert nodes[202]["id"] in trajectory["events"][1]["reveal_node_ids"]
    assert nodes[101]["id"] in trajectory["events"][2]["reveal_node_ids"]
    temporal_edges = [edge for edge in data["graph"]["edges"]
                      if edge.get("kind") == "temporal"]
    assert temporal_edges
    assert trajectory["outcome_reason"] == "correct_after"
    assert trajectory["pk_admitted"] is True
    assert trajectory["pk_probe_n"] == 3
    assert trajectory["pk_stick_old_count"] == 3
    assert trajectory["reasoning_hop_count"] == 3
    assert trajectory["knowledge_cutoff"]["cutoff_date"] == "2024-01-01"
    assert trajectory["target_title_revealed"] is False
    assert trajectory["shortest_navigation_steps"] == 2
    assert trajectory["raw_shortest_navigation_steps"] == 2
    assert trajectory["semantic_route_complete"] is True
    assert trajectory["semantic_waypoints_completed"] == 2
    assert len(trajectory["semantic_waypoint_node_ids"]) == 2
    assert trajectory["actual_steps_to_first_pivot"] == 2
    assert trajectory["detour_steps"] == 0
    assert trajectory["cycle_detected"] is True
    assert nodes[202]["distance_to_pivot"] == 0
    assert trajectory["distance_by_node_id"][nodes[101]["id"]] == 1
    assert trajectory["distance_by_node_id"][nodes[303]["id"]] == 1


def test_outcome_classifier_uses_the_earliest_observable_failure():
    summary = {
        "snapshot_as_of": "2024-01-01", "target_snapshot_as_of": "2025-01-01",
        "page_hit": False,
    }
    outcome = _trajectory_outcome(summary, None, [], False, None)
    assert outcome["stage"] == "temporal_selection"
    assert outcome["reason"] == "selected_before_target_snapshot"

    summary["snapshot_as_of"] = "2025-01-01"
    outcome = _trajectory_outcome(summary, None, [], False, None)
    assert outcome["stage"] == "graph_navigation"
    assert outcome["reason"] == "target_page_not_opened"

    summary["page_hit"] = True
    invisible_gate = {"gate": {"pivot_visible": False, "pivot_comprehended": False}}
    outcome = _trajectory_outcome(summary, invisible_gate, [], False, None)
    assert outcome["stage"] == "exposure_visibility"
    assert outcome["reason"] == "updated_fact_not_visible"

    visible_gate = {"gate": {"pivot_visible": True, "pivot_comprehended": True}}
    outcome = _trajectory_outcome(summary, visible_gate, [], True, None)
    assert outcome["stage"] == "question_answerability"
    assert outcome["reason"] == "no_answerable_followups"


def test_site_builder_copies_self_contained_assets():
    with tempfile.TemporaryDirectory() as tmp:
        output = os.path.join(tmp, "site")
        built = build_site({"graph": {"nodes": [], "edges": []}, "trajectories": []}, output)
        assert (built / "index.html").is_file()
        assert (built / "styles.css").is_file()
        assert (built / "app.js").is_file()
        assert (built / "data.json").is_file()
        assert (built / ".tkg-visualization").is_file()


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"[OK] {test.__name__}")
    print(f"All {len(tests)} visualization tests passed.")


if __name__ == "__main__":
    main()
