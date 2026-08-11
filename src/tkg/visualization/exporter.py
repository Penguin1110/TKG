"""Export Wikipedia snapshot and browsing JSONL into a browser-friendly graph."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "tkg-visualization-v3"
CURRENT_GROUP = "__CURRENT_SNAPSHOT__"


def _excerpt(value: str, limit: int) -> str:
    collapsed = " ".join((value or "").split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit].rstrip() + "…"


def _group(as_of: str | None) -> str:
    return as_of or CURRENT_GROUP


def _stub_id(title: str, snapshot_group: str, kind: str = "linked") -> str:
    digest = hashlib.sha256(
        f"{kind}|{snapshot_group}|{title.casefold()}".encode("utf-8")
    ).hexdigest()[:20]
    return f"stub|{digest}"


def _read_snapshot(
    cache_path: str,
    *,
    include_linked_stubs: bool,
    max_stub_nodes: int,
    excerpt_chars: int,
) -> dict[str, Any]:
    path = Path(cache_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Wikipedia snapshot DB not found: {cache_path}")
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        required = {"page_cache", "page_links"}
        if not required.issubset(tables):
            raise ValueError(
                f"{cache_path} is not a Wikipedia snapshot DB; missing {sorted(required - tables)}"
            )
        rows = conn.execute(
            "SELECT cache_key, data FROM page_cache ORDER BY cache_key"
        ).fetchall()
    finally:
        conn.close()

    nodes: list[dict[str, Any]] = []
    page_records: list[tuple[str, dict[str, Any]]] = []
    title_group_to_id: dict[tuple[str, str], str] = {}
    revision_to_id: dict[tuple[str, int], str] = {}
    for cache_key, raw in rows:
        page = json.loads(raw)
        snapshot_group = _group(page.get("as_of"))
        title = str(page.get("title", ""))
        node = {
            "id": cache_key,
            "title": title,
            "revision_id": page.get("revision_id"),
            "timestamp": page.get("timestamp"),
            "as_of": page.get("as_of"),
            "snapshot_group": snapshot_group,
            "source_url": page.get("source_url"),
            "excerpt": _excerpt(str(page.get("content", "")), excerpt_chars),
            "cached": True,
            "trajectory_only": False,
            "link_count": len(page.get("links", [])),
            "external_link_count": 0,
            "in_degree": 0,
            "out_degree": 0,
        }
        nodes.append(node)
        page_records.append((cache_key, page))
        title_group_to_id[(title.casefold(), snapshot_group)] = cache_key
        revision = page.get("revision_id")
        if isinstance(revision, int):
            revision_to_id[(title.casefold(), revision)] = cache_key

    node_by_id = {node["id"]: node for node in nodes}
    edge_map: dict[tuple[str, str], dict[str, Any]] = {}
    stub_count = 0
    omitted_linked_stubs = 0
    for source_id, page in page_records:
        snapshot_group = _group(page.get("as_of"))
        for link in page.get("links", []):
            target_title = str(link.get("target", ""))
            target_id = title_group_to_id.get((target_title.casefold(), snapshot_group))
            if target_id is None and include_linked_stubs:
                candidate_id = _stub_id(target_title, snapshot_group)
                if candidate_id not in node_by_id:
                    if stub_count >= max_stub_nodes:
                        omitted_linked_stubs += 1
                        continue
                    stub = {
                        "id": candidate_id,
                        "title": target_title,
                        "revision_id": None,
                        "timestamp": None,
                        "as_of": None if snapshot_group == CURRENT_GROUP else snapshot_group,
                        "snapshot_group": snapshot_group,
                        "source_url": None,
                        "excerpt": "Linked page was not fetched into this local snapshot.",
                        "cached": False,
                        "trajectory_only": False,
                        "link_count": 0,
                        "external_link_count": 0,
                        "in_degree": 0,
                        "out_degree": 0,
                    }
                    nodes.append(stub)
                    node_by_id[candidate_id] = stub
                    title_group_to_id[(target_title.casefold(), snapshot_group)] = candidate_id
                    stub_count += 1
                target_id = candidate_id
            if target_id is None:
                node_by_id[source_id]["external_link_count"] += 1
                continue
            key = (source_id, target_id)
            edge = edge_map.setdefault(key, {
                "id": f"edge|{len(edge_map)}",
                "source": source_id,
                "target": target_id,
                "kind": "hyperlink",
                "anchors": [],
                "observed_in_trajectory": False,
            })
            anchor = str(link.get("anchor", "")).strip()
            if anchor and anchor not in edge["anchors"] and len(edge["anchors"]) < 6:
                edge["anchors"].append(anchor)

    return {
        "nodes": nodes,
        "edges": list(edge_map.values()),
        "title_group_to_id": title_group_to_id,
        "revision_to_id": revision_to_id,
        "omitted_linked_stubs": omitted_linked_stubs,
    }


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append({"_invalid_line": line_number, "_error": str(exc)})
                continue
            if isinstance(value, dict):
                value["_line_number"] = line_number
                rows.append(value)
    return rows


def _trajectory_key(row: dict[str, Any]) -> str | None:
    attempt = row.get("attempt_id")
    if attempt:
        return f"attempt:{attempt}"
    trajectory = row.get("trajectory_key")
    if trajectory:
        return f"trajectory:{trajectory}"
    return None


def _pivot_key(row: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    return (
        row.get("case_id"), row.get("arm"), row.get("contract_hash"),
        row.get("snapshot_as_of"),
    )


def _date_precedes(selected: Any, target: Any) -> bool:
    if not isinstance(selected, str) or not isinstance(target, str):
        return False
    try:
        left = datetime.fromisoformat(selected.replace("Z", "+00:00"))
        right = datetime.fromisoformat(target.replace("Z", "+00:00"))
        return left < right
    except ValueError:
        return selected < target


def _trajectory_outcome(
    summary: dict[str, Any],
    gate: dict[str, Any] | None,
    followups: list[dict[str, Any]],
    line_b_skipped: bool,
    selection: dict[str, Any] | None,
) -> dict[str, Any]:
    selected = selection.get("selected_as_of") if selection else summary.get("snapshot_as_of")
    target = summary.get("target_snapshot_as_of")
    if _date_precedes(selected, target):
        return {
            "stage": "temporal_selection",
            "reason": "selected_before_target_snapshot",
            "evidence": ["temporal_selection.selected_as_of", "browse_summary.target_snapshot_as_of"],
        }
    if not summary.get("page_hit"):
        return {
            "stage": "graph_navigation", "reason": "target_page_not_opened",
            "evidence": ["browse_summary.page_hit", "browse_summary.stop_reason"],
        }
    if gate is None:
        return {
            "stage": "exposure_evaluation", "reason": "exposure_gate_missing",
            "evidence": ["browse_summary.page_hit"],
        }
    report = gate.get("gate", {})
    if not report.get("pivot_visible"):
        return {
            "stage": "exposure_visibility", "reason": "updated_fact_not_visible",
            "evidence": ["exposure_gate.gate.visibility_judgment"],
        }
    if not report.get("pivot_comprehended"):
        return {
            "stage": "exposure_comprehension", "reason": "updated_fact_not_comprehended",
            "evidence": ["exposure_gate.gate.comprehension_judgment"],
        }
    if line_b_skipped:
        return {
            "stage": "question_answerability", "reason": "no_answerable_followups",
            "evidence": ["line_b_skipped.reason"],
        }
    if any(row.get("transition") == "reversion" for row in followups):
        return {
            "stage": "knowledge_retention", "reason": "reversion_observed",
            "evidence": ["followup.transition"],
        }
    if any(row.get("label") == "stick_old" for row in followups):
        return {
            "stage": "knowledge_retention", "reason": "old_answer_after_comprehension",
            "evidence": ["followup.label"],
        }
    if followups:
        return {
            "stage": "knowledge_retention", "reason": "no_reversion_observed",
            "evidence": ["followup.label", "followup.transition"],
        }
    return {
        "stage": "knowledge_retention", "reason": "no_followup_rows",
        "evidence": ["exposure_gate.gate.eligible"],
    }


class _GraphResolver:
    def __init__(self, graph: dict[str, Any]):
        self.graph = graph
        self.nodes = graph["nodes"]
        self.edges = graph["edges"]
        self.node_by_id = {node["id"]: node for node in self.nodes}
        self.title_group_to_id = graph["title_group_to_id"]
        self.revision_to_id = graph["revision_to_id"]
        self.edge_by_pair = {
            (edge["source"], edge["target"]): edge for edge in self.edges
        }

    def node_id(
        self,
        title: str,
        *,
        revision_id: int | None = None,
        snapshot_group: str | None = None,
    ) -> str:
        folded = title.casefold()
        if revision_id is not None:
            exact = self.revision_to_id.get((folded, revision_id))
            if exact:
                return exact
        if snapshot_group:
            grouped = self.title_group_to_id.get((folded, snapshot_group))
            if grouped:
                return grouped
        matches = [
            node["id"] for node in self.nodes
            if node["title"].casefold() == folded and node.get("cached")
        ]
        if len(matches) == 1:
            return matches[0]
        group = snapshot_group or "__TRAJECTORY_ONLY__"
        node_id = _stub_id(title, group, kind="trajectory")
        if node_id not in self.node_by_id:
            node = {
                "id": node_id,
                "title": title,
                "revision_id": revision_id,
                "timestamp": None,
                "as_of": None,
                "snapshot_group": group,
                "source_url": None,
                "excerpt": "Page appears in a trajectory but was not found in this snapshot DB.",
                "cached": False,
                "trajectory_only": True,
                "link_count": 0,
                "external_link_count": 0,
                "in_degree": 0,
                "out_degree": 0,
            }
            self.nodes.append(node)
            self.node_by_id[node_id] = node
            self.title_group_to_id[(folded, group)] = node_id
        return node_id

    def observed_edge(self, source: str, target: str) -> None:
        if source == target:
            return
        pair = (source, target)
        edge = self.edge_by_pair.get(pair)
        if edge:
            edge["observed_in_trajectory"] = True
            return
        edge = {
            "id": f"edge|{len(self.edges)}",
            "source": source,
            "target": target,
            "kind": "hyperlink",
            "anchors": [],
            "observed_in_trajectory": True,
        }
        self.edges.append(edge)
        self.edge_by_pair[pair] = edge

    def temporal_edge(self, source: str, target: str) -> str | None:
        if source == target:
            return None
        pair = (source, target)
        edge = self.edge_by_pair.get(pair)
        if edge and edge.get("kind") == "temporal":
            edge["observed_in_trajectory"] = True
            return str(edge["id"])
        edge = {
            "id": f"edge|{len(self.edges)}",
            "source": source,
            "target": target,
            "kind": "temporal",
            "anchors": [],
            "observed_in_trajectory": True,
        }
        self.edges.append(edge)
        self.edge_by_pair[pair] = edge
        return str(edge["id"])


def _temporal_trajectory(
    group_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    resolver: _GraphResolver,
    arena: dict[str, Any] | None = None,
    pk_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    steps = sorted(
        (row for row in group_rows if row.get("slot") == "temporal_step"),
        key=lambda row: (row.get("step", 0), row.get("_line_number", 0)),
    )
    versions = summary.get("visited_versions", [])
    version_by_revision = {
        item.get("revision_id"): item for item in versions if item.get("revision_id") is not None
    }
    pivot_title = str(summary.get("target_title") or summary.get("start_title") or "")
    allowed = summary.get("allowed_as_of", [])
    target_as_of = summary.get("target_snapshot_as_of")
    latest_token = _group(target_as_of if target_as_of is not None else (
        allowed[-1] if allowed else None
    ))
    target_revision_id = summary.get("target_revision_id")
    pivot_version = next((
        item for item in reversed(versions)
        if str(item.get("title", "")).casefold() == pivot_title.casefold()
        and (
            item.get("revision_id") == target_revision_id
            if target_revision_id is not None
            else _group(item.get("as_of")) == latest_token
        )
    ), None)
    if pivot_version is None:
        pivot_version = next((
            item for item in reversed(versions)
            if str(item.get("title", "")).casefold() == pivot_title.casefold()
        ), None)
    pivot_node_id = resolver.node_id(
        pivot_title,
        revision_id=(
            target_revision_id if target_revision_id is not None
            else pivot_version.get("revision_id") if pivot_version else None
        ),
        snapshot_group=latest_token,
    )
    resolver.node_by_id[pivot_node_id]["distance_to_pivot"] = 0
    distance_by_node_id: dict[str, int] = {pivot_node_id: 0}
    if arena:
        for state in arena.get("states", []):
            node_id = resolver.node_id(
                str(state.get("title", "")),
                revision_id=state.get("revision_id"),
                snapshot_group=_group(state.get("as_of")),
            )
            distance = state.get("distance_to_pivot")
            if distance is not None:
                distance_by_node_id[node_id] = int(distance)

    events: list[dict[str, Any]] = []
    path_node_ids: list[str] = []
    current_node_id: str | None = None
    current_title = str(summary.get("start_title") or pivot_title)
    for row in steps:
        action = str(row.get("action", ""))
        from_title = str(row.get("from_title") or current_title)
        to_title = str(row.get("to_title") or from_title)
        from_revision = row.get("from_revision_id")
        to_revision = row.get("revision_id")
        from_version = version_by_revision.get(from_revision, {})
        to_version = version_by_revision.get(to_revision, {})
        to_id = (
            resolver.node_id(
                to_title, revision_id=to_revision,
                snapshot_group=_group(to_version.get("as_of")),
            )
            if to_revision is not None else current_node_id or pivot_node_id
        )
        from_id = (
            resolver.node_id(
                from_title, revision_id=from_revision,
                snapshot_group=_group(from_version.get("as_of")),
            )
            if from_revision is not None else current_node_id or to_id
        )
        distance_to_pivot = row.get("distance_to_pivot")
        if distance_to_pivot is not None:
            resolver.node_by_id[to_id]["distance_to_pivot"] = distance_to_pivot
            distance_by_node_id[to_id] = int(distance_to_pivot)
        direct_edges: list[str] = []
        if (action == "switch_snapshot" and from_revision is not None
                and not str(row.get("result", "")).startswith("Error:")):
            edge_id = resolver.temporal_edge(from_id, to_id)
            if edge_id:
                direct_edges.append(edge_id)
        elif action == "follow_link" and from_id != to_id:
            resolver.observed_edge(from_id, to_id)
        moved = from_id != to_id
        expansion = to_id if action in {"switch_snapshot", "follow_link"} and not str(
            row.get("result", "")
        ).startswith("Error:") else None
        events.append({
            "index": len(events),
            "step": row.get("step"),
            "action": action,
            "from_node_id": from_id,
            "to_node_id": to_id,
            "from_title": from_title,
            "to_title": to_title,
            "moved": moved,
            "expansion_node_id": expansion,
            "direct_reveal_edge_ids": direct_edges,
            "args": row.get("args", {}),
            "result": _excerpt(str(row.get("result", "")), 900),
            "free_text": _excerpt(str(row.get("free_text") or ""), 400),
            "snapshot_token": row.get("snapshot_token"),
            "revision_id": to_revision,
            "navigation_step": row.get("navigation_step"),
            "distance_to_pivot": distance_to_pivot,
            "revisited": bool(row.get("revisited")),
        })
        if (action in {"switch_snapshot", "follow_link"}
                and not str(row.get("result", "")).startswith("Error:")
                and (not path_node_ids or path_node_ids[-1] != to_id)):
            path_node_ids.append(to_id)
        current_node_id = to_id
        current_title = to_title

    checkpoints = [row for row in group_rows if row.get("slot") == "checkpoint"]
    checkpoint = max(checkpoints, key=lambda row: row.get("_line_number", 0)) if checkpoints else None
    judgments = [row for row in group_rows if row.get("slot") == "final_judgment"]
    judgment = max(judgments, key=lambda row: row.get("_line_number", 0)) if judgments else None
    label = judgment.get("label") if judgment else "missing_final_judgment"
    pk_required = str(summary.get("schema_version", "")).startswith("temporal-pk-")
    pk_ok = bool(pk_gate.get("passed")) if pk_gate else not pk_required
    return {
        "id": str(summary.get("attempt_id") or summary.get("trajectory_key")),
        "trajectory_key": summary.get("trajectory_key"),
        "attempt_id": summary.get("attempt_id"),
        "contract_hash": summary.get("contract_hash"),
        "model": summary.get("model"),
        "case_id": summary.get("case_id"),
        "arm": "temporal",
        "start_distance": summary.get("start_distance"),
        "start_title": summary.get("start_title"),
        "reasoning_hop_count": summary.get("reasoning_hop_count"),
        "reasoning_chain": summary.get("reasoning_chain", []),
        "knowledge_cutoff": summary.get("knowledge_cutoff"),
        "target_title_revealed": summary.get("target_title_revealed", True),
        "repeat": summary.get("repeat"),
        "final_title": summary.get("final_title"),
        "pivot_title": pivot_title,
        "pivot_node_id": pivot_node_id,
        "pivot_source": "temporal_summary.target_title",
        "snapshot_as_of": None,
        "page_hit": bool(summary.get("pivot_hit")),
        "stop_reason": summary.get("stop_reason"),
        "completed": bool(checkpoint and checkpoint.get("status") == "complete"),
        "eligible": bool(judgment and pk_ok),
        "failure_reasons": (
            [] if judgment and pk_ok else
            (["missing_pk_admission"] if not pk_ok
             else ["missing_final_judgment"])
        ),
        "pk_admitted": pk_gate.get("passed") if pk_gate else None,
        "pk_gate_reason": pk_gate.get("reason") if pk_gate else "missing_pk_gate",
        "pk_probe_n": pk_gate.get("n") if pk_gate else 0,
        "pk_stick_new_count": pk_gate.get("stick_new_count") if pk_gate else None,
        "pk_stick_old_count": pk_gate.get("stick_old_count") if pk_gate else None,
        "snapshot_selection": None,
        "outcome_stage": "temporal_answer",
        "outcome_reason": label,
        "outcome_evidence": ["final_judgment.label"],
        "path_node_ids": list(dict.fromkeys(path_node_ids)),
        "shortest_navigation_steps": summary.get("shortest_navigation_steps"),
        "actual_steps_to_first_pivot": summary.get("actual_steps_to_first_pivot"),
        "detour_steps": summary.get("detour_steps"),
        "shortest_arrival": bool(summary.get("shortest_arrival")),
        "revisit_count": summary.get("revisit_count", 0),
        "cycle_detected": bool(summary.get("cycle_detected")),
        "distance_by_node_id": distance_by_node_id,
        "events": events,
    }


def _load_trajectories(results_path: str | None, graph: dict[str, Any]) -> tuple[list, int]:
    if not results_path or not Path(results_path).is_file():
        return [], 0
    rows = _read_jsonl(results_path)
    invalid_lines = sum(1 for row in rows if "_invalid_line" in row)
    arenas = {
        str(row.get("navigation_id")): row for row in rows
        if row.get("slot") == "navigation_arena" and row.get("navigation_id")
    }
    pk_gates = {
        (str(row.get("case_id")), str(row.get("model")), str(row.get("contract_hash"))): row
        for row in rows if row.get("slot") == "pk_gate"
    }
    selections = {
        str(row.get("selection", {}).get("selection_id")): row.get("selection", {})
        for row in rows
        if row.get("slot") == "temporal_selection"
        and row.get("selection", {}).get("selection_id")
    }
    # New logs carry target_title directly.  For older logs, a successful peer
    # trajectory in the same case/arm/snapshot identifies the pivot.
    pivot_titles: dict[tuple[Any, Any, Any, Any], tuple[str, str]] = {}
    for row in rows:
        if row.get("slot") != "browse_summary":
            continue
        target_title = str(row.get("target_title") or "").strip()
        if target_title:
            pivot_titles[_pivot_key(row)] = (target_title, "target_title")
        elif row.get("page_hit") and row.get("final_title"):
            pivot_titles.setdefault(
                _pivot_key(row), (str(row["final_title"]), "successful_peer")
            )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _trajectory_key(row)
        if key:
            groups[key].append(row)

    resolver = _GraphResolver(graph)
    trajectories = []
    for key, group_rows in groups.items():
        summaries = [
            row for row in group_rows
            if row.get("slot") in {"browse_summary", "temporal_summary"}
        ]
        if not summaries:
            continue
        summary = max(summaries, key=lambda row: row.get("_line_number", 0))
        if summary.get("slot") == "temporal_summary":
            trajectories.append(_temporal_trajectory(
                group_rows, summary, resolver,
                arenas.get(str(summary.get("navigation_id"))),
                pk_gates.get((
                    str(summary.get("case_id")), str(summary.get("model")),
                    str(summary.get("contract_hash")),
                )),
            ))
            continue
        steps = sorted(
            (row for row in group_rows if row.get("slot") == "browse_step"),
            key=lambda row: (row.get("step", 0), row.get("_line_number", 0)),
        )
        gates = [row for row in group_rows if row.get("slot") == "exposure_gate"]
        checkpoints = [row for row in group_rows if row.get("slot") == "checkpoint"]
        followups = [row for row in group_rows if row.get("slot") == "followup"]
        line_b_skipped = any(row.get("slot") == "line_b_skipped" for row in group_rows)
        gate = max(gates, key=lambda row: row.get("_line_number", 0)) if gates else None
        checkpoint = (
            max(checkpoints, key=lambda row: row.get("_line_number", 0))
            if checkpoints else None
        )
        revision_by_title = {
            str(item.get("title", "")).casefold(): item.get("revision_id")
            for item in summary.get("evidence_revisions", [])
        }
        snapshot_group = (
            _group(summary.get("snapshot_as_of"))
            if "snapshot_as_of" in summary else None
        )
        for title_folded, revision in revision_by_title.items():
            node_id = graph["revision_to_id"].get((title_folded, revision))
            if node_id:
                node = resolver.node_by_id[node_id]
                snapshot_group = node.get("snapshot_group")
                break

        visited_titles = [str(value) for value in summary.get("visited_titles", [])]
        path_node_ids = [
            resolver.node_id(
                title,
                revision_id=revision_by_title.get(title.casefold()),
                snapshot_group=snapshot_group,
            )
            for title in visited_titles
        ]
        pivot_record = pivot_titles.get(_pivot_key(summary))
        if pivot_record:
            pivot_title, pivot_source = pivot_record
        elif summary.get("page_hit") and summary.get("final_title"):
            pivot_title, pivot_source = str(summary["final_title"]), "hit_final_title"
        elif visited_titles:
            # Backward-compatible last resort for an old miss-only result file.
            pivot_title, pivot_source = visited_titles[0], "fallback_start"
        else:
            pivot_title, pivot_source = "", "missing"
        pivot_node_id = (
            resolver.node_id(
                pivot_title,
                revision_id=revision_by_title.get(pivot_title.casefold()),
                snapshot_group=snapshot_group,
            )
            if pivot_title else None
        )
        path_cursor = 0
        events: list[dict[str, Any]] = []
        for row in steps:
            from_title = str(row.get("from_title") or (
                visited_titles[path_cursor] if visited_titles else summary.get("start_title", "")
            ))
            from_id = resolver.node_id(
                from_title,
                revision_id=revision_by_title.get(from_title.casefold()),
                snapshot_group=snapshot_group,
            )
            action = str(row.get("action", ""))
            result = str(row.get("result", ""))
            to_id = from_id
            to_title = from_title
            if action == "follow_link" and not result.lstrip().startswith("Error:"):
                if path_cursor + 1 < len(visited_titles):
                    path_cursor += 1
                    to_title = visited_titles[path_cursor]
                else:
                    to_title = str(row.get("args", {}).get("target", from_title))
                to_id = resolver.node_id(
                    to_title,
                    revision_id=revision_by_title.get(to_title.casefold()),
                    snapshot_group=snapshot_group,
                )
                resolver.observed_edge(from_id, to_id)
            events.append({
                "index": len(events),
                "step": row.get("step"),
                "action": action,
                "from_node_id": from_id,
                "to_node_id": to_id,
                "from_title": from_title,
                "to_title": to_title,
                "moved": from_id != to_id,
                "expansion_node_id": to_id if from_id != to_id else None,
                "args": row.get("args", {}),
                "result": _excerpt(result, 900),
                "free_text": _excerpt(str(row.get("free_text") or ""), 400),
            })

        # A summary is authoritative.  If old or compact logs omit browse_step rows,
        # synthesize path events so the visited path can still be replayed.
        if not events and len(path_node_ids) > 1:
            for index in range(len(path_node_ids) - 1):
                resolver.observed_edge(path_node_ids[index], path_node_ids[index + 1])
                events.append({
                    "index": index,
                    "step": index + 1,
                    "action": "follow_link",
                    "from_node_id": path_node_ids[index],
                    "to_node_id": path_node_ids[index + 1],
                    "from_title": visited_titles[index],
                    "to_title": visited_titles[index + 1],
                    "moved": True,
                    "expansion_node_id": path_node_ids[index + 1],
                    "args": {"target": visited_titles[index + 1]},
                    "result": "Synthesized from browse_summary.visited_titles",
                    "free_text": "",
                })

        start_id = path_node_ids[0] if path_node_ids else None
        start_title = visited_titles[0] if visited_titles else summary.get("start_title")
        if start_id:
            events.insert(0, {
                "index": 0,
                "step": 0,
                "action": "enter_start_page",
                "from_node_id": start_id,
                "to_node_id": start_id,
                "from_title": start_title,
                "to_title": start_title,
                "moved": False,
                "expansion_node_id": start_id,
                "args": {},
                "result": "Entered the start page; reveal its outgoing hyperlinks.",
                "free_text": "",
            })

        selection = selections.get(str(summary.get("snapshot_selection_id")))
        if selection:
            anchor_id = pivot_node_id or start_id
            anchor_title = pivot_title or start_title
            events.insert(0, {
                "index": 0,
                "step": 0,
                "action": "select_snapshot",
                "from_node_id": anchor_id,
                "to_node_id": anchor_id,
                "from_title": anchor_title,
                "to_title": anchor_title,
                "moved": False,
                "expansion_node_id": None,
                "args": {
                    "as_of": selection.get("selection_token"),
                    "intent_code": selection.get("intent_code"),
                    "brief_reason": selection.get("brief_reason"),
                },
                "result": (
                    f"Bound to {selection.get('selection_token')} · "
                    f"{selection.get('intent_code')}: {selection.get('brief_reason')}"
                ),
                "free_text": "",
            })
        for index, event in enumerate(events):
            event["index"] = index

        outcome = _trajectory_outcome(
            summary, gate, followups, line_b_skipped, selection
        )

        trajectory_id = str(summary.get("attempt_id") or summary.get("trajectory_key") or key)
        trajectories.append({
            "id": trajectory_id,
            "trajectory_key": summary.get("trajectory_key"),
            "attempt_id": summary.get("attempt_id"),
            "contract_hash": summary.get("contract_hash"),
            "model": summary.get("model"),
            "case_id": summary.get("case_id"),
            "arm": summary.get("arm"),
            "start_distance": summary.get("start_distance"),
            "start_title": summary.get("start_title"),
            "repeat": summary.get("repeat"),
            "final_title": summary.get("final_title"),
            "pivot_title": pivot_title or None,
            "pivot_node_id": pivot_node_id,
            "pivot_source": pivot_source,
            "snapshot_as_of": summary.get("snapshot_as_of"),
            "page_hit": bool(summary.get("page_hit")),
            "stop_reason": summary.get("stop_reason"),
            "completed": bool(checkpoint and checkpoint.get("status") == "complete"),
            "eligible": bool(gate and gate.get("gate", {}).get("eligible")),
            "failure_reasons": (
                gate.get("gate", {}).get("failure_reasons", []) if gate else []
            ),
            "snapshot_selection": ({
                key: value for key, value in selection.items() if key != "messages"
            } if selection else None),
            "outcome_stage": outcome["stage"],
            "outcome_reason": outcome["reason"],
            "outcome_evidence": outcome["evidence"],
            "path_node_ids": path_node_ids,
            "events": events,
        })

    # Each actually visited page expands exactly one outgoing hyperlink layer.
    # The frontend replays these explicit reveal sets instead of drawing the
    # complete snapshot graph up front.
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in resolver.edges:
        if edge.get("kind", "hyperlink") == "hyperlink":
            outgoing[str(edge["source"])].append(edge)
    for trajectory in trajectories:
        pivot_node_id = trajectory.get("pivot_node_id")
        trajectory["initial_reveal_node_ids"] = (
            [pivot_node_id] if pivot_node_id else []
        )
        for event in trajectory["events"]:
            expansion = event.get("expansion_node_id")
            expanded_edges = outgoing.get(str(expansion), []) if expansion else []
            direct_edge_ids = list(event.get("direct_reveal_edge_ids", []))
            event["reveal_node_ids"] = list(dict.fromkeys(
                ([expansion] if expansion else [])
                + [edge["target"] for edge in expanded_edges]
            ))
            event["reveal_edge_ids"] = list(dict.fromkeys(
                direct_edge_ids + [edge["id"] for edge in expanded_edges]
            ))

    trajectories.sort(key=lambda item: (
        str(item.get("model")), str(item.get("case_id")), str(item.get("arm")),
        int(item.get("start_distance") or 0), str(item.get("start_title")),
        int(item.get("repeat") or 0), str(item.get("id")),
    ))
    return trajectories, invalid_lines


def _finalize_graph(graph: dict[str, Any]) -> None:
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    for node in graph["nodes"]:
        node["in_degree"] = 0
        node["out_degree"] = 0
    for edge in graph["edges"]:
        source = node_by_id.get(edge["source"])
        target = node_by_id.get(edge["target"])
        if source:
            source["out_degree"] += 1
        if target:
            target["in_degree"] += 1
    graph["nodes"].sort(key=lambda node: (
        str(node.get("snapshot_group")), str(node.get("title", "")).casefold(), node["id"]
    ))
    graph["edges"].sort(key=lambda edge: (edge["source"], edge["target"]))


def build_visualization_data(
    cache_path: str,
    results_path: str | None = "temporal_results.jsonl",
    *,
    include_linked_stubs: bool = True,
    max_stub_nodes: int = 2000,
    excerpt_chars: int = 700,
) -> dict[str, Any]:
    graph = _read_snapshot(
        cache_path,
        include_linked_stubs=include_linked_stubs,
        max_stub_nodes=max_stub_nodes,
        excerpt_chars=excerpt_chars,
    )
    trajectories, invalid_lines = _load_trajectories(results_path, graph)
    _finalize_graph(graph)
    public_graph = {"nodes": graph["nodes"], "edges": graph["edges"]}
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "cache_path": str(Path(cache_path).resolve()),
            "results_path": str(Path(results_path).resolve()) if results_path else None,
            "results_found": bool(results_path and Path(results_path).is_file()),
            "include_linked_stubs": include_linked_stubs,
            "omitted_linked_stubs": graph["omitted_linked_stubs"],
            "invalid_jsonl_lines": invalid_lines,
        },
        "graph": public_graph,
        "trajectories": trajectories,
        "filters": {
            "models": sorted({str(item["model"]) for item in trajectories if item.get("model")}),
            "cases": sorted({str(item["case_id"]) for item in trajectories if item.get("case_id")}),
            "arms": sorted({str(item["arm"]) for item in trajectories if item.get("arm")}),
            "snapshot_groups": sorted({str(node["snapshot_group"]) for node in graph["nodes"]}),
        },
    }
