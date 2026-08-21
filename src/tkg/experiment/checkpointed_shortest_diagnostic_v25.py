"""Resumable background reverse-BFS diagnostic for temporal Wikipedia cases.

This program is explicitly non-admitting: its checkpoint/result must never be
read by generation, PK admission, compaction, scoring, or inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tkg.wikipedia.backend import WikipediaError, WikipediaPageBackend
from tkg.wikipedia.snapshot import page_version_key


SCHEMA_VERSION = "checkpointed-temporal-reverse-bfs-v2.5"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_fingerprint(path: Path) -> str:
    parts = []
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        parts.append({
            "name": candidate.name, "size": candidate.stat().st_size if candidate.exists() else 0,
            "sha256": _file_sha256(candidate),
        })
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


class CheckpointedTemporalReverseBFSV25:
    def __init__(
        self, backend: WikipediaPageBackend, *, checkpoint_path: Path,
        cache_path: Path, case_id: str, start_title: str, target_title: str,
        target_as_of: str, allowed_dates: list[str], max_depth: int,
        branch_cap: int = 10, max_nodes: int = 500,
    ) -> None:
        self.backend = backend
        self.path = checkpoint_path
        self.cache_path = cache_path
        self.start_title = start_title
        self.target_title = target_title
        self.target_as_of = target_as_of
        self.allowed_dates = list(dict.fromkeys(allowed_dates))
        self.max_depth = max_depth
        self.branch_cap = branch_cap
        self.max_nodes = max_nodes
        self.parameters = {
            "case_id": case_id, "start_title": start_title,
            "target_title": target_title, "target_as_of": target_as_of,
            "allowed_dates": self.allowed_dates,
            "max_depth": max_depth, "branch_cap": branch_cap,
            "max_nodes": max_nodes,
        }

    def _new(self) -> dict[str, Any]:
        page = self.backend.fetch_page(
            self.target_title, as_of=self.target_as_of,
        )
        key = page_version_key(page.title, page.revision_id)
        return {
            "schema_version": SCHEMA_VERSION, "created_at": _now(),
            "updated_at": _now(), "parameters": self.parameters,
            "status": "incomplete", "shortest_path_status": "incomplete",
            "frontier": [key],
            "visited": {key: {
                "page": page.title, "revision_id": page.revision_id,
                "revision_date": page.timestamp, "as_of_tokens": [self.target_as_of],
                "depth": 0, "parent": None, "action": None,
                "revision_query_cursor": 0, "backlink_cursor": 0,
            }},
            "completed_expansions": 0, "pending_expansions": [],
            "canonical_ordering": "FIFO; dates input-order; backlinks casefold-title",
            "cache": {"version": "wikipedia-backend-sqlite-v1", "path": str(self.cache_path)},
            "errors": [],
        }

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._new()
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("checkpoint schema mismatch")
        if state.get("parameters") != self.parameters:
            raise ValueError("checkpoint parameters changed")
        return state

    def checkpoint(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _now()
        state["cache"]["sha256"] = _cache_fingerprint(self.cache_path)
        _atomic_json(self.path, state)

    def _add(
        self, state: dict[str, Any], page: Any, *, token: str, depth: int,
        parent: str, action: str,
    ) -> None:
        key = page_version_key(page.title, page.revision_id)
        existing = state["visited"].get(key)
        if existing:
            if token not in existing["as_of_tokens"]:
                existing["as_of_tokens"].append(token)
            return
        if len(state["visited"]) >= self.max_nodes:
            state["node_cap_reached"] = True
            return
        state["visited"][key] = {
            "page": page.title, "revision_id": page.revision_id,
            "revision_date": page.timestamp, "as_of_tokens": [token],
            "depth": depth, "parent": parent, "action": action,
            "revision_query_cursor": 0, "backlink_cursor": 0,
        }
        state["frontier"].append(key)

    def run(self, *, max_expansions: int) -> dict[str, Any]:
        state = self.load()
        units = 0
        while state["frontier"] and units < max_expansions:
            key = state["frontier"][0]
            node = state["visited"][key]
            depth = int(node["depth"])
            if depth >= self.max_depth:
                state["frontier"].pop(0)
                state["completed_expansions"] += 1
                continue
            dates = self.allowed_dates
            date_cursor = int(node["revision_query_cursor"])
            tokens = list(node["as_of_tokens"])
            backlink_cursor = int(node["backlink_cursor"])
            if date_cursor < len(dates):
                token = dates[date_cursor]
                pending = {"node": key, "kind": "revision", "cursor": date_cursor}
                state["pending_expansions"] = [pending]
                self.checkpoint(state)
                try:
                    predecessor = self.backend.fetch_page(node["page"], as_of=token)
                    landing = self.backend.fetch_page(predecessor.title, as_of=tokens[0])
                    if page_version_key(landing.title, landing.revision_id) == key:
                        self._add(
                            state, predecessor, token=token, depth=depth + 1,
                            parent=key, action="SWITCH_SNAPSHOT",
                        )
                except WikipediaError as exc:
                    state["errors"].append({"node": key, "kind": "revision", "error": str(exc)})
                node["revision_query_cursor"] = date_cursor + 1
            elif backlink_cursor < len(tokens):
                token = tokens[backlink_cursor]
                pending = {"node": key, "kind": "backlink", "cursor": backlink_cursor}
                state["pending_expansions"] = [pending]
                self.checkpoint(state)
                try:
                    sources = self.backend.find_backlinks(
                        node["page"], as_of=token,
                        max_results=self.branch_cap,
                    )
                    for title in sorted(sources, key=str.casefold):
                        try:
                            page = self.backend.fetch_page(title, as_of=token)
                        except WikipediaError as exc:
                            state["errors"].append({"node": key, "kind": "backlink_page", "error": str(exc)})
                            continue
                        self._add(
                            state, page, token=token, depth=depth + 1,
                            parent=key, action="FOLLOW_LINK",
                        )
                except WikipediaError as exc:
                    state["errors"].append({"node": key, "kind": "backlink", "error": str(exc)})
                node["backlink_cursor"] = backlink_cursor + 1
            else:
                state["frontier"].pop(0)
                state["completed_expansions"] += 1
            state["pending_expansions"] = []
            units += 1
            self.checkpoint(state)

        if not state["frontier"]:
            state["status"] = "complete"
            state["shortest_path_status"] = "bounded_lower_bound"
        else:
            state["status"] = "incomplete"
            state["shortest_path_status"] = "incomplete"
        state["frontier_incomplete"] = bool(state["frontier"])
        state["global_shortest_complete"] = False
        state["bfs_explored_nodes"] = len(state["visited"])
        matching_depths = [
            int(node["depth"]) for node in state["visited"].values()
            if str(node["page"]).casefold() == self.start_title.casefold()
        ]
        state["bounded_shortest_distance"] = min(matching_depths) if matching_depths else None
        state["start_state_found"] = bool(matching_depths)
        self.checkpoint(state)
        return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-expansions", type=int, default=10)
    parser.add_argument("--branch-cap", type=int, default=10)
    parser.add_argument("--max-nodes", type=int, default=500)
    parser.add_argument("--request-interval", type=float, default=1.0)
    args = parser.parse_args()
    rows = json.loads(Path(args.cases).read_text(encoding="utf-8")).get("cases", [])
    case = next((row for row in rows if row.get("id") == args.case_id), None)
    if case is None:
        parser.error("case ID not found")
    cache = Path(args.cache_path)
    backend = WikipediaPageBackend(cache_path=str(cache), min_request_interval=args.request_interval)
    try:
        runner = CheckpointedTemporalReverseBFSV25(
            backend, checkpoint_path=Path(args.checkpoint), cache_path=cache,
            case_id=args.case_id, start_title=case["start_title"],
            target_title=case["wikipedia_title"], target_as_of=case["wikipedia_as_of"],
            allowed_dates=case["required_snapshot_dates"],
            max_depth=case["expected_navigation_distance"],
            branch_cap=args.branch_cap, max_nodes=args.max_nodes,
        )
        state = runner.run(max_expansions=args.max_expansions)
    finally:
        backend.close()
    print(json.dumps({
        "status": state["status"], "shortest_path_status": state["shortest_path_status"],
        "bfs_explored_nodes": state["bfs_explored_nodes"],
        "completed_expansions": state["completed_expansions"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
