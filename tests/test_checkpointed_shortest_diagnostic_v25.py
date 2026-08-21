from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tkg.experiment.checkpointed_shortest_diagnostic_v25 import (
    CheckpointedTemporalReverseBFSV25,
)


@dataclass
class Page:
    title: str
    revision_id: int
    timestamp: str


class Backend:
    def fetch_page(self, title: str, as_of: str | None = None):
        if title == "Target":
            return Page("Target", 2, as_of or "")
        return Page("Start", 1, as_of or "")

    def find_backlinks(self, title: str, as_of: str | None = None, max_results: int = 50):
        del as_of, max_results
        return ["Start"] if title == "Target" else []


def test_checkpointed_bfs_resumes_fifo_without_rescanning_completed_work(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "wiki.db"
    cache.write_bytes(b"cache")
    checkpoint = tmp_path / "checkpoint.json"
    kwargs = dict(
        checkpoint_path=checkpoint, cache_path=cache, case_id="case",
        start_title="Start", target_title="Target", target_as_of="2025-01-01",
        allowed_dates=["2025-01-01"], max_depth=1, branch_cap=5, max_nodes=10,
    )
    first = CheckpointedTemporalReverseBFSV25(Backend(), **kwargs).run(max_expansions=2)  # type: ignore[arg-type]
    assert first["status"] == "incomplete"
    assert first["pending_expansions"] == []
    assert first["start_state_found"] is True
    completed = first["completed_expansions"]

    second = CheckpointedTemporalReverseBFSV25(Backend(), **kwargs).run(max_expansions=10)  # type: ignore[arg-type]
    assert second["status"] == "complete"
    assert second["shortest_path_status"] == "bounded_lower_bound"
    assert second["bounded_shortest_distance"] == 1
    assert second["completed_expansions"] >= completed
    assert second["canonical_ordering"].startswith("FIFO")
    assert second["cache"]["sha256"]
