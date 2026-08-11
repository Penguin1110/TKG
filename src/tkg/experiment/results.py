"""Append-only JSONL results and checkpoints for the Wikipedia experiment."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


PROTECTED_ARTIFACTS = {
    "results.jsonl", "results_free_exploration.jsonl",
    "summary.csv", "summary_free_exploration.csv",
    "summary_conflict_vs_control.csv", "summary_conflict_vs_control_free_exploration.csv",
    "hit_rate_by_distance.csv", "report.txt", "report.html",
}


def assert_new_output_path(path: str):
    if Path(path).name in PROTECTED_ARTIFACTS:
        raise ValueError(
            f"拒絕把新版 Wikipedia 實驗寫入既有 artifact {path!r}；請使用新的輸出檔名"
        )


class JsonlResultStore:
    def __init__(self, path: str, metadata: dict | None = None):
        assert_new_output_path(path)
        self.path = path
        self.metadata = dict(metadata or {})
        self._fh = open(path, "a", encoding="utf-8")

    def close(self):
        self._fh.close()

    def write(self, *, slot: str, case_id: str, model: str, arm: str, **fields):
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": "wikipedia-v1",
            "slot": slot, "case_id": case_id, "model": model, "arm": arm,
            **self.metadata, **fields,
        }
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    @staticmethod
    def completed(path: str, contract_hash: str | None = None) -> set[tuple]:
        completed: set[tuple] = set()
        file = Path(path)
        if not file.exists():
            return completed
        with file.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                contract_matches = contract_hash is None or row.get("contract_hash") == contract_hash
                if (row.get("slot") == "checkpoint" and row.get("status") == "complete"
                        and contract_matches):
                    completed.add((
                        row.get("case_id"), row.get("model"), row.get("arm"),
                        row.get("start_distance"), row.get("start_title"), row.get("repeat"),
                    ))
        return completed

    @staticmethod
    def latest_pk_gate(path: str, case_id: str, model: str,
                       contract_hash: str) -> bool | None:
        file = Path(path)
        if not file.exists():
            return None
        verdict = None
        with file.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (row.get("slot") == "pk_gate" and row.get("case_id") == case_id
                        and row.get("model") == model
                        and row.get("contract_hash") == contract_hash):
                    verdict = bool(row.get("passed"))
        return verdict

    @staticmethod
    def latest_snapshot_selection(
        path: str, case_id: str, model: str, contract_hash: str
    ) -> dict | None:
        file = Path(path)
        if not file.exists():
            return None
        selection = None
        with file.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = row.get("selection", {})
                if (row.get("slot") == "temporal_selection"
                        and row.get("case_id") == case_id
                        and row.get("model") == model
                        and row.get("contract_hash") == contract_hash
                        and value.get("status") == "selected"):
                    selection = value
        return selection
