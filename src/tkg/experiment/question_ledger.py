"""Persistent, content-addressed ledger for renewable question generation.

The ledger is deliberately separate from generated JSON artifacts.  It stops a
scheduled generator from repeatedly proposing the same semantic path while
preserving every decision needed to audit why a path was admitted or rejected.
Infrastructure failures are retryable and therefore do not blacklist a path.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LEDGER_SCHEMA_VERSION = "renewable-question-ledger-v1"
TERMINAL_STATUSES = {
    "machine_pass_human_review_required",
    # Backward-compatible terminal label used by pre-v0.15 development ledgers.
    "machine_accepted_human_review_required",
    "deterministic_reject",
    "judge_reject",
    "duplicate",
}
RETRYABLE_STATUSES = {"pending", "infrastructure_error"}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def question_fingerprint(
    *,
    model_id: str,
    cutoff_date: str,
    target_as_of: str,
    entity_qids: list[str],
    property_steps: list[dict[str, str]],
    snapshot_dates: list[str],
    answer_qid: str,
) -> str:
    """Hash only semantic identity fields, never display labels or evidence text."""
    contract = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "model_id": model_id,
        "cutoff_date": cutoff_date,
        "target_as_of": target_as_of,
        "entity_qids": entity_qids,
        "property_steps": property_steps,
        "snapshot_dates": snapshot_dates,
        "answer_qid": answer_qid,
    }
    return hashlib.sha256(canonical_json(contract).encode()).hexdigest()


class QuestionLedger:
    """SQLite ledger with idempotent upserts and explicit retry semantics."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS question_ledger ("
            "fingerprint TEXT PRIMARY KEY, schema_version TEXT NOT NULL, "
            "seed_id TEXT NOT NULL, model_id TEXT NOT NULL, cutoff_date TEXT NOT NULL, "
            "target_as_of TEXT NOT NULL, answer_qid TEXT NOT NULL, status TEXT NOT NULL, "
            "attempts INTEGER NOT NULL, first_seen_at TEXT NOT NULL, "
            "last_seen_at TEXT NOT NULL, semantic_identity_json TEXT NOT NULL, "
            "seed_json TEXT NOT NULL, audit_json TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS question_ledger_status_idx "
            "ON question_ledger(status)"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def get(self, fingerprint: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT fingerprint,seed_id,model_id,cutoff_date,target_as_of,"
            "answer_qid,status,attempts,first_seen_at,last_seen_at,"
            "semantic_identity_json,seed_json,audit_json "
            "FROM question_ledger WHERE fingerprint=?",
            (fingerprint,),
        ).fetchone()
        if row is None:
            return None
        keys = (
            "fingerprint", "seed_id", "model_id", "cutoff_date",
            "target_as_of", "answer_qid", "status", "attempts",
            "first_seen_at", "last_seen_at", "semantic_identity_json",
            "seed_json", "audit_json",
        )
        result = dict(zip(keys, row))
        for field in ("semantic_identity_json", "seed_json", "audit_json"):
            result[field.removesuffix("_json")] = json.loads(result.pop(field))
        return result

    def excludes(self, fingerprint: str, *, retry_rejected: bool = False) -> bool:
        record = self.get(fingerprint)
        if record is None or record["status"] in RETRYABLE_STATUSES:
            return False
        if retry_rejected and record["status"] in {
            "deterministic_reject", "judge_reject",
        }:
            return False
        return record["status"] in TERMINAL_STATUSES

    def record(
        self,
        *,
        fingerprint: str,
        seed_id: str,
        model_id: str,
        cutoff_date: str,
        target_as_of: str,
        answer_qid: str,
        status: str,
        semantic_identity: dict[str, Any],
        seed: dict[str, Any],
        audit: dict[str, Any],
    ) -> None:
        if status not in TERMINAL_STATUSES | RETRYABLE_STATUSES:
            raise ValueError(f"unknown question ledger status: {status!r}")
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO question_ledger("
            "fingerprint,schema_version,seed_id,model_id,cutoff_date,target_as_of,"
            "answer_qid,status,attempts,first_seen_at,last_seen_at,"
            "semantic_identity_json,seed_json,audit_json) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(fingerprint) DO UPDATE SET "
            "seed_id=excluded.seed_id,status=excluded.status,"
            "attempts=question_ledger.attempts+1,last_seen_at=excluded.last_seen_at,"
            "seed_json=excluded.seed_json,audit_json=excluded.audit_json",
            (
                fingerprint, LEDGER_SCHEMA_VERSION, seed_id, model_id,
                cutoff_date, target_as_of, answer_qid, status, 1, now, now,
                canonical_json(semantic_identity), canonical_json(seed),
                canonical_json(audit),
            ),
        )
        self._conn.commit()

    def counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status,COUNT(*) FROM question_ledger GROUP BY status ORDER BY status"
        ).fetchall()
        return {str(status): int(count) for status, count in rows}
