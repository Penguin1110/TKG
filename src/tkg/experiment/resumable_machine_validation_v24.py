"""Durable per-case orchestration for shortest-arena machine validation.

The underlying validator remains ``multihop_generation``.  This module only
adds process isolation, per-case checkpoints, shared request caches, and resume
semantics so an infrastructure failure cannot discard an otherwise completed
batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

SCHEMA_VERSION = "resumable-shortest-arena-v2.4"
VALIDITY_LEDGER_SCHEMA = "append-only-validity-ledger-v2.5"
COMPLETED_STATUSES = frozenset({
    "machine_pass_human_review_required",
    "deterministic_reject",
    "judge_reject",
})
PK_TERMINAL_STATUSES = frozenset({"admitted", "rejected"})
PENDING_STATUSES = frozenset({
    "pending",
    "pending_timeout",
    "pending_infrastructure_error",
    "pending_process_error",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_first_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            return dict(value) if isinstance(value, dict) else None
    return None


def _read_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("cases", []) if isinstance(value, dict) else []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _seed_ids(path: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".jsonl"):
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        values = parsed.get("seeds", []) if isinstance(parsed, dict) else parsed
    if not isinstance(values, list):
        raise ValueError("seed file must contain a list")
    result = [str(row.get("id") or "") for row in values if isinstance(row, dict)]
    if any(not seed_id for seed_id in result) or len(set(result)) != len(result):
        raise ValueError("seed IDs must be present and unique")
    return result


def _normalized_config(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    # Manifests written before v2.7 always ran per-case PK.
    result.setdefault("run_pk", True)
    return result


def _config_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        _normalized_config(value), sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


@dataclass(frozen=True)
class ValidationConfigV24:
    seed_file: str
    work_dir: str
    generator_model: str | None
    judge_model: str
    judge_min_confidence: float = 0.8
    timeout_seconds: float = 600.0
    request_interval: float = 1.0
    api_call_budget: int = 2000
    api_max_retries: int = 6
    api_backoff_base: float = 2.0
    api_backoff_max: float = 60.0
    backlink_branch_cap: int = 25
    arena_node_cap: int = 500
    max_attempts_per_case: int = 3
    retry_backoff_base: float = 30.0
    retry_backoff_max: float = 900.0
    target_pk_admitted: int = 10
    cache_path: str | None = None
    judge_cache_path: str | None = None
    pk_repeats: int = 5
    pk_temperatures: str = "0.0,0.2,0.5,0.7,1.0"
    pk_max_known_rate: float = 0.0
    pk_timeout_seconds: float = 900.0
    backlink_verify_workers: int = 1
    shortest_policy: str = "required"
    run_pk: bool = True


RunProcess = Callable[..., subprocess.CompletedProcess[str]]


class ResumableMachineValidationV24:
    def __init__(
        self, config: ValidationConfigV24, *,
        run_process: RunProcess = subprocess.run,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.work_dir = Path(config.work_dir)
        self.case_dir = self.work_dir / "cases"
        self.manifest_path = self.work_dir / "manifest.json"
        self.accepted_path = self.work_dir / "machine_pass_cases.json"
        self.validity_ledger_path = self.work_dir / "validity_v25.jsonl"
        self.cache_path = Path(config.cache_path) if config.cache_path else (
            self.work_dir / "wikipedia_requests.db"
        )
        self.judge_cache_path = (
            Path(config.judge_cache_path) if config.judge_cache_path else
            self.work_dir / "judge_requests.db"
        )
        self.run_process = run_process
        self.sleep_fn = sleep_fn

    def _new_manifest(self) -> dict[str, Any]:
        seed_sha = hashlib.sha256(
            Path(self.config.seed_file).read_bytes(),
        ).hexdigest()
        config = asdict(self.config)
        return {
            "schema_version": SCHEMA_VERSION,
            "seed_file": str(Path(self.config.seed_file).resolve()),
            "seed_file_sha256": seed_sha,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "config": config,
            "config_sha256": _config_sha256(config),
            "cases": {},
            "summary": {},
        }

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return self._new_manifest()
        value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        expected = hashlib.sha256(Path(self.config.seed_file).read_bytes()).hexdigest()
        if value.get("seed_file_sha256") != expected:
            raise ValueError("seed file changed since resumable run was created")
        stored_config = value.get("config")
        if not isinstance(stored_config, dict):
            raise ValueError("resumable manifest lacks a config object")
        current_config = asdict(self.config)
        if _config_sha256(stored_config) != _config_sha256(current_config):
            raise ValueError(
                "validation config changed since run creation; use a new work directory"
            )
        return dict(value)

    @staticmethod
    def _summary(manifest: dict[str, Any]) -> dict[str, int]:
        rows = list(manifest.get("cases", {}).values())
        summary = {
            "total": len(rows), "completed": 0, "pending": 0,
            "machine_pass": 0, "deterministic_reject": 0, "judge_reject": 0,
            "pk_admitted": 0, "pk_rejected": 0, "pk_pending": 0,
        }
        for row in rows:
            status = str(row.get("status", "pending"))
            if status in COMPLETED_STATUSES:
                summary["completed"] += 1
            else:
                summary["pending"] += 1
            if status == "machine_pass_human_review_required":
                summary["machine_pass"] += 1
                pk_status = str(row.get("pk_status") or "pending")
                if pk_status == "admitted":
                    summary["pk_admitted"] += 1
                elif pk_status == "rejected":
                    summary["pk_rejected"] += 1
                else:
                    summary["pk_pending"] += 1
            elif status in summary:
                summary[status] += 1
        return summary

    def _checkpoint(self, manifest: dict[str, Any]) -> None:
        manifest["updated_at"] = _utc_now()
        manifest["summary"] = self._summary(manifest)
        _atomic_json(self.manifest_path, manifest)
        accepted: dict[str, dict[str, Any]] = {}
        for seed_id, record in manifest.get("cases", {}).items():
            if record.get("status") != "machine_pass_human_review_required":
                continue
            for case in _read_cases(Path(str(record["cases_output"]))):
                accepted[str(case.get("id") or seed_id)] = case
        _atomic_json(self.accepted_path, {
            "schema_version": "machine-pass-cases-v2.4",
            "cases": [accepted[key] for key in sorted(accepted)],
            "count": len(accepted),
            "updated_at": _utc_now(),
        })
        admitted = {
            str(case.get("id")): case
            for case in accepted.values()
            if manifest.get("cases", {}).get(str(case.get("id")), {}).get(
                "pk_status"
            ) == "admitted"
        }
        _atomic_json(self.work_dir / "pk_admitted_cases.json", {
            "schema_version": "pk-admitted-fresh-cases-v2.4",
            "cases": [admitted[key] for key in sorted(admitted)],
            "count": len(admitted), "target": self.config.target_pk_admitted,
            "updated_at": _utc_now(),
        })

    def _append_validity_event(self, record: dict[str, Any]) -> None:
        """Append one immutable admission result, idempotently across resume."""
        if record.get("status") not in COMPLETED_STATUSES:
            return
        packet = record.get("packet") or {}
        contract = packet.get("validity_contract_v25")
        if not isinstance(contract, dict):
            return
        identity = {
            "seed_id": record.get("seed_id"), "status": record.get("status"),
            "contract_sha256": contract.get("contract_sha256"),
        }
        event_id = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        if self.validity_ledger_path.exists():
            for line in self.validity_ledger_path.read_text(encoding="utf-8").splitlines():
                if line.strip() and json.loads(line).get("event_id") == event_id:
                    return
        event = {
            "schema_version": VALIDITY_LEDGER_SCHEMA,
            "event_id": event_id, "recorded_at": _utc_now(), **identity,
            "attempt": record.get("attempts"), "validity_contract": contract,
            "search_space_diagnostic": packet.get("shortest_path"),
            "packet_output": record.get("packet_output"),
        }
        self.validity_ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.validity_ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _command(self, seed_id: str, attempt: int) -> tuple[list[str], dict[str, Path]]:
        safe_id = hashlib.sha256(seed_id.encode()).hexdigest()[:12]
        attempt_dir = self.case_dir / safe_id / f"attempt-{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "packet": attempt_dir / "packet.jsonl",
            "cases": attempt_dir / "cases.json",
            "usage": attempt_dir / "usage.jsonl",
            "stdout": attempt_dir / "stdout.log",
            "stderr": attempt_dir / "stderr.log",
        }
        command = [
            sys.executable, "-m", "tkg.experiment.multihop_generation",
            "--seed-file", self.config.seed_file,
            "--seed-id", seed_id,
            "--judge-model", self.config.judge_model,
            "--judge-min-confidence", str(self.config.judge_min_confidence),
            "--judge-workers", "1",
            "--cache-path", str(self.cache_path),
            "--judge-cache-path", str(self.judge_cache_path),
            "--request-interval", str(self.config.request_interval),
            "--api-call-budget", str(self.config.api_call_budget),
            "--api-max-retries", str(self.config.api_max_retries),
            "--api-backoff-base", str(self.config.api_backoff_base),
            "--api-backoff-max", str(self.config.api_backoff_max),
            "--backlink-branch-cap", str(self.config.backlink_branch_cap),
            "--arena-node-cap", str(self.config.arena_node_cap),
            "--backlink-verify-workers", str(self.config.backlink_verify_workers),
            "--shortest-policy", self.config.shortest_policy,
            "--output", str(paths["packet"]),
            "--cases-output", str(paths["cases"]),
            "--usage-output", str(paths["usage"]),
        ]
        if self.config.generator_model:
            command.extend(["--generator-model", self.config.generator_model])
        return command, paths

    def _run_one(
        self, seed_id: str, previous: dict[str, Any] | None,
    ) -> dict[str, Any]:
        attempt = int((previous or {}).get("attempts", 0)) + 1
        command, paths = self._command(seed_id, attempt)
        started = time.monotonic()
        base = {
            "seed_id": seed_id, "attempts": attempt,
            "started_at": _utc_now(), "command": command,
            "packet_output": str(paths["packet"]),
            "cases_output": str(paths["cases"]),
            "usage_output": str(paths["usage"]),
        }
        try:
            completed = self.run_process(
                command, cwd=str(Path.cwd()), text=True, capture_output=True,
                timeout=self.config.timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            paths["stdout"].write_text(stdout, encoding="utf-8")
            paths["stderr"].write_text(stderr, encoding="utf-8")
            return {
                **base, "status": "pending_timeout", "completed": False,
                "duration_seconds": time.monotonic() - started,
                "error": f"case exceeded {self.config.timeout_seconds:g}s timeout",
                "stdout": str(paths["stdout"]), "stderr": str(paths["stderr"]),
            }
        paths["stdout"].write_text(completed.stdout or "", encoding="utf-8")
        paths["stderr"].write_text(completed.stderr or "", encoding="utf-8")
        packet = _read_first_jsonl(paths["packet"])
        if packet is None:
            status = "pending_process_error"
            error = f"validator returned {completed.returncode} without packet"
        elif packet.get("status") == "infrastructure_error":
            status = "pending_infrastructure_error"
            error = "; ".join(str(item) for item in packet.get("deterministic_errors", []))
            if packet.get("judge_error"):
                error = f"{error}; {packet['judge_error']}".strip("; ")
        else:
            status = str(packet.get("status") or "pending_process_error")
            error = None
        return {
            **base, "status": status, "completed": status in COMPLETED_STATUSES,
            "duration_seconds": time.monotonic() - started,
            "returncode": completed.returncode, "packet": packet, "error": error,
            "stdout": str(paths["stdout"]), "stderr": str(paths["stderr"]),
        }

    @staticmethod
    def _latest_pk_gate(path: Path, seed_id: str) -> dict[str, Any] | None:
        if not path.exists():
            return None
        latest = None
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("slot") == "pk_gate" and row.get("case_id") == seed_id:
                latest = dict(row)
        return latest

    def _run_pk(self, seed_id: str, record: dict[str, Any]) -> dict[str, Any]:
        pk_attempt = int(record.get("pk_attempts", 0)) + 1
        safe_id = hashlib.sha256(seed_id.encode()).hexdigest()[:12]
        attempt_dir = self.case_dir / safe_id / f"pk-attempt-{pk_attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        results = attempt_dir / "pk_results.jsonl"
        scores = attempt_dir / "pk_scores.csv"
        usage = attempt_dir / "pk_usage.jsonl"
        stdout_path = attempt_dir / "stdout.log"
        stderr_path = attempt_dir / "stderr.log"
        cases = _read_cases(Path(str(record["cases_output"])))
        if len(cases) != 1:
            return {
                **record, "pk_status": "pending_case_artifact_error",
                "pk_error": "machine-pass artifact must contain exactly one case",
                "pk_attempts": pk_attempt,
            }
        allowed = cases[0].get("knowledge_cutoff", {}).get("model_ids", [])
        if not isinstance(allowed, list) or len(allowed) != 1:
            return {
                **record, "pk_status": "pending_case_artifact_error",
                "pk_error": "case must bind exactly one tested model",
                "pk_attempts": pk_attempt,
            }
        command = [
            sys.executable, "-m", "tkg.experiment.temporal_runner",
            "--models", str(allowed[0]), "--judge-model", self.config.judge_model,
            "--cases", str(record["cases_output"]), "--case-ids", seed_id,
            "--pk-only", "--pk-repeats", str(self.config.pk_repeats),
            "--pk-temperatures", self.config.pk_temperatures,
            "--pk-max-known-rate", str(self.config.pk_max_known_rate),
            "--output", str(results), "--score-output", str(scores),
            "--usage-output", str(usage),
        ]
        started = time.monotonic()
        try:
            completed = self.run_process(
                command, cwd=str(Path.cwd()), text=True, capture_output=True,
                timeout=self.config.pk_timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            stdout_path.write_text(stdout, encoding="utf-8")
            stderr_path.write_text(stderr, encoding="utf-8")
            return {
                **record, "pk_status": "pending_timeout", "pk_attempts": pk_attempt,
                "pk_duration_seconds": time.monotonic() - started,
                "pk_error": f"PK exceeded {self.config.pk_timeout_seconds:g}s timeout",
                "pk_results_output": str(results),
            }
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        gate = self._latest_pk_gate(results, seed_id)
        if gate is None:
            pk_status = "pending_process_error"
            error = f"PK returned {completed.returncode} without pk_gate"
        else:
            pk_status = "admitted" if gate.get("passed") is True else "rejected"
            error = None
        return {
            **record, "pk_status": pk_status, "pk_attempts": pk_attempt,
            "pk_duration_seconds": time.monotonic() - started,
            "pk_returncode": completed.returncode, "pk_gate": gate,
            "pk_error": error, "pk_results_output": str(results),
            "pk_usage_output": str(usage), "pk_stdout": str(stdout_path),
            "pk_stderr": str(stderr_path),
        }

    def run(self, *, seed_ids: Sequence[str] | None = None) -> dict[str, Any]:
        self.case_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._load_manifest()
        available = _seed_ids(self.config.seed_file)
        requested = list(seed_ids) if seed_ids else available
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise ValueError(f"unknown seed IDs: {', '.join(unknown)}")
        for seed_id in requested:
            previous = manifest["cases"].get(seed_id)
            if (
                previous and previous.get("status") in COMPLETED_STATUSES
                and previous.get("status") != "machine_pass_human_review_required"
            ):
                continue
            if (
                previous
                and previous.get("status") == "machine_pass_human_review_required"
                and previous.get("pk_status") in PK_TERMINAL_STATUSES
            ):
                continue
            if self._summary(manifest)["pk_admitted"] >= self.config.target_pk_admitted:
                break
            if previous and previous.get("status") == "machine_pass_human_review_required":
                if not self.config.run_pk:
                    continue
                manifest["cases"][seed_id] = self._run_pk(seed_id, previous)
                self._checkpoint(manifest)
                continue
            if int((previous or {}).get("attempts", 0)) >= self.config.max_attempts_per_case:
                continue
            if previous and previous.get("status") in PENDING_STATUSES:
                exponent = max(0, int(previous.get("attempts", 1)) - 1)
                delay = min(
                    self.config.retry_backoff_max,
                    self.config.retry_backoff_base * (2 ** exponent),
                )
                if delay > 0:
                    self.sleep_fn(delay)
            manifest["cases"][seed_id] = self._run_one(seed_id, previous)
            self._append_validity_event(manifest["cases"][seed_id])
            self._checkpoint(manifest)
            if manifest["cases"][seed_id].get("status") == (
                "machine_pass_human_review_required"
            ) and self.config.run_pk:
                manifest["cases"][seed_id] = self._run_pk(
                    seed_id, manifest["cases"][seed_id],
                )
                self._checkpoint(manifest)
        self._checkpoint(manifest)
        return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--generator-model")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-min-confidence", type=float, default=0.8)
    parser.add_argument("--case-timeout", type=float, default=600.0)
    parser.add_argument("--request-interval", type=float, default=1.0)
    parser.add_argument("--api-call-budget", type=int, default=2000)
    parser.add_argument("--api-max-retries", type=int, default=6)
    parser.add_argument("--api-backoff-base", type=float, default=2.0)
    parser.add_argument("--api-backoff-max", type=float, default=60.0)
    parser.add_argument("--backlink-branch-cap", type=int, default=25)
    parser.add_argument("--arena-node-cap", type=int, default=500)
    parser.add_argument("--max-attempts-per-case", type=int, default=3)
    parser.add_argument("--retry-backoff-base", type=float, default=30.0)
    parser.add_argument("--retry-backoff-max", type=float, default=900.0)
    parser.add_argument("--target-pk-admitted", type=int, default=10)
    parser.add_argument("--cache-path")
    parser.add_argument("--judge-cache-path")
    parser.add_argument("--pk-repeats", type=int, default=5)
    parser.add_argument("--pk-temperatures", default="0.0,0.2,0.5,0.7,1.0")
    parser.add_argument("--pk-max-known-rate", type=float, default=0.0)
    parser.add_argument("--pk-timeout", type=float, default=900.0)
    parser.add_argument("--backlink-verify-workers", type=int, default=1)
    parser.add_argument(
        "--shortest-policy", choices=("required", "diagnostic", "skip"),
        default="required",
    )
    parser.add_argument(
        "--skip-pk", action="store_true",
        help="defer PK to a separately deduplicated spine-level admission stage",
    )
    parser.add_argument("--seed-id", action="append", default=[])
    args = parser.parse_args()
    if args.case_timeout <= 0 or args.max_attempts_per_case <= 0:
        parser.error("timeout and max attempts must be positive")
    config = ValidationConfigV24(
        seed_file=args.seed_file, work_dir=args.work_dir,
        generator_model=args.generator_model, judge_model=args.judge_model,
        judge_min_confidence=args.judge_min_confidence,
        timeout_seconds=args.case_timeout, request_interval=args.request_interval,
        api_call_budget=args.api_call_budget, api_max_retries=args.api_max_retries,
        api_backoff_base=args.api_backoff_base, api_backoff_max=args.api_backoff_max,
        backlink_branch_cap=args.backlink_branch_cap,
        arena_node_cap=args.arena_node_cap,
        max_attempts_per_case=args.max_attempts_per_case,
        retry_backoff_base=args.retry_backoff_base,
        retry_backoff_max=args.retry_backoff_max,
        target_pk_admitted=args.target_pk_admitted,
        cache_path=args.cache_path, judge_cache_path=args.judge_cache_path,
        pk_repeats=args.pk_repeats, pk_temperatures=args.pk_temperatures,
        pk_max_known_rate=args.pk_max_known_rate,
        pk_timeout_seconds=args.pk_timeout,
        backlink_verify_workers=args.backlink_verify_workers,
        shortest_policy=args.shortest_policy,
        run_pk=not args.skip_pk,
    )
    manifest = ResumableMachineValidationV24(config).run(seed_ids=args.seed_id)
    print(json.dumps(manifest["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
