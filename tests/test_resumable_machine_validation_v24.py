from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tkg.experiment.resumable_machine_validation_v24 import (
    ResumableMachineValidationV24, ValidationConfigV24,
)


def _seed_file(path: Path) -> Path:
    path.write_text(json.dumps({"seeds": [{
        "id": "seed-1", "model_id": "openai/gpt-4.1-mini", "cutoff_date": "2024-06-01",
        "target_as_of": "2025-01-01", "anchor_label": "Anchor",
        "answer_kind": "entity", "category": "test", "old_answer_keywords": [],
        "hops": [{
            "source_title": "Anchor", "target_title": "Person",
            "target_aliases": ["Person"], "relation": "successor",
            "relative_clause": "the successor of {source}", "as_of": "2025-01-01",
            "evidence": "Person succeeded the former officeholder.",
        }, {
            "source_title": "Person", "target_title": "City",
            "target_aliases": ["City"], "relation": "place of birth",
            "relative_clause": "the birthplace of {source}", "as_of": "2025-01-01",
            "evidence": "Person was born in City.",
        }],
    }]}), encoding="utf-8")
    return path


def _config(seed_file: Path, work: Path) -> ValidationConfigV24:
    return ValidationConfigV24(
        seed_file=str(seed_file), work_dir=str(work), generator_model=None,
        judge_model="judge/model", retry_backoff_base=0, timeout_seconds=1,
    )


def test_completed_case_is_checkpointed_and_skipped_on_resume(tmp_path: Path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(command)
        if "tkg.experiment.temporal_runner" in command:
            output = Path(command[command.index("--output") + 1])
            output.write_text(json.dumps({
                "slot": "pk_gate", "case_id": "seed-1", "passed": True,
                "critical_bridge_known_rate": 0.0,
            }) + "\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "PK pass", "")
        packet = Path(command[command.index("--output") + 1])
        cases = Path(command[command.index("--cases-output") + 1])
        packet.write_text(json.dumps({
            "seed_id": "seed-1", "status": "machine_pass_human_review_required",
            "deterministic_errors": [],
        }) + "\n", encoding="utf-8")
        cases.write_text(json.dumps({"cases": [{
            "id": "seed-1", "knowledge_cutoff": {
                "model_ids": ["openai/gpt-4.1-mini"],
            },
        }]}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    seed_file = _seed_file(tmp_path / "seeds.json")
    runner = ResumableMachineValidationV24(
        _config(seed_file, tmp_path / "run"), run_process=fake_run,
    )
    first = runner.run()
    second = runner.run()
    assert len(calls) == 2
    assert first["summary"]["machine_pass"] == 1
    assert first["summary"]["pk_admitted"] == 1
    assert second["summary"]["completed"] == 1
    accepted = json.loads(runner.accepted_path.read_text(encoding="utf-8"))
    assert accepted["count"] == 1
    admitted = json.loads(
        (runner.work_dir / "pk_admitted_cases.json").read_text(encoding="utf-8")
    )
    assert admitted["count"] == 1


def test_timeout_is_pending_and_can_resume(tmp_path: Path) -> None:
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        if calls == 1:
            raise subprocess.TimeoutExpired(command, 1)
        packet = Path(command[command.index("--output") + 1])
        cases = Path(command[command.index("--cases-output") + 1])
        packet.write_text(json.dumps({
            "seed_id": "seed-1", "status": "deterministic_reject",
            "deterministic_errors": ["invalid"],
        }) + "\n", encoding="utf-8")
        cases.write_text('{"cases": []}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 1, "", "")

    seed_file = _seed_file(tmp_path / "seeds.json")
    runner = ResumableMachineValidationV24(
        _config(seed_file, tmp_path / "run"), run_process=fake_run,
    )
    first = runner.run()
    assert first["cases"]["seed-1"]["status"] == "pending_timeout"
    second = runner.run()
    assert second["cases"]["seed-1"]["status"] == "deterministic_reject"
    assert calls == 2


def test_pk_can_be_deferred_for_spine_level_deduplication(tmp_path: Path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(command)
        packet = Path(command[command.index("--output") + 1])
        cases = Path(command[command.index("--cases-output") + 1])
        packet.write_text(json.dumps({
            "seed_id": "seed-1", "status": "machine_pass_human_review_required",
            "deterministic_errors": [],
        }) + "\n", encoding="utf-8")
        cases.write_text(json.dumps({"cases": [{
            "id": "seed-1", "knowledge_cutoff": {
                "model_ids": ["openai/gpt-4.1-mini"],
            },
        }]}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    seed_file = _seed_file(tmp_path / "seeds.json")
    config = _config(seed_file, tmp_path / "run")
    config = ValidationConfigV24(**{**config.__dict__, "run_pk": False})
    runner = ResumableMachineValidationV24(config, run_process=fake_run)
    first = runner.run()
    second = runner.run()
    assert len(calls) == 1
    assert first["summary"]["machine_pass"] == 1
    assert first["summary"]["pk_pending"] == 1
    assert second["summary"]["machine_pass"] == 1


def test_resume_fails_closed_when_config_changes(tmp_path: Path) -> None:
    seed_file = _seed_file(tmp_path / "seeds.json")
    work = tmp_path / "run"
    first = ResumableMachineValidationV24(_config(seed_file, work))
    first._checkpoint(first._new_manifest())
    changed = _config(seed_file, work)
    changed = ValidationConfigV24(**{**changed.__dict__, "run_pk": False})
    second = ResumableMachineValidationV24(changed)
    try:
        second._load_manifest()
    except ValueError as exc:
        assert "config changed" in str(exc)
    else:
        raise AssertionError("config drift must fail closed")


def test_v25_validity_ledger_is_append_only_and_idempotent(tmp_path: Path) -> None:
    seed_file = _seed_file(tmp_path / "seeds.json")
    runner = ResumableMachineValidationV24(_config(seed_file, tmp_path / "run"))
    contract = {"contract_sha256": "abc", "passed": True}
    record = {
        "seed_id": "seed-1", "status": "machine_pass_human_review_required",
        "attempts": 1, "packet_output": "packet.jsonl",
        "packet": {"validity_contract_v25": contract, "shortest_path": {
            "shortest_path_status": "incomplete",
        }},
    }
    runner._append_validity_event(record)
    runner._append_validity_event(record)
    rows = runner.validity_ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    event = json.loads(rows[0])
    assert event["validity_contract"] == contract
    assert event["search_space_diagnostic"]["shortest_path_status"] == "incomplete"
