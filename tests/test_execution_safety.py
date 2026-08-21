"""Concurrency and model-usage accounting contracts."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

from tkg.api import openrouter
from tkg.experiment.results import JsonlResultStore
from tkg.experiment import temporal_runner


def test_jsonl_result_store_keeps_concurrent_rows_atomic(tmp_path):
    path = tmp_path / "threaded.jsonl"
    store = JsonlResultStore(str(path))
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(
                    store.write, slot="row", case_id=f"case-{index}",
                    model="m", arm="temporal", index=index,
                )
                for index in range(400)
            ]
            for future in futures:
                future.result()
    finally:
        store.close()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 400
    assert {row["index"] for row in rows} == set(range(400))


def test_openrouter_ledger_records_context_tokens_cost_and_cache(tmp_path, monkeypatch):
    class Response:
        ok = True

        @staticmethod
        def json():
            return {
                "id": "generation-1", "model": "provider/model-v2",
                "provider": "provider", "choices": [{"message": {
                    "role": "assistant", "content": "answer",
                }}],
                "usage": {
                    "prompt_tokens": 11, "completion_tokens": 3,
                    "total_tokens": 14, "cost": 0.0012,
                },
            }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-must-not-be-logged")
    monkeypatch.setattr(openrouter.requests, "post", lambda *args, **kwargs: Response())
    path = tmp_path / "usage.jsonl"
    ledger = openrouter.UsageLedger(str(path), metadata={"contract_hash": "abc"})
    openrouter.set_usage_ledger(ledger)
    try:
        with openrouter.model_call_context(role="judge", case_id="case-1"):
            assert openrouter.call_model("provider/model", []) == "answer"
            openrouter.record_usage_event(
                "cache_hit", requested_model="provider/model", cache_key="cache-1",
            )
    finally:
        openrouter.set_usage_ledger(None)
        ledger.close()
    raw = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw.splitlines()]
    assert [row["event_type"] for row in rows] == ["api_call", "cache_hit"]
    assert rows[0]["prompt_tokens"] == 11
    assert rows[0]["completion_tokens"] == 3
    assert rows[0]["total_tokens"] == 14
    assert rows[0]["cost"] == 0.0012
    assert rows[0]["context"] == {"role": "judge", "case_id": "case-1"}
    assert "test-secret-must-not-be-logged" not in raw


def test_pk_only_never_opens_wikipedia_or_builds_navigation(tmp_path, monkeypatch):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"cases": [{
        "id": "pk-only-case",
        "temporal_question": "Who became the new leader?",
        "wikipedia_title": "Example",
        "wikipedia_before": "2024-06-01",
        "wikipedia_as_of": "2025-01-01",
        "old_answer_keywords": ["Old Leader"],
        "new_answer_keywords": ["New Leader"],
        "knowledge_cutoff": {
            "cutoff_date": "2024-06-01",
            "model_ids": ["openai/gpt-4.1-mini"],
        },
    }]}), encoding="utf-8")
    output = tmp_path / "pk.jsonl"
    scores = tmp_path / "pk.csv"

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("PK-only must not construct WikipediaPageBackend")

    def forbidden_review(*args, **kwargs):
        raise AssertionError("PK-only must not resolve human review")

    def fake_pk(**kwargs):
        gate = {
            "n": 1, "passed": True, "reason": "target_answer_not_known",
            "stick_new_count": 0, "stick_new_rate": 0.0,
            "stick_old_count": 1, "stick_old_rate": 1.0,
            "other_count": 0,
        }
        kwargs["store"].write(
            slot="pk_gate", case_id=kwargs["case"]["id"],
            model=kwargs["model"], arm="admission", **gate,
        )
        return gate

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(temporal_runner, "WikipediaPageBackend", forbidden_backend)
    monkeypatch.setattr(temporal_runner, "resolve_human_reviews", forbidden_review)
    monkeypatch.setattr(temporal_runner, "run_pk_admission", fake_pk)
    monkeypatch.setattr(sys, "argv", [
        "tkg-run", "--models", "openai/gpt-4.1-mini",
        "--judge-model", "openai/gpt-4.1", "--cases", str(cases),
        "--pk-only", "--pk-repeats", "1", "--output", str(output),
        "--score-output", str(scores),
    ])
    assert temporal_runner.main() == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert any(row["slot"] == "pk_gate" for row in rows)
    assert not any(row["slot"] == "human_review_admission" for row in rows)
    assert scores.is_file()
