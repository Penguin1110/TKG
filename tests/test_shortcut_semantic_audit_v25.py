from __future__ import annotations

import json
from pathlib import Path

import pytest

from tkg.experiment.shortcut_semantic_audit_v25 import audit_case, freeze_inputs


class UnusedBackend:
    pass


def _case() -> dict:
    page = {
        "title": "Anchor", "revision_id": 1, "timestamp": "2024-05-01T00:00:00Z",
        "as_of": "2024-06-01", "content": "The answer is City.", "links": [],
    }
    return {
        "id": "case", "expected_navigation_distance": 3,
        "knowledge_cutoff": {"cutoff_date": "2024-06-01"},
        "temporal_waypoints": [{
            "title": "Anchor", "revision_id": 1, "as_of": "2024-06-01",
            "role": "relation_source", "relation_hop": 0,
        }],
        "reasoning_chain": [
            {"evidence": "New Person became leader."},
            {"evidence": "New Person was born in City."},
        ],
        "prior_knowledge_contract": {"probes": [
            {"role": "critical_bridge", "hop_index": 0},
            {"role": "tail", "hop_index": 1},
        ]},
        "frozen_wikipedia_evidence": {"pages": [page]},
    }


def test_answer_only_visibility_does_not_count_as_validated_success() -> None:
    diagnostic = {"search_space_diagnostic": {
        "shortcut_status": "SHORTCUT_FOUND", "findings": [{
            "kind": "start_revision_contains_final_answer", "matched_aliases": ["City"],
        }],
    }}
    result = audit_case(_case(), diagnostic, UnusedBackend())  # type: ignore[arg-type]
    candidate = result["candidate_shortcuts"][0]
    assert candidate["classification"] == "ANSWER_ONLY_SHORTCUT"
    assert candidate["structured_evaluator_status"] == "must_fail_without_bridge_evidence"
    assert result["disposition"] == "retain"


def test_freeze_refuses_overwrite(tmp_path: Path) -> None:
    cases = tmp_path / "cases.json"
    ledger = tmp_path / "ledger.jsonl"
    cases.write_text(json.dumps({"cases": [_case()]}), encoding="utf-8")
    ledger.write_text(json.dumps({"seed_id": "case"}) + "\n", encoding="utf-8")
    output = tmp_path / "freeze"
    frozen, manifest = freeze_inputs(
        case_paths=[cases], ledger_paths=[ledger], output_dir=output,
    )
    assert frozen.exists() and manifest.exists()
    with pytest.raises(FileExistsError):
        freeze_inputs(case_paths=[cases], ledger_paths=[ledger], output_dir=output)
