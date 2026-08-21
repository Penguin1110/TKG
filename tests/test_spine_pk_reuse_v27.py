from tkg.experiment.spine_pk_reuse_v27 import apply, plan


def _seed(seed_id: str, tail: str) -> dict:
    return {
        "id": seed_id, "model_id": "model", "cutoff_date": "2024-01-01",
        "target_as_of": "2026-01-01",
        "hops": [{"bridge": 1}, {"bridge": 2}, {"bridge": 3}, {"tail": tail}],
    }


def _case(case_id: str, bridge_question: str = "Who became leader?") -> dict:
    return {
        "id": case_id,
        "prior_knowledge_contract": {
            "probes": [{
                "id": "bridge_1", "role": "critical_bridge",
                "objective": "must_be_unknown", "question": bridge_question,
                "answer_aliases": ["Person"], "event_date": "2025-01-01",
                "hop_index": 1,
            }],
        },
    }


def test_pk_is_planned_once_per_identical_bridge_and_reused() -> None:
    seeds = [_seed("case-a", "birthplace"), _seed("case-b", "spouse")]
    cases = [_case("case-a"), _case("case-b")]
    representatives, mapping = plan(seeds, cases)
    assert representatives["count"] == 1
    assert mapping["case_count"] == 2
    assert mapping["spine_count"] == 1
    admitted, ledger = apply(mapping, [{
        "slot": "pk_gate", "case_id": "case-a", "passed": True,
        "critical_bridge_known_rate": 0.0,
    }], cases)
    assert admitted["count"] == 2
    assert [row["pk_gate_reused"] for row in ledger] == [False, True]


def test_missing_or_failed_representative_gate_does_not_admit() -> None:
    cases = [_case("case-a")]
    representatives, mapping = plan([_seed("case-a", "x")], cases)
    assert representatives["count"] == 1
    admitted, ledger = apply(mapping, [], cases)
    assert admitted["count"] == 0
    assert ledger[0]["pk_status"] == "pending_missing_representative_gate"


def test_different_critical_probe_contracts_are_not_reused() -> None:
    seeds = [_seed("case-a", "birthplace"), _seed("case-b", "spouse")]
    cases = [_case("case-a", "Who became leader?"), _case("case-b", "Who took office?")]
    representatives, mapping = plan(seeds, cases)
    assert representatives["count"] == 2
    assert mapping["spine_count"] == 2


def test_existing_gate_case_is_preferred_as_representative() -> None:
    seeds = [_seed("case-a", "birthplace"), _seed("case-b", "spouse")]
    cases = [_case("case-a"), _case("case-b")]
    representatives, mapping = plan(
        seeds, cases, preferred_representatives={"case-b"},
    )
    assert representatives["cases"][0]["id"] == "case-b"
    assert mapping["groups"][0]["representative_case_id"] == "case-b"
