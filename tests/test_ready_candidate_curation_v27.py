from tkg.experiment.ready_candidate_curation_v27 import curate


def _candidate(candidate_id: str, spine: str, tail: str) -> dict:
    return {
        "id": candidate_id,
        "question_fingerprint": candidate_id,
        "relation_property": "P1308",
        "public_anchor": spine,
        "tail_property": tail,
        "private_chain": [
            {"source_qid": spine, "target_qid": "Q1", "property_id": "P1308"},
            {"source_qid": "Q1", "target_qid": spine, "property_id": "P1308", "event_date": "2025-01-01"},
            {"source_qid": spine, "target_qid": "Q2", "property_id": "P1308", "event_date": "2025-02-01"},
        ],
    }


def test_curate_excludes_sealed_and_round_robins_spines() -> None:
    rows = [
        _candidate("kgcand_a1", "Q10", "P19"),
        _candidate("kgcand_a2", "Q10", "P26"),
        _candidate("kgcand_b1", "Q20", "P19"),
        _candidate("kgcand_b2", "Q20", "P26"),
    ]
    selected = curate(rows, excluded_ids={"kgcand_b1"}, limit=3)
    assert {row["id"] for row in selected} == {"kgcand_a1", "kgcand_a2", "kgcand_b2"}
    assert selected[0]["public_anchor"] != selected[1]["public_anchor"]


def test_curate_deduplicates_fingerprint() -> None:
    row = _candidate("kgcand_a1", "Q10", "P19")
    duplicate = dict(row, id="kgcand_other")
    duplicate["question_fingerprint"] = row["question_fingerprint"]
    assert len(curate([row, duplicate], excluded_ids=set(), limit=10)) == 1


def test_curate_excludes_all_tail_variants_of_certified_spine() -> None:
    rows = [
        _candidate("kgcand_a1", "Q10", "P19"),
        _candidate("kgcand_a2", "Q10", "P26"),
        _candidate("kgcand_b1", "Q20", "P19"),
    ]
    from tkg.experiment.ready_candidate_curation_v27 import _spine_key

    selected = curate(
        rows, excluded_ids=set(), excluded_spines={_spine_key(rows[0])}, limit=10,
    )
    assert [row["id"] for row in selected] == ["kgcand_b1"]
