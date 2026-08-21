from tkg.experiment.candidate_source_comparison import compare_candidate_sources


def _packet(source: str, status: str, *, redirect: bool = False):
    return {
        "status": status,
        "category": "politics",
        "selection_metadata": {"candidate_source": source},
        "deterministic_errors": [],
        "chain": [{
            "source_title": "Canonical source",
            "target_title": "Target",
            "requested_source_title": "Source alias" if redirect else "Canonical source",
            "requested_target_title": "Target",
            "relation_family": "politics",
            "prior_relation_contrast_verified": True,
        }],
    }


def test_candidate_source_report_keeps_discovery_separate_from_proof():
    report = compare_candidate_sources([
        _packet("wikipedia_first", "machine_pass_human_review_required"),
        _packet("wikidata_first", "deterministic_pass", redirect=True),
    ])

    assert report["candidate_count"] == 2
    assert report["comparison_is_conclusive"] is False
    assert report["sample_size_warning"]
    assert report["sources"]["wikipedia_first"]["judge_pass_rate"] == 1.0
    assert report["sources"]["wikidata_first"]["verified_redirect_hops"] == 1
    assert "Wikipedia remains the formal evidence" in report["interpretation"]


def test_candidate_source_report_preserves_rejection_reasons():
    packet = _packet("wikidata_first", "deterministic_reject")
    packet["deterministic_errors"] = ["hop 2: future target already visible"]

    report = compare_candidate_sources([packet])

    assert report["sources"]["wikidata_first"]["deterministic_pass_count"] == 0
    assert report["sources"]["wikidata_first"]["deterministic_error_counts"] == {
        "hop 2: future target already visible": 1,
    }
