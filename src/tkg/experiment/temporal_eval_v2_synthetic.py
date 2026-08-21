"""Generate the deterministic alternative-route acceptance artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_environment_v2 import (
    EnvironmentActionPageV2, EnvironmentActionV2, action_funnel_record_v2,
    compact_solver_actions_v2,
)
from tkg.experiment.temporal_eval_schema_v2 import (
    StructuredSubmissionV2, SubmittedClaimV2, canonical_sha256, load_cases_v2,
)
from tkg.experiment.temporal_evaluation_v2 import (
    evidence_id_v2, reference_route_diagnostics_v2,
    validate_structured_submission_v2,
)


SYNTHETIC_RESULT_SCHEMA = "open-world-temporal-multiroute-synthetic-result-v2"


def generate(case_path: str, output_path: str) -> dict:
    assert_new_output_path(output_path)
    output = Path(output_path)
    if output.exists():
        raise ValueError("synthetic result output must be new")
    cases = load_cases_v2(case_path)
    if len(cases) != 1:
        raise ValueError("synthetic manifest must contain exactly one case")
    case = cases[0]
    route_b = {
        "title": "Route B", "page_id": 20, "revision_id": 20,
        "timestamp": "2025-01-01T00:00:00Z", "as_of": "2025-01-01",
        "content": "On 1 January 2025, Person X became the Club head coach.",
        "links": [{"target": "Person X", "anchor": "Person X"}],
        "source_url": "https://example.invalid/?oldid=20",
    }
    person = {
        "title": "Person X", "page_id": 30, "revision_id": 30,
        "timestamp": "2025-02-01T00:00:00Z", "as_of": "2025-02-01",
        "content": "Person X was born in Answer City.",
        "links": [{"target": "Answer City", "anchor": "Answer City"}],
        "source_url": "https://example.invalid/?oldid=30",
    }
    route_b["evidence_id"] = evidence_id_v2(route_b)
    person["evidence_id"] = evidence_id_v2(person)
    submission = StructuredSubmissionV2(
        answer="Answer City",
        critical_claims=(SubmittedClaimV2(
            claim_id="bridge_1", subject="Club",
            relation="appointed head coach", object="Person X",
            event_time="2025-01-01",
            supporting_evidence_ids=(str(route_b["evidence_id"]),),
        ),),
        tail_claim=SubmittedClaimV2(
            claim_id="tail", subject="Person X", relation="place of birth",
            object="Answer City", event_time=None,
            supporting_evidence_ids=(str(person["evidence_id"]),),
        ),
    )
    evaluation = validate_structured_submission_v2(
        case=case, submission=submission,
        trajectory_evidence=[route_b, person], trajectory_actions_valid=True,
    )

    follow_b = EnvironmentActionV2(
        "FOLLOW_LINK", {"page_title": "Route B"}, "Follow Route B", 0,
    )
    distractors = [
        EnvironmentActionV2(
            "FOLLOW_LINK", {"page_title": f"Distractor {index}"},
            f"Follow Distractor {index}", index + 1,
        ) for index in range(30)
    ]
    follow_a = EnvironmentActionV2(
        "FOLLOW_LINK", {"page_title": "Route A"}, "Follow Route A", 31,
    )
    start_actions = [follow_b, *distractors, follow_a]
    start_page = EnvironmentActionPageV2(
        action_kind="FOLLOW_LINK", cursor=None, next_cursor=None, page_size=32,
        items=tuple(start_actions), full_count=32,
        full_sha256=canonical_sha256([row.to_dict() for row in start_actions]),
        source_node=("Start", 1),
    )
    compacted_start, _ = compact_solver_actions_v2([start_page], dense_limit=30)
    start_scores = {
        row.action_id: (10.0 if row.action_id == follow_b.action_id else 0.0)
        for row in compacted_start
    }
    start_funnel = action_funnel_record_v2(
        environment_actions=start_actions,
        retrieved_pages=[start_page], dense_limit=30,
        ranker_scores=start_scores, expanded_action_ids=[follow_b.action_id],
        parent_node=("Start", 1),
    )

    follow_person = EnvironmentActionV2(
        "FOLLOW_LINK", {"page_title": "Person X"}, "Follow Person X", 0,
    )
    route_b_page = EnvironmentActionPageV2(
        action_kind="FOLLOW_LINK", cursor=None, next_cursor=None, page_size=30,
        items=(follow_person,), full_count=1,
        full_sha256=canonical_sha256([follow_person.to_dict()]),
        source_node=("Route B", 20),
    )
    route_b_funnel = action_funnel_record_v2(
        environment_actions=[follow_person], retrieved_pages=[route_b_page],
        dense_limit=30, ranker_scores={follow_person.action_id: 10.0},
        expanded_action_ids=[follow_person.action_id], parent_node=("Route B", 20),
    )
    trace = [
        {"kind": "FOLLOW_LINK", "params": {"page_title": "Route B"}},
        {"kind": "FOLLOW_LINK", "params": {"page_title": "Person X"}},
        {"kind": "SUBMIT_ANSWER", "params": submission.to_dict()},
    ]
    diagnostics = reference_route_diagnostics_v2(
        route=case.reference_routes[0],
        funnel_steps=[start_funnel, route_b_funnel],
        action_trace=trace, case=case, evaluation=evaluation,
        search_stop_reason="submit_answer",
    )
    result = {
        "schema_version": SYNTHETIC_RESULT_SCHEMA,
        "case_sha256": canonical_sha256(case.to_dict()),
        "public_inference_case": case.public_view().to_dict(),
        "private_reference_route": case.reference_routes[0].to_dict(),
        "trajectory": trace,
        "trajectory_evidence": [route_b, person],
        "action_funnels": [start_funnel, route_b_funnel],
        "structured_submission": submission.to_dict(),
        "evaluation": evaluation,
        "reference_diagnostics": diagnostics,
        "expected_contract": {
            "end_to_end_success": True,
            "reference_route_recalled": False,
            "alternative_valid_route_found": True,
        },
        "formal_benchmark_result": False,
    }
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = generate(args.case, args.output)
    print(json.dumps(result["expected_contract"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
