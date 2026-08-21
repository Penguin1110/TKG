"""Deterministic live-from-Start smoke for the public-only v2.2 runner.

The private case is deliberately loaded only after search returns.  Route B is
an unlisted alternative to the private Route A witness, so this exercises open-
world evaluation instead of replaying a prepared trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tkg.experiment.contracts import PageLink, PageSnapshot
from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_environment_v2 import (
    EnvironmentActionV2, TemporalWikipediaEnvironmentV2,
)
from tkg.experiment.temporal_eval_schema_v2 import (
    PublicTemporalCaseV2, StructuredSubmissionV2, SubmittedClaimV2,
    load_cases_v2,
)
from tkg.experiment.temporal_evaluation_v2 import (
    reference_route_diagnostics_v2, validate_structured_submission_v2,
)
from tkg.experiment.temporal_live_ranker_v22 import CallableLiveActionRankerV22
from tkg.experiment.temporal_live_runner_v22 import (
    LiveSearchConfigV22, run_live_temporal_search_v22,
)


SYNTHETIC_CASE_PATH = Path(
    "examples/temporal_eval_v2/synthetic_multiroute_case_v2.json"
)


def _page(
    title: str, revision_id: int, timestamp: str, content: str,
    links: list[str] | None = None,
) -> PageSnapshot:
    return PageSnapshot(
        title=title,
        page_id=revision_id,
        revision_id=revision_id,
        timestamp=timestamp,
        as_of=timestamp,
        content=content,
        links=[PageLink(target=value, anchor=value) for value in (links or [])],
        source_url=f"synthetic://revision/{revision_id}",
    )


class SyntheticLiveBackendV22:
    """In-memory temporal graph; it contains no private route metadata."""

    def __init__(self) -> None:
        start_links = ["Route B", *[f"Distractor {i:02d}" for i in range(30)], "Route A"]
        pages = [
            _page("Start", 1, "2023-12-31T00:00:00Z", "Start page.", start_links),
            _page("Route B", 19, "2023-12-31T00:00:00Z", "Old Route B page."),
            _page(
                "Route B", 20, "2025-01-01T12:00:00Z",
                "On 1 January 2025, Person X became the Club head coach.",
                ["Person X"],
            ),
            _page("Route A", 9, "2023-12-31T00:00:00Z", "Old Route A page."),
            _page(
                "Route A", 10, "2025-01-01T12:00:00Z",
                "Club appointed Person X as head coach on 1 January 2025.",
                ["Person X"],
            ),
            _page(
                "Person X", 30, "2020-01-01T00:00:00Z",
                "Person X was born in Answer City.",
            ),
        ]
        pages.extend(
            _page(
                f"Distractor {i:02d}", 100 + i, "2020-01-01T00:00:00Z",
                f"Distractor page {i:02d}.",
            )
            for i in range(30)
        )
        self.by_revision = {page.revision_id: page for page in pages}
        self.by_title: dict[str, list[PageSnapshot]] = {}
        for page in pages:
            self.by_title.setdefault(page.title.casefold(), []).append(page)
        for values in self.by_title.values():
            values.sort(key=lambda page: page.timestamp)

    def fetch_page(self, title: str, as_of: str | None = None) -> PageSnapshot:
        values = self.by_title[title.casefold()]
        if as_of is None:
            return values[-1]
        cutoff = as_of[:10]
        eligible = [page for page in values if page.timestamp[:10] <= cutoff]
        if not eligible:
            raise ValueError(f"no synthetic revision for {title} at {as_of}")
        return eligible[-1]

    def fetch_revision(self, revision_id: int) -> PageSnapshot:
        return self.by_revision[revision_id]

    def list_revision_metadata_page(
        self, title: str, from_date: str, to_date: str, *,
        cursor: str | None = None, page_size: int = 50,
    ) -> dict[str, Any]:
        if cursor is not None:
            return {"title": title, "revisions": [], "next_cursor": None}
        rows = [
            {"revision_id": page.revision_id, "timestamp": page.timestamp}
            for page in self.by_title[title.casefold()]
            if from_date <= page.timestamp[:10] <= to_date
        ]
        return {"title": title, "revisions": rows[:page_size], "next_cursor": None}


class SyntheticSubmissionProposerV22:
    """Visible-evidence-only fixture; it owns no private case or route."""

    def propose(
        self, public_case: PublicTemporalCaseV2,
        evidence_pages: list[dict[str, Any]], *, seed: int,
    ) -> tuple[StructuredSubmissionV2, dict[str, Any]]:
        del public_case, seed
        bridge = next((
            page for page in evidence_pages
            if "Person X became the Club head coach" in str(page.get("content"))
        ), None)
        tail = next((
            page for page in evidence_pages
            if "Person X was born in Answer City" in str(page.get("content"))
        ), None)
        if bridge is None or tail is None:
            return StructuredSubmissionV2(
                answer="",
                critical_claims=(),
                tail_claim=SubmittedClaimV2("", "", "", None, ()),
            ), {"fixture": "visible_evidence_only", "complete": False}
        return StructuredSubmissionV2(
            answer="Answer City",
            critical_claims=(SubmittedClaimV2(
                subject="Club",
                relation="appointed head coach",
                object="Person X",
                event_time="2025-01-01",
                supporting_evidence_ids=(str(bridge["evidence_id"]),),
                claim_id="bridge_1",
            ),),
            tail_claim=SubmittedClaimV2(
                subject="Person X",
                relation="place of birth",
                object="Answer City",
                event_time=None,
                supporting_evidence_ids=(str(tail["evidence_id"]),),
                claim_id="tail",
            ),
        ), {"fixture": "visible_evidence_only", "complete": True}


def synthetic_score_v22(
    public_case: PublicTemporalCaseV2, state: Any, action: EnvironmentActionV2,
) -> float:
    """Fixed public policy used only to test runner mechanics."""
    del public_case
    if action.kind == "SUBMIT_ANSWER":
        return 0.0
    if action.kind == "FOLLOW_LINK":
        target = str(action.params.get("page_title"))
        if state.current_page == "Start" and target == "Route B":
            return -0.1
        if state.current_page == "Route B" and target == "Person X":
            return -0.1
        return -10.0
    if action.kind == "SWITCH_SNAPSHOT":
        return -0.1 if int(action.params.get("revision_id", 0)) == 20 else -10.0
    if action.kind == "LIST_LINKS":
        return -0.05 if state.current_revision_id == 20 else -0.2
    if action.kind == "LIST_REVISIONS":
        return -0.15 if state.current_page == "Route B" else -1.0
    return -20.0


def run_synthetic_live_smoke_v22(
    case_path: str | Path = SYNTHETIC_CASE_PATH,
) -> dict[str, Any]:
    # Public projection is created before the private object is released below.
    public_payload = json.loads(Path(case_path).read_text(encoding="utf-8"))
    public_row = public_payload["cases"][0]
    public_case = PublicTemporalCaseV2(**{
        key: public_row[key]
        for key in (
            "case_id", "model_id", "question", "start_page", "cutoff_date",
            "target_date", "schema_version",
        )
        if key in public_row and key != "schema_version"
    })
    backend = SyntheticLiveBackendV22()
    result = run_live_temporal_search_v22(
        public_case=public_case,
        backend=backend,
        environment=TemporalWikipediaEnvironmentV2(backend),
        ranker=CallableLiveActionRankerV22(synthetic_score_v22),
        submission_proposer=SyntheticSubmissionProposerV22(),
        config=LiveSearchConfigV22(
            beam_width=1,
            max_expansions=12,
            max_actions_per_state=1,
            link_page_size=50,
            revision_page_size=50,
            dense_action_limit=30,
            seed=17,
        ),
    )

    # Private witnesses/routes enter only after the complete live search result exists.
    private_case = load_cases_v2(case_path)[0]
    if result.final_state.submitted is None:
        raise AssertionError("synthetic live runner did not submit")
    from tkg.experiment.temporal_eval_schema_v2 import structured_submission_from_dict

    submission = structured_submission_from_dict(result.final_state.submitted)
    transition_rows = [
        row for row in result.final_state.action_trace
        if row["action"]["kind"] in {"FOLLOW_LINK", "SWITCH_SNAPSHOT"}
    ]
    actions_valid = all(
        row["result"] == "ok"
        and (
            row["hyperlink_valid"] is True
            or row["revision_valid"] is True
        )
        for row in transition_rows
    )
    evaluation = validate_structured_submission_v2(
        case=private_case,
        submission=submission,
        trajectory_evidence=list(result.final_state.collected_evidence),
        trajectory_actions_valid=actions_valid,
    )
    diagnostics = reference_route_diagnostics_v2(
        route=private_case.reference_routes[0],
        funnel_steps=[
            step["action_funnel"] for step in result.audit_steps
            if "action_funnel" in step
        ],
        action_trace=list(result.final_state.action_trace),
        case=private_case,
        evaluation=evaluation,
        search_stop_reason=result.stop_reason,
    )
    action_kinds = [row["action"]["kind"] for row in result.final_state.action_trace]
    return {
        "schema_version": "open-world-temporal-live-synthetic-smoke-v2.2",
        "search": result.to_dict(),
        "posthoc_private_evaluation": evaluation,
        "posthoc_reference_diagnostics": diagnostics,
        "checks": {
            "started_from_public_start_state": True,
            "no_prebuilt_trajectory_supplied": True,
            "private_case_loaded_only_post_search": True,
            "expected_action_kinds": action_kinds == [
                "LIST_LINKS", "FOLLOW_LINK", "LIST_REVISIONS",
                "SWITCH_SNAPSHOT", "LIST_LINKS", "FOLLOW_LINK",
                "SUBMIT_ANSWER",
            ],
            "alternative_route_validated": bool(
                evaluation["end_to_end_success"]
                and diagnostics["alternative_valid_route_found"]
            ),
        },
        "formal_conclusion_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=str(SYNTHETIC_CASE_PATH))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert_new_output_path(args.output)
    result = run_synthetic_live_smoke_v22(args.case)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[done] synthetic live v2.2 smoke: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
