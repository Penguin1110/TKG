"""Fresh deterministic fixtures and artifact generator for joint live v2.3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tkg.experiment.contracts import PageLink, PageSnapshot
from tkg.experiment.joint_controller_v23 import (
    CallableJointControllerV23, JointCandidateActionV23, SUBMIT_SLOT_ID_V23,
)
from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_environment_v2 import TemporalWikipediaEnvironmentV2
from tkg.experiment.temporal_eval_schema_v2 import (
    PublicTemporalCaseV2, StructuredSubmissionV2, SubmittedClaimV2,
    load_cases_v2, structured_submission_from_dict,
)
from tkg.experiment.temporal_evaluation_v2 import (
    reference_route_diagnostics_v2, validate_structured_submission_v2,
)
from tkg.experiment.temporal_live_runner_v23 import (
    LiveSearchConfigV23, run_live_temporal_search_v23,
)


SYNTHETIC_CASE_V23 = Path(
    "examples/temporal_eval_v23/synthetic_joint_case_v23.json"
)


def _page(
    title: str, revision_id: int, timestamp: str, content: str,
    links: list[str] | None = None,
) -> PageSnapshot:
    return PageSnapshot(
        title=title, page_id=revision_id, revision_id=revision_id,
        timestamp=timestamp, as_of=timestamp, content=content,
        links=[PageLink(target=target, anchor=target) for target in (links or [])],
        source_url=f"synthetic-v23://revision/{revision_id}",
    )


class SyntheticJointBackendV23:
    def __init__(self) -> None:
        hub_links = [
            "Alternative Route",
            *[f"Joint Distractor {index:02d}" for index in range(30)],
            "Reference Route",
        ]
        pages = [
            _page("Joint Hub", 101, "2023-12-31T00:00:00Z", "Joint hub.", hub_links),
            _page("Alternative Route", 110, "2023-12-31T00:00:00Z", "Old alternative."),
            _page(
                "Alternative Route", 111, "2025-02-02T12:00:00Z",
                "On 2 February 2025, Scientist Z became Director of Lab.",
                ["Scientist Z"],
            ),
            _page("Reference Route", 130, "2023-12-31T00:00:00Z", "Old reference."),
            _page(
                "Reference Route", 131, "2025-02-02T12:00:00Z",
                "Lab appointed Scientist Z as director on 2 February 2025.",
                ["Scientist Z"],
            ),
            _page(
                "Scientist Z", 120, "2020-01-01T00:00:00Z",
                "Scientist Z was born in Harbor City.", ["Further Context"],
            ),
            _page("Further Context", 121, "2020-01-01T00:00:00Z", "No new facts."),
        ]
        pages.extend(
            _page(
                f"Joint Distractor {index:02d}", 200 + index,
                "2020-01-01T00:00:00Z", f"Joint distractor {index:02d}.",
            )
            for index in range(30)
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
        eligible = [page for page in values if page.timestamp[:10] <= as_of[:10]]
        if not eligible:
            raise ValueError(f"no synthetic v2.3 revision for {title} at {as_of}")
        return eligible[-1]

    def fetch_revision(self, revision_id: int) -> PageSnapshot:
        return self.by_revision[revision_id]

    def list_revision_metadata_page(
        self, title: str, from_date: str, to_date: str, *,
        cursor: str | None = None, page_size: int = 50,
    ) -> dict[str, Any]:
        if cursor is not None:
            return {"title": title, "revisions": [], "next_cursor": None}
        revisions = [
            {"revision_id": page.revision_id, "timestamp": page.timestamp}
            for page in self.by_title[title.casefold()]
            if from_date <= page.timestamp[:10] <= to_date
        ]
        return {"title": title, "revisions": revisions[:page_size], "next_cursor": None}


def public_case_v23() -> PublicTemporalCaseV2:
    row = json.loads(SYNTHETIC_CASE_V23.read_text(encoding="utf-8"))["cases"][0]
    return PublicTemporalCaseV2(
        case_id=row["case_id"], model_id=row["model_id"],
        question=row["question"], start_page=row["start_page"],
        cutoff_date=row["cutoff_date"], target_date=row["target_date"],
    )


def _submission(state: Any, *, invalid: bool = False) -> StructuredSubmissionV2:
    bridge = next(
        page for page in state.collected_evidence
        if "Scientist Z became Director of Lab" in str(page.get("content"))
    )
    tail = next(
        page for page in state.collected_evidence
        if "Scientist Z was born in Harbor City" in str(page.get("content"))
    )
    bridge_id = "evidence_not_in_trajectory" if invalid else str(bridge["evidence_id"])
    return StructuredSubmissionV2(
        answer="Harbor City",
        critical_claims=(SubmittedClaimV2(
            subject="Lab", relation="appointed director", object="Scientist Z",
            event_time="2025-02-02", supporting_evidence_ids=(bridge_id,),
            claim_id="bridge_1",
        ),),
        tail_claim=SubmittedClaimV2(
            subject="Scientist Z", relation="place of birth", object="Harbor City",
            event_time=None, supporting_evidence_ids=(str(tail["evidence_id"]),),
            claim_id="tail",
        ),
    )


def joint_fixture_policy_v23(mode: str = "valid"):
    def control(
        case: PublicTemporalCaseV2, state: Any,
        actions: list[JointCandidateActionV23],
    ) -> tuple[dict[str, float], StructuredSubmissionV2 | None]:
        del case
        complete = (
            any("Scientist Z became Director of Lab" in str(page.get("content"))
                for page in state.collected_evidence)
            and any("Scientist Z was born in Harbor City" in str(page.get("content"))
                    for page in state.collected_evidence)
        )
        scores = {}
        for action in actions:
            score = -8.0
            payload = action.to_dict()
            if action.action_id == SUBMIT_SLOT_ID_V23:
                score = 0.0 if complete else -9.0
            elif action.kind == "FOLLOW_LINK":
                target = str(payload["params"].get("page_title"))
                if target in {"Alternative Route", "Scientist Z"}:
                    score = -0.1
                elif target == "Further Context":
                    score = -0.5
            elif action.kind == "SWITCH_SNAPSHOT":
                score = -0.1 if int(payload["params"].get("revision_id", 0)) == 111 else -8.0
            elif action.kind == "LIST_LINKS":
                score = -0.05 if state.current_revision_id == 111 else -0.2
            elif action.kind == "LIST_REVISIONS":
                score = -0.15 if state.current_page == "Alternative Route" else -1.0
            scores[action.action_id] = score
        if not complete:
            submission = None
        elif mode == "invalid":
            submission = _submission(state, invalid=True)
        else:
            submission = _submission(state)
        return scores, submission
    return control


def run_joint_synthetic_v23(
    *, mode: str = "valid", beam_width: int = 1,
    max_actions_per_state: int = 1, max_expansions: int = 16,
) -> tuple[Any, CallableJointControllerV23]:
    backend = SyntheticJointBackendV23()
    controller = CallableJointControllerV23(joint_fixture_policy_v23(mode))
    result = run_live_temporal_search_v23(
        public_case=public_case_v23(), backend=backend,
        environment=TemporalWikipediaEnvironmentV2(backend), controller=controller,
        config=LiveSearchConfigV23(
            beam_width=beam_width, max_expansions=max_expansions,
            max_actions_per_state=max_actions_per_state,
            dense_action_limit=30, seed=23,
        ),
    )
    return result, controller


def synthetic_artifact_v23() -> dict[str, Any]:
    search, controller = run_joint_synthetic_v23()
    private_case = load_cases_v2(SYNTHETIC_CASE_V23)[0]
    if search.final_state.submitted is None:
        raise AssertionError("v2.3 positive fixture did not submit")
    evaluation = validate_structured_submission_v2(
        case=private_case,
        submission=structured_submission_from_dict(search.final_state.submitted),
        trajectory_evidence=list(search.final_state.collected_evidence),
        trajectory_actions_valid=all(
            row["result"] == "ok"
            for row in search.final_state.action_trace
            if row["action"]["kind"] in {"FOLLOW_LINK", "SWITCH_SNAPSHOT"}
        ),
    )
    diagnostics = reference_route_diagnostics_v2(
        route=private_case.reference_routes[0],
        funnel_steps=[step["action_funnel"] for step in search.audit_steps
                      if "action_funnel" in step],
        action_trace=list(search.final_state.action_trace), case=private_case,
        evaluation=evaluation, search_stop_reason=search.stop_reason,
    )
    return {
        "schema_version": "joint-live-synthetic-artifact-v2.3",
        "search": search.to_dict(),
        "posthoc_private_evaluation": evaluation,
        "posthoc_reference_diagnostics": diagnostics,
        "checks": {
            "one_controller_call_per_visited_state": (
                controller.calls == len(search.audit_steps)
            ),
            "standalone_proposer_used": False,
            "end_to_end_success": evaluation["end_to_end_success"],
            "reference_route_recalled": diagnostics["reference_route_recalled"],
            "alternative_valid_route_found": diagnostics["alternative_valid_route_found"],
        },
        "formal_conclusion_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert_new_output_path(args.output)
    Path(args.output).write_text(
        json.dumps(synthetic_artifact_v23(), ensure_ascii=False, indent=2,
                   sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[done] joint synthetic v2.3: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
