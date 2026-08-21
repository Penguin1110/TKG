"""Terminal-only patch over the frozen joint live runner v2.3.

No prompt, candidate, compaction, score, tie-break, or expansion policy changes.
The adapter only converts the otherwise empty-proposal condition into an
auditable exhausted/no-answer terminal state.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from tkg.experiment.joint_controller_v23 import (
    JointCandidateActionV23, JointControllerOutputV23,
    JointControllerContractErrorV23, JointRankAndSubmitControllerV23,
    SUBMIT_SLOT_ID_V23,
    validate_submission_public_v23,
)
from tkg.experiment.temporal_environment_v2 import TemporalWikipediaEnvironmentV2
from tkg.experiment.temporal_eval_schema_v2 import PublicTemporalCaseV2
from tkg.experiment.temporal_live_runner_v22 import LiveBeamStateV22
from tkg.experiment.temporal_live_runner_v23 import (
    LiveSearchConfigV23, LiveSearchResultV23,
    _normalized_joint_scores, run_live_temporal_search_v23,
)


LIVE_STATE_SCHEMA_V231 = "open-world-temporal-live-state-v2.3.1"
LIVE_TRAJECTORY_SCHEMA_V231 = "open-world-temporal-live-trajectory-v2.3.1"
TERMINAL_SENTINEL_V231 = "v2.3.1:no_executable_child"


class _NoExecutableChildV231(JointControllerContractErrorV23):
    pass


class _TerminalOnlyControllerV231:
    def __init__(self, delegate: JointRankAndSubmitControllerV23):
        self.delegate = delegate
        self.controller_name = delegate.controller_name
        self.terminal_events: dict[str, dict[str, Any]] = {}

    def control(
        self, public_case: PublicTemporalCaseV2, state: Any,
        actions: list[JointCandidateActionV23], *, seed: int,
        budget: dict[str, int],
    ) -> JointControllerOutputV23:
        output = self.delegate.control(
            public_case, state, actions, seed=seed, budget=budget,
        )
        only_slot = (
            len(actions) == 1 and actions[0].action_id == SUBMIT_SLOT_ID_V23
        )
        submit_executable = False
        if output.submission is not None:
            submit_executable = validate_submission_public_v23(
                output.submission, list(state.collected_evidence),
            ).valid
        if only_slot and not submit_executable:
            self.terminal_events[state.state_id] = {
                "state": state,
                "actions": actions,
                "output": output,
            }
            raise _NoExecutableChildV231(TERMINAL_SENTINEL_V231)
        return output


def _terminal_state_v231(state: LiveBeamStateV22) -> LiveBeamStateV22:
    return replace(
        state, finished=True, submitted=None,
        stop_reason="exhausted_no_legal_progress", error="",
        schema_version=LIVE_STATE_SCHEMA_V231,
    )


def _terminal_audit_v231(
    old: dict[str, Any], event: dict[str, Any],
) -> dict[str, Any]:
    state = event["state"]
    actions = event["actions"]
    output = event["output"]
    scored = _normalized_joint_scores(actions, output)
    validation = (
        output.submission_validation
        if output.submission is None
        else validate_submission_public_v23(
            output.submission, list(state.collected_evidence),
        )
    )
    candidates = [{
        **row,
        "expanded": False,
        "retained": False,
        "pruning_reason": (
            "JOINT_SUBMISSION_ABSTAINED"
            if validation.status == "abstained"
            else "JOINT_SUBMISSION_PAYLOAD_INVALID"
        ),
        "submission_validation": validation.to_dict(),
    } for row in scored]
    terminal = _terminal_state_v231(state)
    return {
        "schema_version": LIVE_TRAJECTORY_SCHEMA_V231,
        "controller_protocol": "joint_rank_submit_v2.3",
        "patch_protocol": "terminal_only_v2.3.1",
        "model_calls_per_state_for_rank_and_submit": 1,
        "standalone_submission_proposer_used": False,
        "iteration": old.get("iteration"),
        "parent_state": state.to_dict(),
        "visible_evidence_ids": [
            page["evidence_id"] for page in state.collected_evidence
        ],
        "action_funnel": {
            "schema_version": "temporal-solver-action-funnel-v2.3",
            "policy": (
                "retrieval_order_transitions_reserving_pagination_and_"
                "submit_slot_v2.3"
            ),
            "dense_limit": 30,
            "submit_slot_id": SUBMIT_SLOT_ID_V23,
            "submit_slot_always_present": True,
            "solver_retrieved_actions": [action.to_dict() for action in actions],
            "compacted_ranker_actions": [action.to_dict() for action in actions],
            "ranker_scores": output.scores,
            "ranker_contract_valid": True,
            "expanded_actions": [],
        },
        "joint_controller": {
            "name": "api_joint_rank_submit_external_controller_v2.3",
            "score_kind": output.score_kind,
            "reasoning_summary": output.reasoning_summary,
            "extracted_entities": list(output.extracted_entities),
            "evidence_notes": list(output.evidence_notes),
            "abstain_reason": output.abstain_reason,
            "attempts": list(output.attempts),
        },
        "submission_contract": validation.to_dict(),
        "submission_payload": (
            output.submission.to_dict() if output.submission else None
        ),
        "submission_payload_sha256": None,
        "candidate_actions": candidates,
        "selected_actions": [],
        "retained_actions": [],
        "terminal_state": terminal.to_dict(),
        "terminal_status": "exhausted_no_legal_progress",
        "complete_trajectory_available": True,
        "pruning_reason": "NO_EXECUTABLE_CHILD_TERMINALIZED",
        "error": "",
        "environment_manifest_reference": old.get(
            "environment_manifest_reference"
        ),
    }


def run_live_temporal_search_v231(
    *, public_case: PublicTemporalCaseV2, backend: Any,
    environment: TemporalWikipediaEnvironmentV2,
    controller: JointRankAndSubmitControllerV23,
    config: LiveSearchConfigV23,
) -> LiveSearchResultV23:
    adapter = _TerminalOnlyControllerV231(controller)
    result = run_live_temporal_search_v23(
        public_case=public_case, backend=backend, environment=environment,
        controller=adapter, config=config,
    )
    if not adapter.terminal_events:
        return replace(result, schema_version=LIVE_TRAJECTORY_SCHEMA_V231)

    terminal_ids = set(adapter.terminal_events)
    transformed_audits = []
    for audit in result.audit_steps:
        state_id = audit.get("parent_state", {}).get("state_id")
        if state_id in terminal_ids:
            transformed_audits.append(_terminal_audit_v231(
                audit, adapter.terminal_events[state_id],
            ))
        else:
            transformed_audits.append(dict(audit))
    transformed_states = tuple(
        _terminal_state_v231(state)
        if state.error == TERMINAL_SENTINEL_V231 else replace(
            state, schema_version=LIVE_STATE_SCHEMA_V231,
        )
        for state in result.retained_states
    )
    final = (
        _terminal_state_v231(result.final_state)
        if result.final_state.error == TERMINAL_SENTINEL_V231 else replace(
            result.final_state, schema_version=LIVE_STATE_SCHEMA_V231,
        )
    )
    return replace(
        result, final_state=final, retained_states=transformed_states,
        audit_steps=tuple(transformed_audits),
        stop_reason="exhausted_no_legal_progress",
        controller_calls=(
            result.controller_calls + sum(
                len(event["output"].attempts)
                for event in adapter.terminal_events.values()
            )
        ),
        schema_version=LIVE_TRAJECTORY_SCHEMA_V231,
    )
