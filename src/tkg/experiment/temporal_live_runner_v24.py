"""Live temporal beam runner with compact evidence submission v2.4."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from tkg.experiment.compact_joint_controller_v24 import (
    CompactJointOutputV24, CompactSubmitSlotActionV24,
)
from tkg.experiment.compact_submission_v24 import (
    CompactSubmissionValidationV24, validate_compact_submission_public_v24,
)
from tkg.experiment.joint_controller_v23 import (
    JointControllerContractErrorV23, SUBMIT_SLOT_ID_V23,
)
from tkg.experiment.temporal_environment_v2 import (
    EnvironmentActionV2, TemporalWikipediaEnvironmentV2,
)
from tkg.experiment.temporal_eval_schema_v2 import PublicTemporalCaseV2
from tkg.experiment.temporal_live_ranker_v22 import LiveRankOutputV22
from tkg.experiment.temporal_live_runner_v22 import (
    LiveBeamStateV22, _EnvironmentManifestCacheV22, _canonical_hash,
    _execute_action, _snapshot_for_state, _stable_order, initial_live_state_v22,
)
from tkg.experiment.temporal_live_runner_v23 import (
    LiveSearchConfigV23, LiveSearchResultV23, _graph_and_control_actions_v23,
    _normalized_joint_scores,
)
from tkg.wikipedia.backend import WikipediaError


LIVE_STATE_SCHEMA_V24 = "open-world-temporal-live-state-v2.4"
LIVE_TRAJECTORY_SCHEMA_V24 = "open-world-temporal-live-trajectory-v2.4"


@dataclass(frozen=True)
class LiveSearchResultV24(LiveSearchResultV23):
    schema_version: str = LIVE_TRAJECTORY_SCHEMA_V24

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update({
            "schema_version": LIVE_TRAJECTORY_SCHEMA_V24,
            "controller_protocol": "joint_rank_compact_submit_v2.4",
            "submission_schema": "compact-temporal-evidence-submission-v2.4",
            "model_supplied_claim_shapes": False,
            "private_posthoc_evaluator_supplies_claim_shapes": True,
            "formal_conclusion_allowed": False,
        })
        return payload


def _state_v24(state: LiveBeamStateV22) -> LiveBeamStateV22:
    return replace(state, schema_version=LIVE_STATE_SCHEMA_V24)


def _graph_and_control_actions_v24(
    state: LiveBeamStateV22, case: PublicTemporalCaseV2,
    config: LiveSearchConfigV23,
) -> tuple[list[Any], dict[str, Any]]:
    actions, funnel = _graph_and_control_actions_v23(state, case, config)
    compact_slot = CompactSubmitSlotActionV24()
    actions = [
        compact_slot if action.action_id == SUBMIT_SLOT_ID_V23 else action
        for action in actions
    ]
    for key in ("solver_retrieved_actions", "compacted_ranker_actions"):
        funnel[key] = [
            compact_slot.to_dict() if row["action_id"] == SUBMIT_SLOT_ID_V23 else row
            for row in funnel[key]
        ]
    funnel["submission_schema"] = "compact-temporal-evidence-submission-v2.4"
    return actions, funnel


def _instantiate_submit_v24(
    state: LiveBeamStateV22, output: CompactJointOutputV24,
) -> tuple[EnvironmentActionV2 | None, CompactSubmissionValidationV24, str | None]:
    if output.submission is None:
        return None, output.submission_schema_validation, None
    validation = validate_compact_submission_public_v24(
        output.submission, list(state.collected_evidence),
    )
    if not validation.valid:
        return None, validation, None
    payload = output.submission.to_dict()
    payload_hash = _canonical_hash(payload)
    return EnvironmentActionV2(
        "SUBMIT_ANSWER", payload, f"Submit compact payload {payload_hash[:16]}",
    ), validation, payload_hash


def _execute_v24(
    *, state: LiveBeamStateV22, candidate: dict[str, Any],
    output: CompactJointOutputV24, submit: EnvironmentActionV2 | None,
    validation: CompactSubmissionValidationV24, payload_hash: str | None,
    environment: TemporalWikipediaEnvironmentV2, backend: Any,
    case: PublicTemporalCaseV2,
) -> LiveBeamStateV22:
    is_submit = candidate["action_id"] == SUBMIT_SLOT_ID_V23
    action = submit if is_submit else EnvironmentActionV2(
        kind=str(candidate["kind"]), params=dict(candidate["params"]),
        label=str(candidate["label"]),
        environment_order=candidate.get("environment_order"),
    )
    if action is None:
        raise ValueError("non-executable compact submit reached executor")
    rank = LiveRankOutputV22(
        scores={action.action_id: float(candidate["raw_controller_utility"])},
        score_kind=output.score_kind,
        reasoning_summary=output.reasoning_summary,
        extracted_entities=output.extracted_entities,
        evidence_notes=output.evidence_notes,
    )
    if is_submit:
        trace = {
            "index": len(state.action_trace) + 1,
            "action": action.to_dict(),
            "action_score": float(candidate["action_score"]),
            "from_node": list(state.node_key),
            "to_node": list(state.node_key),
            "result": "ok",
            "error": "",
            "hyperlink_valid": None,
            "revision_valid": None,
            "environment_query_valid": None,
            "compact_submission_public_gate_passed": True,
            "submit_slot_id": SUBMIT_SLOT_ID_V23,
            "instantiated_action_id": action.action_id,
            "compact_payload": action.params,
            "payload_canonical_sha256": payload_hash,
            "public_submission_validation": validation.to_dict(),
            "model_supplied_claim_shapes": False,
            "cumulative_score_after_action": (
                state.cumulative_score + float(candidate["action_score"])
            ),
        }
        return _state_v24(replace(
            state,
            reasoning_summary=output.reasoning_summary[:2000],
            action_trace=(*state.action_trace, trace),
            cumulative_score=(
                state.cumulative_score + float(candidate["action_score"])
            ),
            finished=True, submitted=dict(action.params),
            stop_reason="submit_answer", error="",
        ))
    child, _ = _execute_action(
        state=state, action=action, action_score=float(candidate["action_score"]),
        ranker_output=rank, environment=environment, backend=backend, case=case,
    )
    child = _state_v24(child)
    return child


def run_live_temporal_search_v24(
    *, public_case: PublicTemporalCaseV2, backend: Any,
    environment: TemporalWikipediaEnvironmentV2,
    controller: Any, config: LiveSearchConfigV23,
) -> LiveSearchResultV24:
    """Run from the public start state; no private case enters the search."""
    frontier = [_state_v24(initial_live_state_v22(public_case, backend))]
    manifests = _EnvironmentManifestCacheV22(environment, public_case, config)
    audits: list[dict[str, Any]] = []
    expansions = repeated = controller_calls = iteration = 0
    stop_reason = "exhausted_search"
    while frontier and expansions < config.max_expansions:
        iteration += 1
        proposals: list[tuple[LiveBeamStateV22, int, int]] = []
        iteration_audits: list[dict[str, Any]] = []
        for state in sorted(frontier, key=lambda item: _stable_order(item, config.seed)):
            if state.finished:
                proposals.append((state, -1, -1))
                continue
            manifest = None
            try:
                manifest = manifests.get(_snapshot_for_state(state, backend))
                actions, funnel = _graph_and_control_actions_v24(
                    state, public_case, config,
                )
                output = controller.control(
                    public_case, state, actions, seed=config.seed,
                    budget={"expansions_used": expansions,
                            "max_expansions": config.max_expansions,
                            "beam_width": config.beam_width,
                            "max_actions_per_state": config.max_actions_per_state},
                )
                controller_calls += len(output.attempts)
                scored = sorted(
                    _normalized_joint_scores(actions, output),
                    key=lambda row: (-float(row["action_score"]), _canonical_hash(
                        [config.seed, state.state_id, row["action_id"]]
                    )),
                )
                submit, validation, payload_hash = _instantiate_submit_v24(state, output)
                funnel.update({
                    "parent_page": state.current_page,
                    "parent_revision_id": state.current_revision_id,
                    "environment_legal_action_count": manifest.action_count,
                    "environment_legal_actions_sha256": manifest.actions_sha256,
                    "environment_legal_actions_artifact_reference": manifest.manifest_id,
                    "ranker_scores": output.scores, "ranker_contract_valid": True,
                })
                audit: dict[str, Any] = {
                    "schema_version": LIVE_TRAJECTORY_SCHEMA_V24,
                    "controller_protocol": "joint_rank_compact_submit_v2.4",
                    "iteration": iteration, "parent_state": state.to_dict(),
                    "visible_evidence_ids": [
                        page["evidence_id"] for page in state.collected_evidence
                    ],
                    "action_funnel": funnel,
                    "joint_controller": {
                        "name": controller.controller_name,
                        "score_kind": output.score_kind,
                        "reasoning_summary": output.reasoning_summary,
                        "extracted_entities": list(output.extracted_entities),
                        "evidence_notes": list(output.evidence_notes),
                        "abstain_reason": output.abstain_reason,
                        "attempts": list(output.attempts),
                    },
                    "submission_contract": validation.to_dict(),
                    "submission_payload": output.submission.to_dict()
                    if output.submission else None,
                    "submission_payload_sha256": payload_hash,
                    "model_supplied_claim_shapes": False,
                    "candidate_actions": [], "selected_actions": [],
                    "retained_actions": [],
                }
                expanded_for_parent = 0
                audit_index = len(iteration_audits)
                for candidate in scored:
                    record = {**candidate, "expanded": False, "retained": False,
                              "pruning_reason": ""}
                    is_submit = record["action_id"] == SUBMIT_SLOT_ID_V23
                    if is_submit and submit is None:
                        record["pruning_reason"] = (
                            "JOINT_SUBMISSION_ABSTAINED" if validation.status == "abstained"
                            else "JOINT_SUBMISSION_PAYLOAD_INVALID"
                        )
                    elif expanded_for_parent >= config.max_actions_per_state:
                        record["pruning_reason"] = "local_expansion_cap"
                    elif expansions >= config.max_expansions:
                        record["pruning_reason"] = "max_expansions"
                    else:
                        child = _execute_v24(
                            state=state, candidate=record, output=output, submit=submit,
                            validation=validation, payload_hash=payload_hash,
                            environment=environment, backend=backend, case=public_case,
                        )
                        expansions += 1
                        expanded_for_parent += 1
                        record["expanded"] = True
                        record["resulting_state"] = child.to_dict()
                        audit["selected_actions"].append(record["action_id"])
                        funnel["expanded_actions"].append(record["action_id"])
                        proposals.append((child, audit_index,
                                          len(audit["candidate_actions"])))
                    audit["candidate_actions"].append(record)
                if expanded_for_parent == 0:
                    terminal = _state_v24(replace(
                        state, finished=True, submitted=None,
                        stop_reason="exhausted_no_legal_progress", error="",
                    ))
                    proposals.append((terminal, -1, -1))
                    audit["terminal_status"] = "exhausted_no_legal_progress"
                    audit["complete_trajectory_available"] = True
                iteration_audits.append(audit)
            except JointControllerContractErrorV23 as exc:
                terminal = _state_v24(replace(
                    state, finished=True, stop_reason="joint_ranking_contract_failure",
                    error=str(exc),
                ))
                proposals.append((terminal, -1, -1))
                iteration_audits.append({
                    "schema_version": LIVE_TRAJECTORY_SCHEMA_V24,
                    "iteration": iteration, "parent_state": state.to_dict(),
                    "candidate_actions": [], "selected_actions": [],
                    "retained_actions": [], "pruning_reason": "RANKER_CONTRACT_FAILURE",
                    "error": str(exc),
                })
            except (KeyError, TypeError, ValueError, WikipediaError) as exc:
                terminal = _state_v24(replace(
                    state, finished=True, stop_reason="runner_or_environment_error",
                    error=str(exc),
                ))
                proposals.append((terminal, -1, -1))
                iteration_audits.append({
                    "schema_version": LIVE_TRAJECTORY_SCHEMA_V24,
                    "iteration": iteration, "parent_state": state.to_dict(),
                    "candidate_actions": [], "selected_actions": [],
                    "retained_actions": [],
                    "pruning_reason": "runner_or_environment_error", "error": str(exc),
                })

        best: dict[tuple[Any, ...], tuple[LiveBeamStateV22, int, int]] = {}
        for proposal in proposals:
            child, audit_index, candidate_index = proposal
            key = child.dedup_key()
            previous = best.get(key)
            if previous is None or _stable_order(child, config.seed) < _stable_order(
                previous[0], config.seed,
            ):
                best[key] = proposal
            else:
                repeated += 1
                if audit_index >= 0:
                    iteration_audits[audit_index]["candidate_actions"][candidate_index][
                        "pruning_reason"
                    ] = "duplicate_state_lower_score"
        ordered = sorted(best.values(), key=lambda row: _stable_order(row[0], config.seed))
        retained = ordered[:config.beam_width]
        retained_ids = {row[0].state_id for row in retained}
        for child, audit_index, candidate_index in ordered:
            if audit_index < 0:
                continue
            candidate = iteration_audits[audit_index]["candidate_actions"][candidate_index]
            if child.state_id in retained_ids:
                candidate["retained"] = True
                candidate["pruning_reason"] = ""
                iteration_audits[audit_index]["retained_actions"].append(
                    candidate["action_id"]
                )
            elif not candidate["pruning_reason"]:
                candidate["pruning_reason"] = "global_beam_prune"
        audits.extend(iteration_audits)
        frontier = [row[0] for row in retained]
        if frontier and all(state.finished for state in frontier):
            stop_reason = "all_retained_error" if all(
                state.error for state in frontier
            ) else "all_retained_finished"
            break
    if expansions >= config.max_expansions and frontier and not all(
        state.finished for state in frontier
    ):
        stop_reason = "max_expansions"
    elif not frontier:
        stop_reason = "exhausted_search"
    submitted = [state for state in frontier if state.finished and state.submitted]
    pool = submitted or frontier
    if not pool:
        raise RuntimeError("v2.4 live search produced no terminal state")
    final = sorted(pool, key=lambda state: _stable_order(state, config.seed))[0]
    return LiveSearchResultV24(
        public_case=public_case, config=config,
        controller_name=controller.controller_name, final_state=final,
        retained_states=tuple(frontier), audit_steps=tuple(audits),
        environment_manifests=manifests.values(), expansions=expansions,
        repeated_state_count=repeated, stop_reason=stop_reason,
        controller_calls=controller_calls,
    )
