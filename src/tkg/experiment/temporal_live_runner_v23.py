"""Append-only live temporal beam runner with one joint v2.3 controller call."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tkg.experiment.joint_controller_v23 import (
    ApiJointRankAndSubmitControllerV23, JointCandidateActionV23,
    JointControllerContractErrorV23, JointControllerOutputV23,
    JointRankAndSubmitControllerV23, SUBMIT_SLOT_ID_V23, SubmitSlotActionV23,
    SubmissionValidationV23, validate_submission_public_v23,
)
from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_environment_v2 import (
    EnvironmentActionV2, TemporalWikipediaEnvironmentV2,
    initial_environment_queries_v2,
)
from tkg.experiment.temporal_eval_schema_v2 import PublicTemporalCaseV2
from tkg.experiment.temporal_live_ranker_v22 import LiveRankOutputV22
from tkg.experiment.temporal_live_runner_v22 import (
    EnvironmentNodeManifestV22, LiveBeamStateV22, LiveSearchConfigV22,
    _EnvironmentManifestCacheV22, _canonical_hash, _execute_action,
    _prospective_node_key, _snapshot_for_state, _stable_order,
    initial_live_state_v22, load_public_manifest_v22,
)
from tkg.wikipedia.backend import WikipediaError, WikipediaPageBackend


LIVE_STATE_SCHEMA_V23 = "open-world-temporal-live-state-v2.3"
LIVE_TRAJECTORY_SCHEMA_V23 = "open-world-temporal-live-trajectory-v2.3"
LIVE_RUN_SCHEMA_V23 = "open-world-temporal-live-run-v2.3"
COMPACTION_POLICY_V23 = (
    "retrieval_order_transitions_reserving_pagination_and_submit_slot_v2.3"
)

LiveSearchConfigV23 = LiveSearchConfigV22
LiveBeamStateV23 = LiveBeamStateV22


@dataclass(frozen=True)
class LiveSearchResultV23:
    public_case: PublicTemporalCaseV2
    config: LiveSearchConfigV23
    controller_name: str
    final_state: LiveBeamStateV23
    retained_states: tuple[LiveBeamStateV23, ...]
    audit_steps: tuple[dict[str, Any], ...]
    environment_manifests: tuple[EnvironmentNodeManifestV22, ...]
    expansions: int
    repeated_state_count: int
    stop_reason: str
    controller_calls: int
    schema_version: str = LIVE_TRAJECTORY_SCHEMA_V23

    def to_dict(self) -> dict[str, Any]:
        score_recomputed = sum(
            float(row["action_score"]) for row in self.final_state.action_trace
        )
        return {
            "schema_version": self.schema_version,
            "controller_protocol": "joint_rank_submit_v2.3",
            "model_calls_per_state_for_rank_and_submit": 1,
            "max_model_calls_per_state_with_retry": 2,
            "standalone_submission_proposer_used": False,
            "public_case": self.public_case.to_dict(),
            "config": asdict(self.config),
            "controller_name": self.controller_name,
            "final_state": self.final_state.to_dict(),
            "retained_states": [state.to_dict() for state in self.retained_states],
            "audit_steps": list(self.audit_steps),
            "environment_manifests": [
                manifest.to_dict() for manifest in self.environment_manifests
            ],
            "expansions": self.expansions,
            "repeated_state_count": self.repeated_state_count,
            "stop_reason": self.stop_reason,
            "controller_calls": self.controller_calls,
            "score_recomputed": score_recomputed,
            "score_recomputable": math.isclose(
                self.final_state.cumulative_score, score_recomputed,
                rel_tol=0.0, abs_tol=1e-12,
            ),
            "formal_conclusion_allowed": False,
            "graph_integrated_decoding": False,
        }


def _state_v23(state: LiveBeamStateV22) -> LiveBeamStateV23:
    return replace(state, schema_version=LIVE_STATE_SCHEMA_V23)


def _graph_and_control_actions_v23(
    state: LiveBeamStateV23, case: PublicTemporalCaseV2,
    config: LiveSearchConfigV23,
) -> tuple[list[JointCandidateActionV23], dict[str, Any]]:
    transitions: list[EnvironmentActionV2] = [
        action for action in (
            *state.retrieved_link_actions, *state.retrieved_revision_actions,
        )
        if not (
            action.kind in {"FOLLOW_LINK", "SWITCH_SNAPSHOT"}
            and _prospective_node_key(state, action) in state.visited_nodes
        )
    ]
    pagination: list[EnvironmentActionV2] = []
    if state.environment_queries_used < config.max_environment_queries_per_node:
        first_links, first_revisions = initial_environment_queries_v2(
            link_page_size=config.link_page_size,
            revision_page_size=config.revision_page_size,
            from_date=case.cutoff_date,
            to_date=case.target_date,
        )
        if not state.link_query_started:
            pagination.append(first_links)
        elif not state.links_exhausted and state.link_next_cursor is not None:
            pagination.append(EnvironmentActionV2(
                "LIST_LINKS",
                {"cursor": state.link_next_cursor, "page_size": config.link_page_size},
                f"List next hyperlink page from cursor {state.link_next_cursor}",
            ))
        if not state.revision_query_started:
            pagination.append(first_revisions)
        elif not state.revisions_exhausted and state.revision_next_cursor is not None:
            pagination.append(EnvironmentActionV2(
                "LIST_REVISIONS",
                {
                    "cursor": state.revision_next_cursor,
                    "page_size": config.revision_page_size,
                    "time_window": [case.cutoff_date, case.target_date],
                },
                f"List next revision page from cursor {state.revision_next_cursor}",
            ))
    submit_slot = SubmitSlotActionV23()
    reserved: list[JointCandidateActionV23] = [*pagination, submit_slot]
    if len(reserved) > config.dense_action_limit:
        raise ValueError("reserved v2.3 controls exceed dense action limit")
    kept_transitions = transitions[:config.dense_action_limit - len(reserved)]
    compacted: list[JointCandidateActionV23] = [*kept_transitions, *reserved]
    kept_ids = {action.action_id for action in compacted}
    all_before: list[JointCandidateActionV23] = [
        *transitions, *pagination, submit_slot,
    ]
    pruning = [{
        "action_id": action.action_id,
        "reason": "retained" if action.action_id in kept_ids else "dense_limit_document_order",
    } for action in all_before]
    return compacted, {
        "schema_version": "temporal-solver-action-funnel-v2.3",
        "policy": COMPACTION_POLICY_V23,
        "dense_limit": config.dense_action_limit,
        "submit_slot_id": SUBMIT_SLOT_ID_V23,
        "submit_slot_always_present": True,
        "solver_retrieved_actions": [action.to_dict() for action in all_before],
        "compacted_ranker_actions": [action.to_dict() for action in compacted],
        "compaction_records": pruning,
        "expanded_actions": [],
    }


def _normalized_joint_scores(
    actions: list[JointCandidateActionV23], output: JointControllerOutputV23,
) -> list[dict[str, Any]]:
    expected = {action.action_id for action in actions}
    if set(output.scores) != expected:
        raise JointControllerContractErrorV23("joint score coverage changed in runner")
    raw = {key: float(value) for key, value in output.scores.items()}
    if output.score_kind == "length_normalized_conditional_logprob":
        normalized = raw
    else:
        maximum = max(raw.values())
        denominator = sum(math.exp(value - maximum) for value in raw.values())
        normalized = {
            key: value - maximum - math.log(denominator) for key, value in raw.items()
        }
    return [{
        **action.to_dict(),
        "raw_controller_utility": raw[action.action_id],
        "action_score": normalized[action.action_id],
        "score_kind": output.score_kind,
    } for action in actions]


def _instantiate_submit_v23(
    state: LiveBeamStateV23, output: JointControllerOutputV23,
) -> tuple[EnvironmentActionV2 | None, SubmissionValidationV23, str | None]:
    if output.submission is None:
        return None, output.submission_validation, None
    validation = validate_submission_public_v23(
        output.submission, list(state.collected_evidence),
    )
    if not validation.valid:
        return None, validation, None
    payload = output.submission.to_dict()
    payload_hash = _canonical_hash(payload)
    return EnvironmentActionV2(
        kind="SUBMIT_ANSWER",
        params=payload,
        label=f"Submit structured payload {payload_hash[:16]}",
    ), validation, payload_hash


def _execute_v23(
    *, state: LiveBeamStateV23, candidate: dict[str, Any],
    joint_output: JointControllerOutputV23,
    instantiated_submit: EnvironmentActionV2 | None,
    submit_validation: SubmissionValidationV23,
    submit_payload_hash: str | None,
    environment: TemporalWikipediaEnvironmentV2, backend: Any,
    case: PublicTemporalCaseV2,
) -> LiveBeamStateV23:
    slot = candidate["action_id"] == SUBMIT_SLOT_ID_V23
    action = instantiated_submit if slot else EnvironmentActionV2(
        kind=str(candidate["kind"]), params=dict(candidate["params"]),
        label=str(candidate["label"]),
        environment_order=candidate.get("environment_order"),
    )
    if action is None:
        raise ValueError("non-executable submit slot reached executor")
    rank_output = LiveRankOutputV22(
        scores={action.action_id: float(candidate["raw_controller_utility"])},
        score_kind=joint_output.score_kind,
        reasoning_summary=joint_output.reasoning_summary,
        extracted_entities=joint_output.extracted_entities,
        evidence_notes=joint_output.evidence_notes,
    )
    child, _ = _execute_action(
        state=state, action=action,
        action_score=float(candidate["action_score"]),
        ranker_output=rank_output,
        environment=environment, backend=backend, case=case,
    )
    child = _state_v23(child)
    if slot:
        trace = dict(child.action_trace[-1])
        trace.update({
            "submit_slot_id": SUBMIT_SLOT_ID_V23,
            "instantiated_action_id": action.action_id,
            "structured_payload": action.params,
            "payload_canonical_sha256": submit_payload_hash,
            "public_submission_validation": submit_validation.to_dict(),
            "cumulative_score_after_action": child.cumulative_score,
        })
        child = replace(child, action_trace=(*child.action_trace[:-1], trace))
    return child


def run_live_temporal_search_v23(
    *, public_case: PublicTemporalCaseV2, backend: Any,
    environment: TemporalWikipediaEnvironmentV2,
    controller: JointRankAndSubmitControllerV23,
    config: LiveSearchConfigV23,
) -> LiveSearchResultV23:
    """Run joint public-only search; private evaluation has no API entry point."""
    frontier = [_state_v23(initial_live_state_v22(public_case, backend))]
    manifests = _EnvironmentManifestCacheV22(environment, public_case, config)
    audit_steps: list[dict[str, Any]] = []
    expansions = 0
    repeated = 0
    controller_calls = 0
    iteration = 0
    stop_reason = "exhausted_search"
    while frontier and expansions < config.max_expansions:
        iteration += 1
        proposals: list[tuple[LiveBeamStateV23, int, int]] = []
        iteration_audits: list[dict[str, Any]] = []
        for state in sorted(frontier, key=lambda item: _stable_order(item, config.seed)):
            if state.finished:
                proposals.append((state, -1, -1))
                continue
            manifest = None
            try:
                snapshot = _snapshot_for_state(state, backend)
                manifest = manifests.get(snapshot)
                actions, funnel = _graph_and_control_actions_v23(
                    state, public_case, config,
                )
                output = controller.control(
                    public_case, state, actions, seed=config.seed,
                    budget={
                        "expansions_used": expansions,
                        "max_expansions": config.max_expansions,
                        "beam_width": config.beam_width,
                        "max_actions_per_state": config.max_actions_per_state,
                    },
                )
                controller_calls += len(output.attempts)
                scored = sorted(
                    _normalized_joint_scores(actions, output),
                    key=lambda row: (
                        -float(row["action_score"]),
                        _canonical_hash([config.seed, state.state_id, row["action_id"]]),
                    ),
                )
                instantiated, submission_validation, payload_hash = (
                    _instantiate_submit_v23(state, output)
                )
                funnel.update({
                    "parent_page": state.current_page,
                    "parent_revision_id": state.current_revision_id,
                    "environment_legal_actions": [],
                    "environment_legal_action_count": manifest.action_count,
                    "environment_legal_actions_sha256": manifest.actions_sha256,
                    "environment_legal_actions_artifact_reference": manifest.manifest_id,
                    "ranker_scores": output.scores,
                    "ranker_contract_valid": True,
                })
                audit: dict[str, Any] = {
                    "schema_version": LIVE_TRAJECTORY_SCHEMA_V23,
                    "controller_protocol": "joint_rank_submit_v2.3",
                    "model_calls_per_state_for_rank_and_submit": 1,
                    "standalone_submission_proposer_used": False,
                    "iteration": iteration,
                    "parent_state": state.to_dict(),
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
                    "submission_contract": submission_validation.to_dict(),
                    "submission_payload": (
                        output.submission.to_dict() if output.submission else None
                    ),
                    "submission_payload_sha256": payload_hash,
                    "candidate_actions": [],
                    "selected_actions": [],
                    "retained_actions": [],
                }
                expanded_for_parent = 0
                audit_index = len(iteration_audits)
                for candidate in scored:
                    record = dict(candidate)
                    record.update({
                        "expanded": False, "retained": False, "pruning_reason": "",
                    })
                    is_slot = record["action_id"] == SUBMIT_SLOT_ID_V23
                    if is_slot and instantiated is None:
                        record["pruning_reason"] = (
                            "JOINT_SUBMISSION_ABSTAINED"
                            if submission_validation.status == "abstained"
                            else "JOINT_SUBMISSION_PAYLOAD_INVALID"
                        )
                        record["submission_validation"] = submission_validation.to_dict()
                    elif expanded_for_parent >= config.max_actions_per_state:
                        record["pruning_reason"] = "local_expansion_cap"
                    elif expansions >= config.max_expansions:
                        record["pruning_reason"] = "max_expansions"
                    else:
                        child = _execute_v23(
                            state=state, candidate=record, joint_output=output,
                            instantiated_submit=instantiated,
                            submit_validation=submission_validation,
                            submit_payload_hash=payload_hash,
                            environment=environment, backend=backend, case=public_case,
                        )
                        expansions += 1
                        expanded_for_parent += 1
                        record["expanded"] = True
                        record["resulting_state"] = child.to_dict()
                        audit["selected_actions"].append(record["action_id"])
                        funnel["expanded_actions"].append(record["action_id"])
                        action_kind = "SUBMIT_SLOT" if is_slot else record["kind"]
                        revisited = (
                            action_kind in {"FOLLOW_LINK", "SWITCH_SNAPSHOT"}
                            and child.node_key in state.visited_nodes
                        )
                        if revisited:
                            record["pruning_reason"] = "repeated_path_node"
                            repeated += 1
                        else:
                            proposals.append((
                                child, audit_index, len(audit["candidate_actions"]),
                            ))
                    audit["candidate_actions"].append(record)
                iteration_audits.append(audit)
            except JointControllerContractErrorV23 as exc:
                terminal = _state_v23(replace(
                    state, finished=True, stop_reason="joint_ranking_contract_failure",
                    error=str(exc),
                ))
                proposals.append((terminal, -1, -1))
                iteration_audits.append({
                    "schema_version": LIVE_TRAJECTORY_SCHEMA_V23,
                    "iteration": iteration, "parent_state": state.to_dict(),
                    "candidate_actions": [], "selected_actions": [],
                    "retained_actions": [],
                    "pruning_reason": "RANKER_CONTRACT_FAILURE",
                    "error": str(exc),
                    "environment_manifest_reference": (
                        manifest.manifest_id if manifest else None
                    ),
                })
            except (KeyError, TypeError, ValueError, WikipediaError) as exc:
                terminal = _state_v23(replace(
                    state, finished=True, stop_reason="runner_or_environment_error",
                    error=str(exc),
                ))
                proposals.append((terminal, -1, -1))
                iteration_audits.append({
                    "schema_version": LIVE_TRAJECTORY_SCHEMA_V23,
                    "iteration": iteration, "parent_state": state.to_dict(),
                    "candidate_actions": [], "selected_actions": [],
                    "retained_actions": [],
                    "pruning_reason": "runner_or_environment_error",
                    "error": str(exc),
                    "environment_manifest_reference": (
                        manifest.manifest_id if manifest else None
                    ),
                })

        best: dict[tuple[Any, ...], tuple[LiveBeamStateV23, int, int]] = {}
        for proposal in proposals:
            child, audit_index, candidate_index = proposal
            key = child.dedup_key()
            previous = best.get(key)
            if previous is None or _stable_order(child, config.seed) < _stable_order(
                previous[0], config.seed,
            ):
                if previous is not None and previous[1] >= 0:
                    old = iteration_audits[previous[1]]["candidate_actions"][previous[2]]
                    old["pruning_reason"] = "duplicate_state_lower_score"
                    repeated += 1
                best[key] = proposal
            else:
                if audit_index >= 0:
                    current = iteration_audits[audit_index]["candidate_actions"][
                        candidate_index
                    ]
                    current["pruning_reason"] = "duplicate_state_lower_score"
                repeated += 1
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
        audit_steps.extend(iteration_audits)
        frontier = [row[0] for row in retained]
        if frontier and all(state.finished for state in frontier):
            stop_reason = (
                "all_retained_error" if all(state.error for state in frontier)
                else "all_retained_finished"
            )
            break
    if expansions >= config.max_expansions and not all(
        state.finished for state in frontier
    ):
        stop_reason = "max_expansions"
    elif not frontier:
        stop_reason = "exhausted_search"
    submitted = [state for state in frontier if state.finished and state.submitted]
    pool = submitted or frontier
    if not pool:
        raise RuntimeError("v2.3 live search produced no retained state")
    final = sorted(pool, key=lambda state: _stable_order(state, config.seed))[0]
    return LiveSearchResultV23(
        public_case=public_case, config=config,
        controller_name=controller.controller_name, final_state=final,
        retained_states=tuple(frontier), audit_steps=tuple(audit_steps),
        environment_manifests=manifests.values(), expansions=expansions,
        repeated_state_count=repeated, stop_reason=stop_reason,
        controller_calls=controller_calls,
    )


class _ExclusiveJsonlWriterV23:
    def __init__(self, path: str | Path):
        assert_new_output_path(str(path))
        self.handle = Path(path).open("x", encoding="utf-8")

    def write(self, **fields: Any) -> None:
        self.handle.write(json.dumps({
            "schema_version": LIVE_RUN_SCHEMA_V23,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **fields,
        }, ensure_ascii=False, sort_keys=True) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Public-only joint live runner v2.3")
    parser.add_argument("--public-cases", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-expansions", type=int, default=40)
    parser.add_argument("--max-actions-per-state", type=int, default=4)
    parser.add_argument("--dense-action-limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--wikipedia-cache", default="live_v23_wikipedia.db")
    parser.add_argument("--controller-cache", default="live_v23_controller.db")
    parser.add_argument("--wikipedia-offline-only", action="store_true")
    parser.add_argument("--request-interval", type=float, default=0.7)
    parser.add_argument("--api-call-budget", type=int, default=2000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is required")
    cases = load_public_manifest_v22(args.public_cases)
    if any(case.model_id != args.model for case in cases):
        parser.error("every public case must be bound to --model")
    config = LiveSearchConfigV23(
        beam_width=args.beam_width, max_expansions=args.max_expansions,
        max_actions_per_state=args.max_actions_per_state,
        dense_action_limit=args.dense_action_limit, seed=args.seed,
    )
    backend = WikipediaPageBackend(
        cache_path=args.wikipedia_cache,
        offline_only=args.wikipedia_offline_only,
        min_request_interval=args.request_interval,
        max_api_calls=args.api_call_budget,
    )
    controller = ApiJointRankAndSubmitControllerV23(
        args.model, cache_path=args.controller_cache,
        max_dense_actions=args.dense_action_limit,
    )
    environment = TemporalWikipediaEnvironmentV2(backend)
    writer = _ExclusiveJsonlWriterV23(args.output)
    try:
        for case in cases:
            result = run_live_temporal_search_v23(
                public_case=case, backend=backend, environment=environment,
                controller=controller, config=config,
            )
            for manifest in result.environment_manifests:
                writer.write(slot="environment_node_manifest", case_id=case.case_id,
                             manifest=manifest.to_dict())
            for step in result.audit_steps:
                writer.write(slot="joint_beam_expansion", case_id=case.case_id,
                             step=step)
            writer.write(slot="joint_beam_summary", case_id=case.case_id,
                         result=result.to_dict(), formal_conclusion_allowed=False)
    finally:
        writer.close()
        controller.close()
        backend.close()
    print(f"[done] live v2.3 output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
