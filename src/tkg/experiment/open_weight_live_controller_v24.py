"""Open-weight log-prob action controller adapter for the v2.4 live beam."""

from __future__ import annotations

import json
import math
from typing import Any

from tkg.experiment.compact_joint_controller_v24 import CompactJointOutputV24
from tkg.experiment.compact_submission_v24 import (
    CompactSubmissionV24, CompactSubmissionValidationV24,
)
from tkg.experiment.joint_controller_v23 import JointCandidateActionV23
from tkg.experiment.open_weight_action_scorer_v24 import (
    OpenWeightConditionalActionScorerV24,
)
from tkg.experiment.open_weight_answer_generator_v24 import CompactPayloadProposalV24
from tkg.experiment.temporal_eval_schema_v2 import PublicTemporalCaseV2


OPEN_WEIGHT_LIVE_CONTROLLER_PROTOCOL_V24 = "open-weight-live-controller-v2.4"
MODE_VERBALIZER_ROTATIONS_V24 = (
    (("A", "CONTINUE"), ("B", "SUBMIT")),
    (("A", "SUBMIT"), ("B", "CONTINUE")),
)


class OpenWeightLiveControllerV24:
    """Scores traversal/submit actions with logits; payload extraction is explicit."""

    controller_name = "open_weight_conditional_logprob_controller_v2.4"

    def __init__(
        self, scorer: OpenWeightConditionalActionScorerV24, *,
        compact_payload_proposer: Any,
        payload_proposer_name: str,
    ):
        self.scorer = scorer
        self.compact_payload_proposer = compact_payload_proposer
        self.payload_proposer_name = payload_proposer_name

    def _propose_payload(
        self, case: PublicTemporalCaseV2, state: Any,
    ) -> tuple[CompactSubmissionV24 | None, dict[str, Any]]:
        proposer = self.compact_payload_proposer
        method = getattr(proposer, "propose", None)
        if callable(method):
            result = method(case, state)
            if not isinstance(result, CompactPayloadProposalV24):
                raise TypeError("payload proposer returned unexpected result type")
            return result.submission, result.audit_dict()
        if not callable(proposer):
            raise TypeError("payload proposer must be callable or expose propose()")
        submission = proposer(state)
        return submission, {
            "protocol": "legacy-callable-payload-proposer-v2.4",
            "status": "candidate" if submission is not None else "abstained",
        }

    @staticmethod
    def prompt_for(
        case: PublicTemporalCaseV2, state: Any,
        actions: list[JointCandidateActionV23], budget: dict[str, int],
    ) -> str:
        evidence = [{
            "evidence_id": page["evidence_id"], "title": page["title"],
            "revision_id": page["revision_id"], "timestamp": page["timestamp"],
            "content": str(page.get("content") or "")[:12_000],
        } for page in state.collected_evidence[-8:]]
        public_actions = [{
            "kind": action.kind, "label": action.label,
            "params": action.to_dict().get("params", {}),
        } for action in actions]
        public_actions.sort(key=lambda row: json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ))
        public = {
            "question": case.question,
            "cutoff_date": case.cutoff_date, "target_date": case.target_date,
            "current_page": state.current_page,
            "current_revision_id": state.current_revision_id,
            "reasoning_summary": state.reasoning_summary,
            "visible_evidence": evidence,
            "candidate_actions": public_actions,
            "budget": budget,
        }
        return (
            "Select the best legal next action for the temporal Wikipedia graph QA "
            "task. A hyperlink may be followed only when useful for the question. "
            "Switch/list revisions when the needed fact is temporal. Submit only "
            "when visible evidence contains both the post-cutoff bridge and final "
            "tail answer.\nPUBLIC STATE:\n" +
            json.dumps(public, ensure_ascii=False, sort_keys=True)
        )

    def control(
        self, case: PublicTemporalCaseV2, state: Any,
        actions: list[JointCandidateActionV23], *, seed: int,
        budget: dict[str, int],
    ) -> CompactJointOutputV24:
        del seed
        prompt = self.prompt_for(case, state, actions, budget)
        scored = self.scorer.score(prompt, actions)
        submission, proposal_audit = self._propose_payload(case, state)
        validation = CompactSubmissionValidationV24(
            "schema_valid_pending_public_gate", False,
            "payload proposer returned compact candidate",
        ) if submission is not None else CompactSubmissionValidationV24(
            "abstained", False, "payload proposer found incomplete evidence",
        )
        return CompactJointOutputV24(
            scores=scored.scores,
            reasoning_summary="Open-weight action scores computed from visible state.",
            extracted_entities=(), evidence_notes=(), submission=submission,
            submission_schema_validation=validation,
            abstain_reason=None if submission else "bridge or tail evidence incomplete",
            attempts=({
                "protocol": OPEN_WEIGHT_LIVE_CONTROLLER_PROTOCOL_V24,
                "backend_name": scored.backend_name,
                "token_counts": scored.token_counts,
                "serialized_actions": scored.serialized_actions,
                "payload_proposer_name": self.payload_proposer_name,
                "payload_proposal": proposal_audit,
                "ranking_contract_valid": set(scored.scores) == {
                    action.action_id for action in actions
                },
            },),
            score_kind="length_normalized_conditional_logprob",
        )


def _log_softmax_v24(scores: dict[str, float]) -> dict[str, float]:
    maximum = max(scores.values())
    normalizer = maximum + math.log(sum(
        math.exp(value - maximum) for value in scores.values()
    ))
    return {key: value - normalizer for key, value in scores.items()}


class HierarchicalOpenWeightLiveControllerV24(OpenWeightLiveControllerV24):
    """Factor P(action|state) into submit mode and conditional graph choice."""

    controller_name = "hierarchical_open_weight_logprob_controller_v2.4"

    @staticmethod
    def _mode_prompt(case: PublicTemporalCaseV2, state: Any) -> str:
        evidence = "\n".join(
            f'[{page["evidence_id"]}] {page["title"]}: '
            f'{str(page.get("content") or "")[:12_000]}'
            for page in state.collected_evidence[-8:]
        )
        return (
            f"Question: {case.question}\nVisible evidence:\n{evidence}\n"
            "Is the visible evidence complete enough to answer?"
        )

    def _mode_scores(
        self, case: PublicTemporalCaseV2, state: Any,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        backend = self.scorer.backend
        batch_method = getattr(backend, "conditional_token_logprobs_batch", None)
        records = []
        assigned: dict[str, list[float]] = {"CONTINUE": [], "SUBMIT": []}
        descriptions = {
            "CONTINUE": "Continue traversing the graph to collect evidence",
            "SUBMIT": "Submit the compact answer with bridge and tail evidence IDs",
        }
        for mapping in MODE_VERBALIZER_ROTATIONS_V24:
            prompt = self._mode_prompt(case, state) + "\n" + "\n".join(
                f"{label}. {descriptions[mode]}" for label, mode in mapping
            ) + "\nAnswer:"
            continuations = [" A", " B"]
            if callable(batch_method):
                token_rows = batch_method(prompt, continuations)
            else:
                token_rows = [
                    backend.conditional_token_logprobs(prompt, continuation)
                    for continuation in continuations
                ]
            label_scores = {
                label: sum(values) / len(values)
                for label, values in zip(("A", "B"), token_rows, strict=True)
            }
            for label, mode in mapping:
                assigned[mode].append(label_scores[label])
            records.append({
                "mapping": [list(row) for row in mapping],
                "label_scores": label_scores,
            })
        raw = {mode: sum(values) / len(values) for mode, values in assigned.items()}
        return _log_softmax_v24(raw), records

    def control(
        self, case: PublicTemporalCaseV2, state: Any,
        actions: list[JointCandidateActionV23], *, seed: int,
        budget: dict[str, int],
    ) -> CompactJointOutputV24:
        del seed
        submit_actions = [action for action in actions if action.kind == "SUBMIT_SLOT"]
        graph_actions = [action for action in actions if action.kind != "SUBMIT_SLOT"]
        if len(submit_actions) != 1:
            raise ValueError("hierarchical scoring needs exactly one submit slot")
        mode_scores, mode_records = self._mode_scores(case, state)
        if graph_actions:
            prompt = self.prompt_for(case, state, graph_actions, budget)
            graph_raw = self.scorer.score(prompt, graph_actions)
            graph_conditional = _log_softmax_v24(graph_raw.scores)
        else:
            graph_raw = None
            graph_conditional = {}
        scores = {action_id: mode_scores["CONTINUE"] + score
                  for action_id, score in graph_conditional.items()}
        scores[submit_actions[0].action_id] = mode_scores["SUBMIT"]
        submission, proposal_audit = self._propose_payload(case, state)
        validation = CompactSubmissionValidationV24(
            "schema_valid_pending_public_gate", False,
            "payload proposer returned compact candidate",
        ) if submission is not None else CompactSubmissionValidationV24(
            "abstained", False, "payload proposer found incomplete evidence",
        )
        return CompactJointOutputV24(
            scores=scores,
            reasoning_summary="Hierarchical open-weight mode and graph scores.",
            extracted_entities=(), evidence_notes=(), submission=submission,
            submission_schema_validation=validation,
            abstain_reason=None if submission else "bridge or tail evidence incomplete",
            attempts=({
                "protocol": "hierarchical-open-weight-live-controller-v2.4",
                "factorization": "P(mode|state)*P(graph_action|continue,state)",
                "mode_log_probabilities": mode_scores,
                "mode_label_rotation_records": mode_records,
                "graph_raw_scores": graph_raw.scores if graph_raw else {},
                "graph_conditional_log_probabilities": graph_conditional,
                "backend_name": (
                    graph_raw.backend_name if graph_raw
                    else self.scorer.backend.backend_name
                ),
                "payload_proposer_name": self.payload_proposer_name,
                "payload_proposal": proposal_audit,
                "ranking_contract_valid": set(scores) == {
                    action.action_id for action in actions
                },
            },),
            score_kind="length_normalized_conditional_logprob",
        )
