"""Gold-free evidence-pheromone ant search over the frozen v2.4 action space.

This module is append-only.  It reuses v2.4 candidate generation, transition
validation, action serialization, and model scores, but replaces deterministic
global top-k retention with seeded per-ant sampling and a public-only pheromone
ledger.  Private evaluation objects are intentionally absent from every search
entry point.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from tkg.experiment.joint_controller_v23 import (
    JointControllerContractErrorV23, SUBMIT_SLOT_ID_V23,
)
from tkg.experiment.temporal_environment_v2 import TemporalWikipediaEnvironmentV2
from tkg.experiment.temporal_eval_schema_v2 import PublicTemporalCaseV2
from tkg.experiment.temporal_live_runner_v22 import (
    LiveBeamStateV22, _EnvironmentManifestCacheV22,
    _snapshot_for_state, initial_live_state_v22,
)
from tkg.experiment.temporal_live_runner_v23 import (
    LiveSearchConfigV23, _normalized_joint_scores,
)
from tkg.experiment.temporal_live_runner_v24 import (
    _execute_v24, _graph_and_control_actions_v24, _instantiate_submit_v24,
    _state_v24,
)
from tkg.wikipedia.backend import WikipediaError


ANT_SEARCH_SCHEMA_V26 = "temporal-evidence-ant-search-v2.6"
ANT_STEP_SCHEMA_V26 = "temporal-evidence-ant-step-v2.6"
PHEROMONE_SCHEMA_V26 = "bounded-evidence-pheromone-ledger-v2.6"
METHODS_V26 = ("STOCHASTIC_LM", "STRUCTURAL_ACO", "EVIDENCE_ACO")
FORBIDDEN_PRIVATE_KEYS_V26 = frozenset({
    "private", "private_route", "reference_route", "reference_routes",
    "accepted_answer", "accepted_answers", "accepted_final_answer_aliases",
    "aliases", "qid", "qids", "expected_entity", "expected_revision",
    "distance_to_gold", "correctness", "gold", "gold_route",
})


def _fold(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_public_only_v26(value: Any, *, path: str = "root") -> None:
    """Fail closed if a search input contains a known private-evaluation key."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            folded = _fold(key).replace(" ", "_")
            if folded in FORBIDDEN_PRIVATE_KEYS_V26:
                raise ValueError(f"private-key leakage at {path}.{key}")
            assert_public_only_v26(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_public_only_v26(child, path=f"{path}[{index}]")


@dataclass(frozen=True)
class AntSearchConfigV26:
    method: str
    seed: int
    max_expansions: int = 40
    ant_count: int = 4
    max_steps_per_ant: int = 10
    dense_action_limit: int = 30
    link_page_size: int = 50
    revision_page_size: int = 50
    max_environment_queries_per_node: int = 4
    alpha: float = 0.8
    beta: float = 1.0
    gamma: float = 0.35
    evaporation: float = 0.15
    pheromone_initial: float = 1.0
    pheromone_min: float = 0.05
    pheromone_max: float = 5.0
    epsilon: float = 1e-9
    credit_horizon: int = 4
    credit_decay: float = 0.6

    def __post_init__(self) -> None:
        if self.method not in METHODS_V26:
            raise ValueError(f"unsupported ant method: {self.method}")
        if self.max_expansions <= 0 or self.ant_count <= 0:
            raise ValueError("expansion and ant counts must be positive")
        if self.max_steps_per_ant <= 0:
            raise ValueError("max_steps_per_ant must be positive")
        if self.ant_count * self.max_steps_per_ant < self.max_expansions:
            raise ValueError("ant horizons cannot satisfy the expansion budget")
        if self.dense_action_limit != 30:
            raise ValueError("v2.6 kill test freezes dense_action_limit=30")
        if not 0.0 <= self.evaporation < 1.0:
            raise ValueError("evaporation must be in [0, 1)")
        if not 0.0 < self.pheromone_min <= self.pheromone_initial <= self.pheromone_max:
            raise ValueError("invalid pheromone bounds")
        if self.epsilon <= 0.0 or self.credit_horizon <= 0:
            raise ValueError("epsilon and credit_horizon must be positive")
        if not 0.0 < self.credit_decay <= 1.0:
            raise ValueError("credit_decay must be in (0, 1]")
        if self.method == "STOCHASTIC_LM" and (self.alpha != 0.0 or self.gamma != 0.0):
            raise ValueError("STOCHASTIC_LM requires alpha=gamma=0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def frozen_method_config_v26(method: str, seed: int) -> AntSearchConfigV26:
    if method == "STOCHASTIC_LM":
        return AntSearchConfigV26(
            method=method, seed=seed, alpha=0.0, gamma=0.0,
        )
    if method == "STRUCTURAL_ACO":
        return AntSearchConfigV26(
            method=method, seed=seed, alpha=0.8, gamma=0.20,
        )
    if method == "EVIDENCE_ACO":
        return AntSearchConfigV26(
            method=method, seed=seed, alpha=0.8, gamma=0.35,
        )
    raise ValueError(f"unsupported ant method: {method}")


@dataclass(frozen=True)
class PublicProgressV26:
    valid_transition: bool
    new_node: bool
    repeated_node: bool
    valid_temporal_switch: bool
    new_evidence_count: int
    new_retrieved_action_count: int
    question_token_overlap: int
    post_cutoff_evidence: bool
    public_submission_complete: bool
    structural_reward: float
    evidence_reward: float
    zero_progress: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_STOPWORDS = frozenset({
    "a", "an", "and", "after", "as", "at", "be", "became", "by", "for",
    "from", "had", "has", "have", "identified", "in", "is", "it", "knowledge",
    "model", "next", "of", "on", "or", "person", "question", "registered",
    "step", "that", "the", "this", "to", "was", "what", "which", "who",
    "with", "previous", "tested", "cutoff", "snapshot", "target",
})
_ACTION_WORDS = frozenset({
    "follow", "hyperlink", "switch", "revision", "list", "next", "page",
    "from", "cursor", "submit", "compact", "answer", "visible", "evidence",
})


def _tokens(value: Any) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", _fold(value))
        if len(token) >= 3 and token not in _STOPWORDS
    }


def edge_key_v26(state: LiveBeamStateV22, action: Mapping[str, Any]) -> str:
    return "|".join((
        _fold(state.current_page), str(state.current_revision_id),
        str(action.get("action_id") or ""),
    ))


def candidate_public_heuristic_v26(
    case: PublicTemporalCaseV2, state: LiveBeamStateV22,
    action: Mapping[str, Any], *, evidence_conditioned: bool,
) -> dict[str, float]:
    """Deterministic public-only pre-transition heuristic."""
    kind = str(action.get("kind") or "")
    params = action.get("params") or {}
    structural = 0.0
    visited_pages = {_fold(page) for page, _ in state.visited_nodes}
    if kind == "FOLLOW_LINK":
        target = _fold(params.get("page_title"))
        structural = -0.75 if target in visited_pages else 0.35
    elif kind == "SWITCH_SNAPSHOT":
        key = (_fold(state.current_page), int(params.get("revision_id", -1)))
        structural = -0.75 if key in state.visited_nodes else 0.30
        timestamp = str(params.get("revision_timestamp") or "")[:10]
        if timestamp and timestamp > case.cutoff_date:
            structural += 0.15
    elif kind in {"LIST_LINKS", "LIST_REVISIONS"}:
        structural = 0.05
    elif kind == "SUBMIT_SLOT":
        structural = 1.0

    evidence = 0.0
    if evidence_conditioned:
        question = _tokens(case.question)
        label = _tokens(action.get("label")) - _ACTION_WORDS
        overlap = len(question & label)
        evidence = min(1.0, overlap / 3.0)
        if kind == "SWITCH_SNAPSHOT":
            timestamp = str(params.get("revision_timestamp") or "")[:10]
            if timestamp and timestamp > case.cutoff_date:
                evidence += 0.25
    return {
        "structural": structural,
        "evidence": evidence,
        "total": structural + evidence,
    }


def transition_public_progress_v26(
    case: PublicTemporalCaseV2, parent: LiveBeamStateV22,
    child: LiveBeamStateV22, action: Mapping[str, Any],
) -> PublicProgressV26:
    """Judge progress using only public question, action, and visible states."""
    kind = str(action.get("kind") or "")
    valid = not bool(child.error)
    node_was_seen = child.node_key in parent.visited_nodes
    changed_node_action = kind in {"FOLLOW_LINK", "SWITCH_SNAPSHOT"}
    new_node = valid and changed_node_action and not node_was_seen
    repeated = valid and changed_node_action and node_was_seen
    temporal = valid and kind == "SWITCH_SNAPSHOT"
    new_evidence = max(0, len(child.collected_evidence) - len(parent.collected_evidence))
    before_actions = {
        row.action_id for row in (
            *parent.retrieved_link_actions, *parent.retrieved_revision_actions,
        )
    }
    after_actions = {
        row.action_id for row in (
            *child.retrieved_link_actions, *child.retrieved_revision_actions,
        )
    }
    new_retrieved = len(after_actions - before_actions)
    new_pages = child.collected_evidence[len(parent.collected_evidence):]
    question_tokens = _tokens(case.question)
    overlap = len(question_tokens & set().union(*(
        _tokens(page.get("content")) for page in new_pages
    ))) if new_pages else 0
    post_cutoff = any(
        str(page.get("timestamp") or "")[:10] > case.cutoff_date
        for page in new_pages
    )
    submission = bool(
        child.submitted and child.finished and not child.error
        and kind in {"SUBMIT_SLOT", "SUBMIT_ANSWER"}
    )

    structural = 0.0
    if not valid:
        structural -= 1.0
    if new_node:
        structural += 1.0
    if temporal:
        structural += 0.5
    if new_evidence:
        structural += min(0.5, 0.25 * new_evidence)
    if new_retrieved:
        structural += min(0.4, 0.1 * new_retrieved)
    if repeated:
        structural -= 0.75

    evidence = 0.0
    if overlap:
        evidence += min(1.0, overlap / 4.0)
    if post_cutoff:
        evidence += 0.5
    if submission:
        evidence += 2.0
    zero = not any((new_node, temporal, new_evidence, new_retrieved, submission))
    if valid and zero:
        structural -= 0.1
    return PublicProgressV26(
        valid_transition=valid, new_node=new_node, repeated_node=repeated,
        valid_temporal_switch=temporal, new_evidence_count=new_evidence,
        new_retrieved_action_count=new_retrieved,
        question_token_overlap=overlap, post_cutoff_evidence=post_cutoff,
        public_submission_complete=submission,
        structural_reward=structural, evidence_reward=evidence,
        zero_progress=zero,
    )


def _log_softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in values))
    return [value - normalizer for value in values]


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [value / total for value in weights]


def score_and_sample_action_v26(
    *, case: PublicTemporalCaseV2, state: LiveBeamStateV22,
    candidates: Sequence[dict[str, Any]], pheromones: Mapping[str, float],
    config: AntSearchConfigV26, rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not candidates:
        raise ValueError("cannot sample an empty candidate set")
    normalized_lm = _log_softmax([float(row["action_score"]) for row in candidates])
    rows: list[dict[str, Any]] = []
    logits = []
    for candidate, lm_score in zip(candidates, normalized_lm, strict=True):
        key = edge_key_v26(state, candidate)
        tau = float(pheromones.get(key, config.pheromone_initial))
        heuristic = candidate_public_heuristic_v26(
            case, state, candidate,
            evidence_conditioned=config.method == "EVIDENCE_ACO",
        )
        selection = (
            config.beta * lm_score
            + config.alpha * math.log(tau + config.epsilon)
            + config.gamma * heuristic["total"]
        )
        logits.append(selection)
        rows.append({
            **candidate,
            "edge_key": key,
            "normalized_lm_action_score": lm_score,
            "pheromone_before_selection": tau,
            "log_pheromone": math.log(tau + config.epsilon),
            "public_progress_heuristic": heuristic,
            "selection_score": selection,
        })
    probabilities = _softmax(logits)
    draw = rng.random()
    cumulative = 0.0
    selected_index = len(rows) - 1
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw <= cumulative:
            selected_index = index
            break
    for index, (row, probability) in enumerate(zip(rows, probabilities, strict=True)):
        row["selection_probability"] = probability
        row["selected"] = index == selected_index
        row["rng_draw"] = draw if index == selected_index else None
    return rows[selected_index], rows


@dataclass
class PheromoneLedgerV26:
    config: AntSearchConfigV26
    values: dict[str, float]
    history: list[dict[str, Any]]

    @classmethod
    def create(cls, config: AntSearchConfigV26) -> "PheromoneLedgerV26":
        return cls(config=config, values={}, history=[])

    def value(self, key: str) -> float:
        return self.values.get(key, self.config.pheromone_initial)

    def register(self, keys: Iterable[str]) -> None:
        for key in keys:
            self.values.setdefault(key, self.config.pheromone_initial)

    def snapshot(self, keys: Iterable[str]) -> dict[str, float]:
        return {key: self.value(key) for key in keys}

    def update(
        self, *, selected_edge: str, ant_path: Sequence[str], reward: float,
        global_step: int,
    ) -> dict[str, Any]:
        before = dict(sorted(self.values.items()))
        if self.config.method == "STOCHASTIC_LM":
            record = {
                "schema_version": PHEROMONE_SCHEMA_V26,
                "global_step": global_step, "method": self.config.method,
                "evaporation": 0.0, "reward": reward,
                "before": before, "updates": [], "after": before,
                "table_sha256": _sha(before),
            }
            self.history.append(record)
            return record

        evaporated = {
            key: max(self.config.pheromone_min, value * (1.0 - self.config.evaporation))
            for key, value in before.items()
        }
        self.values = evaporated
        deltas: dict[str, float] = {}
        if reward > 0.0:
            recent = list(ant_path)[-self.config.credit_horizon:]
            for distance, edge in enumerate(reversed(recent)):
                deltas[edge] = deltas.get(edge, 0.0) + reward * (
                    self.config.credit_decay ** distance
                )
        elif reward < 0.0:
            deltas[selected_edge] = reward
        updates = []
        for key in sorted(deltas):
            old = self.value(key)
            new = min(
                self.config.pheromone_max,
                max(self.config.pheromone_min, old + deltas[key]),
            )
            self.values[key] = new
            updates.append({"edge_key": key, "before_deposit": old,
                            "delta": deltas[key], "after_deposit": new})
        after = dict(sorted(self.values.items()))
        record = {
            "schema_version": PHEROMONE_SCHEMA_V26,
            "global_step": global_step, "method": self.config.method,
            "evaporation": self.config.evaporation, "reward": reward,
            "before": before, "after_evaporation": evaporated,
            "updates": updates, "after": after,
            "table_sha256": _sha(after),
        }
        self.history.append(record)
        return record


def replay_pheromone_history_v26(
    config: AntSearchConfigV26, history: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for record in history:
        recorded_before = {
            str(key): float(value) for key, value in (
                record.get("before") or {}
            ).items()
        }
        for key, value in recorded_before.items():
            values.setdefault(key, value)
        if values != recorded_before:
            raise ValueError(
                f"pheromone pre-update mismatch at step {record.get('global_step')}"
            )
        if record.get("method") == "STOCHASTIC_LM":
            values = {str(key): float(value) for key, value in (
                record.get("after") or {}
            ).items()}
            if _sha(dict(sorted(values.items()))) != record.get("table_sha256"):
                raise ValueError("pheromone table hash mismatch")
            continue
        values = {
            key: max(config.pheromone_min, value * (1.0 - config.evaporation))
            for key, value in values.items()
        }
        for update in record.get("updates") or []:
            key = str(update["edge_key"])
            old = values.get(key, config.pheromone_initial)
            new = min(
                config.pheromone_max,
                max(config.pheromone_min, old + float(update["delta"])),
            )
            values[key] = new
        expected = record.get("after") or {}
        if values != expected:
            raise ValueError(f"pheromone replay mismatch at step {record.get('global_step')}")
        if _sha(dict(sorted(values.items()))) != record.get("table_sha256"):
            raise ValueError("pheromone table hash mismatch")
    return values


@dataclass
class _AntRuntimeV26:
    ant_id: int
    state: LiveBeamStateV22
    edge_path: list[str]
    step_count: int = 0
    structural_progress: float = 0.0
    evidence_progress: float = 0.0
    terminal_reason: str = ""


@dataclass(frozen=True)
class TemporalEvidenceAntSearchResultV26:
    public_case: PublicTemporalCaseV2
    config: AntSearchConfigV26
    ants: tuple[dict[str, Any], ...]
    audit_steps: tuple[dict[str, Any], ...]
    pheromone_history: tuple[dict[str, Any], ...]
    final_pheromones: dict[str, float]
    environment_manifests: tuple[dict[str, Any], ...]
    expansions: int
    parent_states_scored: int
    scored_actions: int
    controller_calls: int
    repeated_or_cyclic_transitions: int
    stop_reason: str
    schema_version: str = ANT_SEARCH_SCHEMA_V26

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "public_case": self.public_case.to_dict(),
            "config": self.config.to_dict(),
            "ants": list(self.ants),
            "audit_steps": list(self.audit_steps),
            "pheromone_history": list(self.pheromone_history),
            "final_pheromones": dict(sorted(self.final_pheromones.items())),
            "environment_manifests": list(self.environment_manifests),
            "expansions": self.expansions,
            "executed_transitions": self.expansions,
            "parent_states_scored": self.parent_states_scored,
            "scored_actions": self.scored_actions,
            "controller_calls": self.controller_calls,
            "repeated_or_cyclic_transitions": self.repeated_or_cyclic_transitions,
            "stop_reason": self.stop_reason,
            "pheromone_replay_valid": replay_pheromone_history_v26(
                self.config, self.pheromone_history,
            ) == self.final_pheromones,
            "inference_private_inputs_used": False,
            "formal_conclusion_allowed": False,
            "method_superiority_claim_allowed": False,
        }


def _terminal_state(state: LiveBeamStateV22, reason: str) -> LiveBeamStateV22:
    return _state_v24(replace(
        state, finished=True, submitted=None, stop_reason=reason, error="",
    ))


def run_temporal_evidence_ant_search_v26(
    *, public_case: PublicTemporalCaseV2, backend: Any,
    environment: TemporalWikipediaEnvironmentV2, controller: Any,
    config: AntSearchConfigV26,
) -> TemporalEvidenceAntSearchResultV26:
    """Execute seeded ant search without accepting any private evaluation data."""
    assert_public_only_v26(public_case.to_dict(), path="public_case")
    assert_public_only_v26(config.to_dict(), path="config")
    live_config = LiveSearchConfigV23(
        beam_width=1, max_expansions=config.max_expansions,
        max_actions_per_state=1, link_page_size=config.link_page_size,
        revision_page_size=config.revision_page_size,
        max_environment_queries_per_node=config.max_environment_queries_per_node,
        dense_action_limit=config.dense_action_limit, seed=config.seed,
    )
    initial = _state_v24(initial_live_state_v22(public_case, backend))
    ants = [
        _AntRuntimeV26(ant_id=index, state=initial, edge_path=[])
        for index in range(config.ant_count)
    ]
    rng = random.Random(config.seed)
    ledger = PheromoneLedgerV26.create(config)
    manifests = _EnvironmentManifestCacheV22(environment, public_case, live_config)
    audits: list[dict[str, Any]] = []
    expansions = parent_states_scored = scored_actions = controller_calls = 0
    repeated = 0
    cursor = 0

    while expansions < config.max_expansions:
        eligible = [
            ant for ant in ants
            if not ant.state.finished and ant.step_count < config.max_steps_per_ant
        ]
        if not eligible:
            break
        ant = ants[cursor % len(ants)]
        cursor += 1
        if ant.state.finished or ant.step_count >= config.max_steps_per_ant:
            continue
        state = ant.state
        try:
            manifest = manifests.get(_snapshot_for_state(state, backend))
            actions, funnel = _graph_and_control_actions_v24(
                state, public_case, live_config,
            )
            output = controller.control(
                public_case, state, actions, seed=config.seed,
                budget={
                    "expansions_used": expansions,
                    "max_expansions": config.max_expansions,
                    "beam_width": 1,
                    "max_actions_per_state": 1,
                },
            )
            controller_calls += len(output.attempts)
            parent_states_scored += 1
            scored = _normalized_joint_scores(actions, output)
            scored_actions += len(scored)
            submit, validation, payload_hash = _instantiate_submit_v24(state, output)
            executable = [
                row for row in scored
                if row["action_id"] != SUBMIT_SLOT_ID_V23 or submit is not None
            ]
            if not executable:
                ant.state = _terminal_state(state, "exhausted_no_legal_progress")
                ant.terminal_reason = ant.state.stop_reason
                audits.append({
                    "schema_version": ANT_STEP_SCHEMA_V26,
                    "global_step": expansions, "ant_id": ant.ant_id,
                    "parent_state": state.to_dict(), "candidate_actions": [],
                    "selected_action": None,
                    "terminal_status": "exhausted_no_legal_progress",
                })
                continue
            ledger.register(edge_key_v26(state, row) for row in executable)
            selected, candidate_records = score_and_sample_action_v26(
                case=public_case, state=state, candidates=executable,
                pheromones=ledger.values, config=config, rng=rng,
            )
            child = _execute_v24(
                state=state, candidate=selected, output=output, submit=submit,
                validation=validation, payload_hash=payload_hash,
                environment=environment, backend=backend, case=public_case,
            )
            expansions += 1
            ant.step_count += 1
            edge = str(selected["edge_key"])
            ant.edge_path.append(edge)
            progress = transition_public_progress_v26(
                public_case, state, child, selected,
            )
            ant.structural_progress += progress.structural_reward
            ant.evidence_progress += progress.evidence_reward
            reward = 0.0
            if config.method == "STRUCTURAL_ACO":
                reward = progress.structural_reward
            elif config.method == "EVIDENCE_ACO":
                reward = progress.structural_reward + progress.evidence_reward
            update = ledger.update(
                selected_edge=edge, ant_path=ant.edge_path,
                reward=reward, global_step=expansions,
            )
            if progress.repeated_node:
                repeated += 1
            selected_with_result = dict(selected)
            selected_with_result["resulting_state"] = child.to_dict()
            selected_with_result["public_progress"] = progress.to_dict()
            audit = {
                "schema_version": ANT_STEP_SCHEMA_V26,
                "global_step": expansions, "ant_id": ant.ant_id,
                "ant_step": ant.step_count, "parent_state": state.to_dict(),
                "visible_evidence_ids": [
                    row["evidence_id"] for row in state.collected_evidence
                ],
                "action_funnel": {
                    **funnel,
                    "environment_legal_action_count": manifest.action_count,
                    "environment_legal_actions_sha256": manifest.actions_sha256,
                    "environment_legal_actions_artifact_reference": manifest.manifest_id,
                },
                "controller": {
                    "name": controller.controller_name,
                    "score_kind": output.score_kind,
                    "attempts": list(output.attempts),
                    "submission_validation": validation.to_dict(),
                },
                "candidate_actions": candidate_records,
                "selected_action": selected_with_result,
                "public_progress_judgment": progress.to_dict(),
                "pheromone_update": update,
                "private_progress_or_correctness_used": False,
            }
            audits.append(audit)
            ant.state = child
            if child.finished:
                ant.terminal_reason = child.stop_reason or "finished"
        except JointControllerContractErrorV23 as exc:
            ant.state = _state_v24(replace(
                state, finished=True, stop_reason="ranking_contract_failure",
                error=str(exc),
            ))
            ant.terminal_reason = ant.state.stop_reason
            audits.append({
                "schema_version": ANT_STEP_SCHEMA_V26,
                "global_step": expansions, "ant_id": ant.ant_id,
                "parent_state": state.to_dict(), "candidate_actions": [],
                "selected_action": None, "error": str(exc),
                "terminal_status": "ranking_contract_failure",
            })
        except (KeyError, TypeError, ValueError, WikipediaError) as exc:
            ant.state = _state_v24(replace(
                state, finished=True, stop_reason="runner_or_environment_error",
                error=str(exc),
            ))
            ant.terminal_reason = ant.state.stop_reason
            audits.append({
                "schema_version": ANT_STEP_SCHEMA_V26,
                "global_step": expansions, "ant_id": ant.ant_id,
                "parent_state": state.to_dict(), "candidate_actions": [],
                "selected_action": None, "error": str(exc),
                "terminal_status": "runner_or_environment_error",
            })

    stop_reason = (
        "max_expansions" if expansions >= config.max_expansions
        else "all_ants_terminal_or_horizon_exhausted"
    )
    ant_payloads = []
    for ant in ants:
        terminal = ant.terminal_reason
        if not terminal and ant.step_count >= config.max_steps_per_ant:
            terminal = "ant_horizon_exhausted"
        ant_payloads.append({
            "ant_id": ant.ant_id, "step_count": ant.step_count,
            "state": ant.state.to_dict(), "edge_path": list(ant.edge_path),
            "structural_progress": ant.structural_progress,
            "evidence_progress": ant.evidence_progress,
            "terminal_reason": terminal,
        })
    return TemporalEvidenceAntSearchResultV26(
        public_case=public_case, config=config, ants=tuple(ant_payloads),
        audit_steps=tuple(audits), pheromone_history=tuple(ledger.history),
        final_pheromones=dict(sorted(ledger.values.items())),
        environment_manifests=tuple(
            row.to_dict() for row in manifests.values()
        ),
        expansions=expansions, parent_states_scored=parent_states_scored,
        scored_actions=scored_actions, controller_calls=controller_calls,
        repeated_or_cyclic_transitions=repeated, stop_reason=stop_reason,
    )
