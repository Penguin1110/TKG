"""Inference-level beam search over rendered Wikipedia page revisions.

The search engine only accepts a public request plus a Wikipedia backend.  Gold
routes, Wikidata identifiers, answer aliases, and private case dictionaries are
deliberately absent from its interface.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from typing import Any, Protocol

from tkg.experiment.contracts import PageSnapshot
from tkg.wikipedia.backend import WikipediaError, normalize_title


BEAM_TRAJECTORY_SCHEMA = "temporal-graph-beam-trajectory-v2"
ACTION_KINDS = {
    "FOLLOW_LINK", "LIST_REVISIONS", "SWITCH_SNAPSHOT", "SUBMIT_ANSWER",
}


class RankerContractError(ValueError):
    """The action ranker failed its required machine-readable contract."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _fold(value: str) -> str:
    return " ".join(value.casefold().split())


@dataclass(frozen=True)
class TemporalSearchRequest:
    case_id: str
    question: str
    start_page: str
    cutoff_date: str
    target_date: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class RevisionOption:
    revision_id: int
    revision_date: str
    revision_timestamp: str
    as_of: str


@dataclass(frozen=True)
class TemporalAction:
    kind: str
    params: dict[str, Any]
    label: str

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"unsupported temporal action: {self.kind!r}")

    @property
    def action_id(self) -> str:
        return f"{self.kind.casefold()}:{_digest([self.kind, self.params])[:16]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind,
            "params": self.params,
            "label": self.label,
        }


@dataclass(frozen=True)
class RankerOutput:
    scores: dict[str, float]
    score_kind: str
    reasoning_summary: str = ""
    extracted_entities: tuple[str, ...] = ()
    evidence_notes: tuple[str, ...] = ()
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class AnswerProposal:
    """One answer candidate generated independently of graph actions.

    ``supported`` is decided by the controller from public visible evidence.  It
    is never based on a gold answer, private route, or final-answer correctness.
    """

    answer: str
    supporting_evidence_ids: tuple[str, ...] = ()
    supported: bool = False
    support_policy: str = "verbatim_visible_evidence_v1"
    support_reason: str = ""
    raw: dict[str, Any] | None = None


class ActionRanker(Protocol):
    ranker_name: str

    def propose_answer(
        self,
        request: TemporalSearchRequest,
        state: "TemporalBeamState",
        visible_evidence: list[dict[str, Any]],
        *,
        seed: int,
    ) -> AnswerProposal:
        ...

    def rank(
        self,
        request: TemporalSearchRequest,
        state: "TemporalBeamState",
        visible_evidence: list[dict[str, Any]],
        actions: list[TemporalAction],
        *,
        seed: int,
    ) -> RankerOutput:
        ...


@dataclass(frozen=True)
class TemporalBeamState:
    current_page: str
    current_revision_id: int
    current_revision_date: str
    snapshot_as_of: str
    reasoning_summary: str
    extracted_entities: tuple[str, ...]
    temporal_constraints: dict[str, str]
    collected_evidence: tuple[dict[str, Any], ...]
    known_revisions: tuple[RevisionOption, ...]
    visited_nodes: tuple[tuple[str, int], ...]
    action_trace: tuple[dict[str, Any], ...]
    cumulative_score: float
    finished: bool
    submitted_answer: str
    stop_reason: str = ""
    error: str = ""

    @property
    def node_key(self) -> tuple[str, int]:
        return (_fold(self.current_page), self.current_revision_id)

    @property
    def state_id(self) -> str:
        return _digest(self.dedup_key())[:24]

    def dedup_key(self) -> tuple[Any, ...]:
        """Information-state key; page/revision alone is insufficient after LIST."""
        return (
            *self.node_key,
            tuple(option.revision_id for option in self.known_revisions),
            self.finished,
            _fold(self.submitted_answer),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["state_id"] = self.state_id
        result["visited_nodes"] = [list(value) for value in self.visited_nodes]
        return result


@dataclass(frozen=True)
class BeamSearchConfig:
    beam_width: int = 3
    max_expansions: int = 16
    max_actions_per_state: int = 5
    max_links: int = 20
    revision_limit: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        for name in (
            "beam_width", "max_expansions", "max_actions_per_state",
            "max_links", "revision_limit",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.revision_limit > 20:
            raise ValueError("revision_limit must be <= 20")


@dataclass(frozen=True)
class BeamSearchResult:
    request: TemporalSearchRequest
    config: BeamSearchConfig
    ranker_name: str
    final_state: TemporalBeamState
    retained_states: tuple[TemporalBeamState, ...]
    audit_steps: tuple[dict[str, Any], ...]
    expansions: int
    repeated_state_count: int
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BEAM_TRAJECTORY_SCHEMA,
            "request": self.request.to_dict(),
            "config": asdict(self.config),
            "ranker_name": self.ranker_name,
            "final_state": self.final_state.to_dict(),
            "retained_states": [state.to_dict() for state in self.retained_states],
            "audit_steps": list(self.audit_steps),
            "expansions": self.expansions,
            "repeated_state_count": self.repeated_state_count,
            "stop_reason": self.stop_reason,
            "score_recomputed": recompute_cumulative_score(self.final_state),
            "score_recomputable": math.isclose(
                self.final_state.cumulative_score,
                recompute_cumulative_score(self.final_state),
                rel_tol=0.0, abs_tol=1e-12,
            ),
        }


def _evidence(page: PageSnapshot) -> dict[str, Any]:
    return {
        "title": page.title,
        "revision_id": page.revision_id,
        "timestamp": page.timestamp,
        "as_of": page.as_of,
        "content": page.content,
        "links": [asdict(link) for link in page.links],
    }


def initial_beam_state(
    request: TemporalSearchRequest, backend: Any,
) -> TemporalBeamState:
    page = backend.fetch_page(request.start_page, as_of=request.cutoff_date)
    node = (_fold(page.title), page.revision_id)
    return TemporalBeamState(
        current_page=page.title,
        current_revision_id=page.revision_id,
        current_revision_date=page.timestamp[:10],
        snapshot_as_of=request.cutoff_date,
        reasoning_summary="",
        extracted_entities=(),
        temporal_constraints={
            "range_start": request.cutoff_date,
            "range_end": request.target_date,
            "event_time_policy": "world_event_or_tenure_onset_not_observation_time",
        },
        collected_evidence=(_evidence(page),),
        known_revisions=(),
        visited_nodes=(node,),
        action_trace=(),
        cumulative_score=0.0,
        finished=False,
        submitted_answer="",
    )


def _current_page(state: TemporalBeamState, backend: Any) -> PageSnapshot:
    page = backend.fetch_page(state.current_page, as_of=state.snapshot_as_of)
    if page.revision_id != state.current_revision_id:
        raise WikipediaError(
            "cached/current revision no longer matches the serialized beam state"
        )
    return page


def legal_actions(
    state: TemporalBeamState, backend: Any, config: BeamSearchConfig,
) -> tuple[list[TemporalAction], PageSnapshot]:
    actions, page, _ = legal_actions_with_compaction(state, backend, config)
    return actions, page


def legal_actions_with_compaction(
    state: TemporalBeamState, backend: Any, config: BeamSearchConfig,
) -> tuple[list[TemporalAction], PageSnapshot, dict[str, Any]]:
    if state.finished:
        return [], _current_page(state, backend), {
            "policy": "finished_state", "pre_compaction_actions": [],
            "post_compaction_actions": [],
        }
    page = _current_page(state, backend)
    # Preserve revision document order.  Alphabetical truncation can silently
    # remove a prominent incumbent link before the model ever sees it.
    links: dict[str, tuple[str, str]] = {}
    for link in page.links:
        folded = _fold(link.target)
        links.setdefault(folded, (normalize_title(link.target), link.anchor))
    full_link_actions = [
        TemporalAction(
            "FOLLOW_LINK", {"page_title": target},
            f'Follow hyperlink "{anchor}" to {target}',
        )
        for target, anchor in links.values()
    ]
    actions = full_link_actions[:config.max_links]
    non_link_actions: list[TemporalAction] = []
    if state.known_revisions:
        visited = set(state.visited_nodes)
        non_link_actions.extend(
            TemporalAction(
                "SWITCH_SNAPSHOT",
                {
                    "revision_id": option.revision_id,
                    "revision_date": option.revision_date,
                    "as_of": option.as_of,
                },
                f"Switch this page to revision {option.revision_id} "
                f"({option.revision_date})",
            )
            for option in state.known_revisions
            if option.revision_id != state.current_revision_id
            and (_fold(state.current_page), option.revision_id) not in visited
        )
    else:
        non_link_actions.append(TemporalAction(
            "LIST_REVISIONS",
            {
                "from": state.temporal_constraints["range_start"],
                "to": state.temporal_constraints["range_end"],
                "limit": config.revision_limit,
                "sampling_policy": "even_calendar_probe_resolved_revision",
            },
            "List sampled revision IDs and dates for the current page",
        ))
    actions.extend(non_link_actions)
    full_actions = [*full_link_actions, *non_link_actions]
    return actions, page, {
        "policy": "backend_render_order_first_n_links_v1",
        "max_links": config.max_links,
        "pre_compaction_count": len(full_actions),
        "post_compaction_count": len(actions),
        "pre_compaction_actions": [action.to_dict() for action in full_actions],
        "post_compaction_actions": [action.to_dict() for action in actions],
    }


def evidence_id(page: dict[str, Any]) -> str:
    payload = {
        key: page.get(key)
        for key in ("title", "revision_id", "timestamp", "as_of", "content", "links")
    }
    return "evidence_" + _digest(payload)[:24]


def validate_answer_proposal(
    proposal: AnswerProposal,
    visible_evidence: tuple[dict[str, Any], ...],
) -> AnswerProposal:
    """Require a concise answer and its literal support in cited visible pages.

    This deliberately narrow controller gate is auditable and cannot consult a
    hidden reference answer.  Semantic/alias support is a future extension.
    """
    answer = " ".join(proposal.answer.split())
    if not answer:
        return replace(
            proposal, answer="", supported=False,
            support_reason="empty_answer_candidate",
        )
    if len(answer) > 200 or len(answer.split()) > 24:
        return replace(
            proposal, answer=answer, supported=False,
            support_reason="answer_candidate_not_concise",
        )
    rejected = {
        "unknown", "not visible", "not enough information",
        "insufficient information", "more information is needed",
    }
    if _fold(answer).strip(".?!") in rejected:
        return replace(
            proposal, answer=answer, supported=False,
            support_reason="meta_answer_candidate",
        )
    by_id = {evidence_id(page): page for page in visible_evidence}
    cited = tuple(dict.fromkeys(proposal.supporting_evidence_ids))
    if not cited:
        return replace(
            proposal, answer=answer, supported=False,
            support_reason="no_supporting_evidence_cited",
        )
    unknown = [value for value in cited if value not in by_id]
    if unknown:
        return replace(
            proposal, answer=answer, supporting_evidence_ids=cited,
            supported=False, support_reason="unknown_supporting_evidence_id",
        )
    answer_folded = _fold(answer).strip(".?!")
    matched = [
        value for value in cited
        if answer_folded and answer_folded in _fold(str(by_id[value].get("content", "")))
    ]
    if not matched:
        return replace(
            proposal, answer=answer, supporting_evidence_ids=cited,
            supported=False, support_reason="answer_not_verbatim_in_cited_evidence",
        )
    return replace(
        proposal, answer=answer, supporting_evidence_ids=tuple(matched),
        supported=True, support_reason="verbatim_match_in_cited_visible_evidence",
    )


def submit_action_for_proposal(proposal: AnswerProposal) -> TemporalAction | None:
    if not proposal.supported:
        return None
    return TemporalAction(
        "SUBMIT_ANSWER",
        {
            "answer": proposal.answer,
            "supporting_evidence_ids": list(proposal.supporting_evidence_ids),
            "support_policy": proposal.support_policy,
        },
        f'Submit evidence-supported answer "{proposal.answer}"',
    )


def _score_actions(
    actions: list[TemporalAction], output: RankerOutput,
) -> list[dict[str, Any]]:
    expected = {action.action_id for action in actions}
    returned = set(output.scores)
    if returned != expected:
        raise RankerContractError(
            "ranker_score_coverage_mismatch:"
            f"missing={len(expected - returned)}:unexpected={len(returned - expected)}"
        )
    raw_values = {
        action.action_id: float(output.scores[action.action_id])
        for action in actions
    }
    if output.score_kind == "length_normalized_conditional_logprob":
        normalized = raw_values
    else:
        maximum = max(raw_values.values(), default=0.0)
        denominator = sum(math.exp(value - maximum) for value in raw_values.values())
        normalized = {
            key: value - maximum - math.log(denominator)
            for key, value in raw_values.items()
        }
    return [{
        **action.to_dict(),
        "raw_ranker_score": raw_values[action.action_id],
        "action_score": normalized[action.action_id],
        "score_kind": output.score_kind,
    } for action in actions]


def _revision_options(
    state: TemporalBeamState, backend: Any, action: TemporalAction,
) -> tuple[RevisionOption, ...]:
    params = action.params
    start = date.fromisoformat(str(params["from"]))
    end = date.fromisoformat(str(params["to"]))
    limit = int(params["limit"])
    span = (end - start).days
    offsets = (
        [span] if limit == 1
        else sorted({round(index * span / (limit - 1)) for index in range(limit)})
    )
    dates = [(start + timedelta(days=offset)).isoformat() for offset in offsets]
    options: dict[int, RevisionOption] = {}
    for as_of in dates:
        page = backend.fetch_page(state.current_page, as_of=as_of)
        if _fold(page.title) != _fold(state.current_page):
            raise WikipediaError("revision discovery changed the current page")
        options[page.revision_id] = RevisionOption(
            revision_id=page.revision_id,
            revision_date=page.timestamp[:10],
            revision_timestamp=page.timestamp,
            as_of=as_of,
        )
    return tuple(sorted(options.values(), key=lambda row: (
        row.revision_timestamp, row.revision_id,
    )))


def _merge_entities(previous: tuple[str, ...], additions: tuple[str, ...]) -> tuple[str, ...]:
    values = {value for value in previous if value.strip()}
    values.update(value.strip() for value in additions if value.strip())
    return tuple(sorted(values, key=str.casefold))


def _execute_action(
    state: TemporalBeamState,
    action: TemporalAction,
    action_score: float,
    ranker_output: RankerOutput,
    backend: Any,
) -> tuple[TemporalBeamState, dict[str, Any]]:
    page = _current_page(state, backend)
    origin_node = state.node_key
    next_page = page
    known_revisions = state.known_revisions
    evidence = state.collected_evidence
    finished = False
    answer = ""
    stop_reason = ""
    error = ""
    validity: dict[str, bool | None] = {
        "hyperlink_valid": None, "revision_valid": None,
    }
    try:
        if action.kind == "FOLLOW_LINK":
            requested = normalize_title(str(action.params["page_title"]))
            targets = {_fold(link.target): link.target for link in page.links}
            canonical = targets.get(_fold(requested))
            if canonical is None:
                raise ValueError("FOLLOW_LINK target is absent from the current revision")
            next_page = backend.fetch_page(canonical, as_of=state.snapshot_as_of)
            evidence = (*evidence, _evidence(next_page))
            known_revisions = ()
            validity["hyperlink_valid"] = True
        elif action.kind == "LIST_REVISIONS":
            known_revisions = _revision_options(state, backend, action)
            validity["revision_valid"] = True
        elif action.kind == "SWITCH_SNAPSHOT":
            revision_id = int(action.params["revision_id"])
            option = next(
                (row for row in state.known_revisions if row.revision_id == revision_id),
                None,
            )
            if option is None:
                raise ValueError("SWITCH_SNAPSHOT revision was not listed for this page")
            switched = backend.fetch_page(state.current_page, as_of=option.as_of)
            if _fold(switched.title) != _fold(state.current_page):
                raise ValueError("SWITCH_SNAPSHOT changed page title")
            if switched.revision_id != revision_id:
                raise ValueError("SWITCH_SNAPSHOT resolved to a different revision")
            next_page = switched
            evidence = (*evidence, _evidence(next_page))
            validity["revision_valid"] = True
        elif action.kind == "SUBMIT_ANSWER":
            answer = " ".join(str(action.params.get("answer", "")).split())
            finished = True
            stop_reason = "submit_answer" if answer else "empty_answer"
            if not answer:
                error = "SUBMIT_ANSWER did not provide a non-empty answer"
        else:  # pragma: no cover - TemporalAction validates kinds
            raise ValueError(f"unknown action {action.kind}")
    except (WikipediaError, ValueError, KeyError) as exc:
        error = str(exc)
        finished = True
        stop_reason = "action_error"

    node = (_fold(next_page.title), next_page.revision_id)
    visited = state.visited_nodes if node in state.visited_nodes else (*state.visited_nodes, node)
    trace_record = {
        "index": len(state.action_trace) + 1,
        "action": action.to_dict(),
        "action_score": action_score,
        "from_node": list(origin_node),
        "to_node": list(node),
        "result": "error" if error else "ok",
        "error": error,
        **validity,
    }
    child = TemporalBeamState(
        current_page=next_page.title,
        current_revision_id=next_page.revision_id,
        current_revision_date=next_page.timestamp[:10],
        snapshot_as_of=(
            str(action.params["as_of"])
            if action.kind == "SWITCH_SNAPSHOT" and not error
            else state.snapshot_as_of
        ),
        reasoning_summary=ranker_output.reasoning_summary[:2000],
        extracted_entities=_merge_entities(
            state.extracted_entities, ranker_output.extracted_entities,
        ),
        temporal_constraints=state.temporal_constraints,
        collected_evidence=evidence,
        known_revisions=known_revisions,
        visited_nodes=visited,
        action_trace=(*state.action_trace, trace_record),
        cumulative_score=state.cumulative_score + action_score,
        finished=finished,
        submitted_answer=answer,
        stop_reason=stop_reason,
        error=error,
    )
    return child, trace_record


def _stable_order(state: TemporalBeamState, seed: int) -> tuple[float, str]:
    tie = _digest([seed, state.state_id])
    return (-state.cumulative_score, tie)


def recompute_cumulative_score(state: TemporalBeamState) -> float:
    return sum(float(row["action_score"]) for row in state.action_trace)


def run_temporal_beam_search(
    request: TemporalSearchRequest,
    backend: Any,
    ranker: ActionRanker,
    config: BeamSearchConfig,
) -> BeamSearchResult:
    frontier = [initial_beam_state(request, backend)]
    audit_steps: list[dict[str, Any]] = []
    expansions = 0
    repeated_states = 0
    iteration = 0
    stop_reason = "exhausted_search"

    while frontier and expansions < config.max_expansions:
        iteration += 1
        proposals: list[tuple[TemporalBeamState, int, int]] = []
        iteration_audits: list[dict[str, Any]] = []
        for parent_index, state in enumerate(sorted(
            frontier, key=lambda row: _stable_order(row, config.seed),
        )):
            if state.finished:
                proposals.append((state, -1, -1))
                continue
            compaction: dict[str, Any] | None = None
            try:
                graph_actions, current, compaction = legal_actions_with_compaction(
                    state, backend, config,
                )
                visible = [_evidence(current)]
                proposed = ranker.propose_answer(
                    request, state, list(state.collected_evidence), seed=config.seed,
                )
                answer_candidate = validate_answer_proposal(
                    proposed, state.collected_evidence,
                )
                submit_action = submit_action_for_proposal(answer_candidate)
                actions = [*graph_actions]
                if submit_action is not None:
                    actions.append(submit_action)
                    compaction["pre_compaction_actions"].append(submit_action.to_dict())
                    compaction["post_compaction_actions"].append(submit_action.to_dict())
                    compaction["pre_compaction_count"] += 1
                    compaction["post_compaction_count"] += 1
                output = ranker.rank(
                    request, state, visible, actions, seed=config.seed,
                )
                scored = sorted(
                    _score_actions(actions, output),
                    key=lambda row: (
                        -float(row["action_score"]),
                        _digest([config.seed, state.state_id, row["action_id"]]),
                    ),
                )
                audit: dict[str, Any] = {
                    "schema_version": BEAM_TRAJECTORY_SCHEMA,
                    "iteration": iteration,
                    "parent_state": state.to_dict(),
                    "visible_evidence": visible,
                    "candidate_actions": [],
                    "ranker": {
                        "name": ranker.ranker_name,
                        "score_kind": output.score_kind,
                        "reasoning_summary": output.reasoning_summary,
                        "extracted_entities": list(output.extracted_entities),
                        "evidence_notes": list(output.evidence_notes),
                        "raw": output.raw,
                    },
                    "answer_candidate": {
                        **asdict(answer_candidate),
                        "literal_support_gate_passed": answer_candidate.supported,
                        "semantic_relation_support": "not_evaluated",
                    },
                    "action_compaction": compaction,
                    "selected_actions": [],
                    "retained_actions": [],
                }
                proposal_count_before = len(proposals)
                expanded_for_parent = 0
                for candidate_index, scored_action in enumerate(scored):
                    candidate = dict(scored_action)
                    candidate["expanded"] = False
                    candidate["retained"] = False
                    candidate["pruning_reason"] = ""
                    action = next(
                        row for row in actions
                        if row.action_id == candidate["action_id"]
                    )
                    if expanded_for_parent >= config.max_actions_per_state:
                        candidate["pruning_reason"] = "local_expansion_cap"
                    elif expansions >= config.max_expansions:
                        candidate["pruning_reason"] = "max_expansions"
                    else:
                        child, transition = _execute_action(
                            state, action, float(candidate["action_score"]), output, backend,
                        )
                        expansions += 1
                        expanded_for_parent += 1
                        candidate["expanded"] = True
                        candidate["resulting_state"] = child.to_dict()
                        candidate["transition"] = transition
                        audit["selected_actions"].append(candidate["action_id"])
                        revisited = (
                            action.kind in {"FOLLOW_LINK", "SWITCH_SNAPSHOT"}
                            and child.node_key in state.visited_nodes
                        )
                        if revisited:
                            candidate["pruning_reason"] = "repeated_path_node"
                            repeated_states += 1
                        else:
                            proposals.append((
                                child, len(iteration_audits),
                                len(audit["candidate_actions"]),
                            ))
                    audit["candidate_actions"].append(candidate)
                if len(proposals) == proposal_count_before:
                    proposals.append((replace(
                        state,
                        finished=True,
                        stop_reason="exhausted_no_legal_progress",
                    ), -1, -1))
                iteration_audits.append(audit)
            except RankerContractError as exc:
                terminal = replace(
                    state, finished=True, stop_reason="ranker_infrastructure_error",
                    error=str(exc),
                )
                proposals.append((terminal, -1, -1))
                iteration_audits.append({
                    "schema_version": BEAM_TRAJECTORY_SCHEMA,
                    "iteration": iteration,
                    "parent_state": state.to_dict(),
                    "visible_evidence": [],
                    "candidate_actions": [],
                    "selected_actions": [],
                    "retained_actions": [],
                    "pruning_reason": "ranker_infrastructure_error",
                    "error": str(exc),
                    "action_compaction": compaction,
                })
            except (WikipediaError, ValueError, KeyError) as exc:
                terminal = replace(
                    state, finished=True, stop_reason="ranker_or_backend_error",
                    error=str(exc),
                )
                proposals.append((terminal, -1, -1))
                iteration_audits.append({
                    "schema_version": BEAM_TRAJECTORY_SCHEMA,
                    "iteration": iteration,
                    "parent_state": state.to_dict(),
                    "visible_evidence": [],
                    "candidate_actions": [],
                    "selected_actions": [],
                    "retained_actions": [],
                    "pruning_reason": "ranker_or_backend_error",
                    "error": str(exc),
                })

        best_by_key: dict[tuple[Any, ...], tuple[TemporalBeamState, int, int]] = {}
        for proposal in proposals:
            child, audit_index, candidate_index = proposal
            key = child.dedup_key()
            previous = best_by_key.get(key)
            if previous is None or _stable_order(child, config.seed) < _stable_order(
                previous[0], config.seed,
            ):
                if previous is not None and previous[1] >= 0:
                    old = iteration_audits[previous[1]]["candidate_actions"][previous[2]]
                    old["pruning_reason"] = "duplicate_state_lower_score"
                    repeated_states += 1
                best_by_key[key] = proposal
            else:
                if audit_index >= 0:
                    candidate = iteration_audits[audit_index]["candidate_actions"][candidate_index]
                    candidate["pruning_reason"] = "duplicate_state_lower_score"
                repeated_states += 1

        ordered = sorted(
            best_by_key.values(), key=lambda row: _stable_order(row[0], config.seed),
        )
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
                "all_retained_error"
                if all(state.error for state in frontier)
                else "all_retained_finished"
            )
            break

    if expansions >= config.max_expansions and not all(
        state.finished for state in frontier
    ):
        stop_reason = "max_expansions"
    elif not frontier:
        stop_reason = "exhausted_search"
    finished = [state for state in frontier if state.finished and state.submitted_answer]
    final_pool = finished or frontier
    if not final_pool:
        raise RuntimeError("beam search produced no final or retained state")
    final_state = sorted(
        final_pool, key=lambda row: _stable_order(row, config.seed),
    )[0]
    return BeamSearchResult(
        request=request,
        config=config,
        ranker_name=ranker.ranker_name,
        final_state=final_state,
        retained_states=tuple(frontier),
        audit_steps=tuple(audit_steps),
        expansions=expansions,
        repeated_state_count=repeated_states,
        stop_reason=stop_reason,
    )


def run_temporal_greedy_search(
    request: TemporalSearchRequest,
    backend: Any,
    ranker: ActionRanker,
    config: BeamSearchConfig,
) -> BeamSearchResult:
    """Greedy reference using the exact same transition and scoring machinery."""
    return run_temporal_beam_search(
        request, backend, ranker, replace(config, beam_width=1),
    )
