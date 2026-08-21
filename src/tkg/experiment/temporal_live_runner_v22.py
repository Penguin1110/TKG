"""Public-only live open-world temporal beam search v2.2.

This runner composes the frozen v2 environment/evaluation primitives without
accepting a private case.  Private claims, aliases, witnesses and reference
routes can only be supplied later to a post-hoc evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_beam import RankerContractError
from tkg.experiment.temporal_environment_v2 import (
    EnvironmentActionV2, TemporalWikipediaEnvironmentV2,
    execute_environment_query_v2, initial_environment_queries_v2,
)
from tkg.experiment.temporal_eval_schema_v2 import (
    PUBLIC_CASE_SCHEMA_V2, PublicTemporalCaseV2, StructuredSubmissionV2,
    structured_submission_from_dict,
)
from tkg.experiment.temporal_evaluation_v2 import evidence_id_v2
from tkg.experiment.temporal_live_ranker_v22 import (
    ApiLiveActionRankerV22, LiveActionRankerV22, LiveRankOutputV22,
)
from tkg.experiment.temporal_submission_v2 import StructuredSubmissionProposerV2
from tkg.wikipedia.backend import WikipediaError, WikipediaPageBackend


LIVE_STATE_SCHEMA_V22 = "open-world-temporal-live-state-v2.2"
LIVE_TRAJECTORY_SCHEMA_V22 = "open-world-temporal-live-trajectory-v2.2"
LIVE_RUN_SCHEMA_V22 = "open-world-temporal-live-run-v2.2"
PUBLIC_MANIFEST_SCHEMA_V22 = "open-world-temporal-public-manifest-v2.2"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _fold(value: Any) -> str:
    return " ".join(str(value).casefold().split()).strip(" .?!")


def _page_evidence(page: Any) -> dict[str, Any]:
    value = page.to_dict()
    value["evidence_id"] = evidence_id_v2(value)
    return value


class LiveSubmissionProposerV22(Protocol):
    def propose(
        self, public_case: PublicTemporalCaseV2,
        evidence_pages: list[dict[str, Any]], *, seed: int,
    ) -> tuple[StructuredSubmissionV2, dict[str, Any]]:
        ...


@dataclass(frozen=True)
class LiveSearchConfigV22:
    beam_width: int = 3
    max_expansions: int = 40
    max_actions_per_state: int = 4
    link_page_size: int = 50
    revision_page_size: int = 50
    max_environment_queries_per_node: int = 4
    dense_action_limit: int = 30
    seed: int = 0

    def __post_init__(self) -> None:
        for name in (
            "beam_width", "max_expansions", "max_actions_per_state",
            "link_page_size", "revision_page_size",
            "max_environment_queries_per_node", "dense_action_limit",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.link_page_size > 500 or self.revision_page_size > 500:
            raise ValueError("environment page size cannot exceed 500")
        if self.dense_action_limit > 30:
            raise ValueError("v2.2 dense action limit cannot exceed 30")


@dataclass(frozen=True)
class LiveBeamStateV22:
    current_page: str
    current_revision_id: int
    current_revision_timestamp: str
    snapshot_as_of: str
    reasoning_summary: str
    extracted_entities: tuple[str, ...]
    collected_evidence: tuple[dict[str, Any], ...]
    retrieved_link_actions: tuple[EnvironmentActionV2, ...]
    retrieved_revision_actions: tuple[EnvironmentActionV2, ...]
    link_query_started: bool
    link_next_cursor: str | None
    links_exhausted: bool
    revision_query_started: bool
    revision_next_cursor: str | None
    revisions_exhausted: bool
    environment_queries_used: int
    visited_nodes: tuple[tuple[str, int], ...]
    action_trace: tuple[dict[str, Any], ...]
    cumulative_score: float
    finished: bool
    submitted: dict[str, Any] | None
    stop_reason: str = ""
    error: str = ""
    schema_version: str = LIVE_STATE_SCHEMA_V22

    @property
    def node_key(self) -> tuple[str, int]:
        return (_fold(self.current_page), self.current_revision_id)

    def dedup_key(self) -> tuple[Any, ...]:
        return (
            *self.node_key,
            self.snapshot_as_of,
            self.reasoning_summary,
            tuple(self.extracted_entities),
            tuple(sorted(page["evidence_id"] for page in self.collected_evidence)),
            tuple(row.action_id for row in self.retrieved_link_actions),
            tuple(row.action_id for row in self.retrieved_revision_actions),
            self.link_query_started,
            self.link_next_cursor,
            self.links_exhausted,
            self.revision_query_started,
            self.revision_next_cursor,
            self.revisions_exhausted,
            self.environment_queries_used,
            self.finished,
            _canonical_hash(self.submitted) if self.submitted else "",
        )

    @property
    def state_id(self) -> str:
        return _canonical_hash(self.dedup_key())[:24]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["state_id"] = self.state_id
        result["retrieved_link_actions"] = [
            row.to_dict() for row in self.retrieved_link_actions
        ]
        result["retrieved_revision_actions"] = [
            row.to_dict() for row in self.retrieved_revision_actions
        ]
        result["visited_nodes"] = [list(row) for row in self.visited_nodes]
        return result


@dataclass(frozen=True)
class EnvironmentNodeManifestV22:
    manifest_id: str
    page_title: str
    revision_id: int
    time_window: tuple[str, str]
    actions: tuple[EnvironmentActionV2, ...]
    action_count: int
    actions_sha256: str
    link_count: int
    revision_count: int
    link_pages: int
    revision_pages: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["actions"] = [row.to_dict() for row in self.actions]
        result["time_window"] = list(self.time_window)
        return result


@dataclass(frozen=True)
class LiveSearchResultV22:
    public_case: PublicTemporalCaseV2
    config: LiveSearchConfigV22
    ranker_name: str
    final_state: LiveBeamStateV22
    retained_states: tuple[LiveBeamStateV22, ...]
    audit_steps: tuple[dict[str, Any], ...]
    environment_manifests: tuple[EnvironmentNodeManifestV22, ...]
    expansions: int
    repeated_state_count: int
    stop_reason: str
    schema_version: str = LIVE_TRAJECTORY_SCHEMA_V22

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "public_case": self.public_case.to_dict(),
            "config": asdict(self.config),
            "ranker_name": self.ranker_name,
            "final_state": self.final_state.to_dict(),
            "retained_states": [row.to_dict() for row in self.retained_states],
            "audit_steps": list(self.audit_steps),
            "environment_manifests": [
                row.to_dict() for row in self.environment_manifests
            ],
            "expansions": self.expansions,
            "repeated_state_count": self.repeated_state_count,
            "stop_reason": self.stop_reason,
            "score_recomputed": sum(
                float(row["action_score"])
                for row in self.final_state.action_trace
            ),
            "score_recomputable": math.isclose(
                self.final_state.cumulative_score,
                sum(float(row["action_score"])
                    for row in self.final_state.action_trace),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "formal_conclusion_allowed": False,
        }


def validate_public_case_v22(case: PublicTemporalCaseV2) -> None:
    if case.schema_version != PUBLIC_CASE_SCHEMA_V2:
        raise ValueError(f"public case must use {PUBLIC_CASE_SCHEMA_V2}")
    required = (
        case.case_id, case.model_id, case.question, case.start_page,
        case.cutoff_date, case.target_date,
    )
    if not all(str(value).strip() for value in required):
        raise ValueError("public live case fields must be non-empty")
    if case.cutoff_date >= case.target_date:
        raise ValueError("public cutoff must precede target")
    if re.search(r"\b[QP][1-9]\d*\b", case.question):
        raise ValueError("public question exposes a Wikidata identifier")


def load_public_manifest_v22(path: str | Path) -> list[PublicTemporalCaseV2]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        PUBLIC_MANIFEST_SCHEMA_V22
    ):
        raise ValueError(f"public manifest must use {PUBLIC_MANIFEST_SCHEMA_V22}")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("public live manifest needs cases")
    cases = []
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ValueError("public live case must be an object")
        case = PublicTemporalCaseV2(**raw)
        validate_public_case_v22(case)
        cases.append(case)
    if len({row.case_id for row in cases}) != len(cases):
        raise ValueError("duplicate public live case ID")
    return cases


def initial_live_state_v22(
    case: PublicTemporalCaseV2, backend: Any,
) -> LiveBeamStateV22:
    validate_public_case_v22(case)
    page = backend.fetch_page(case.start_page, as_of=case.cutoff_date)
    evidence = _page_evidence(page)
    return LiveBeamStateV22(
        current_page=page.title,
        current_revision_id=page.revision_id,
        current_revision_timestamp=page.timestamp,
        snapshot_as_of=case.cutoff_date,
        reasoning_summary="",
        extracted_entities=(),
        collected_evidence=(evidence,),
        retrieved_link_actions=(),
        retrieved_revision_actions=(),
        link_query_started=False,
        link_next_cursor=None,
        links_exhausted=False,
        revision_query_started=False,
        revision_next_cursor=None,
        revisions_exhausted=False,
        environment_queries_used=0,
        visited_nodes=((_fold(page.title), page.revision_id),),
        action_trace=(),
        cumulative_score=0.0,
        finished=False,
        submitted=None,
    )


def _snapshot_for_state(state: LiveBeamStateV22, backend: Any) -> Any:
    page = backend.fetch_revision(state.current_revision_id)
    if _fold(page.title) != _fold(state.current_page):
        raise WikipediaError("serialized live state revision changed page")
    return page


class _EnvironmentManifestCacheV22:
    def __init__(
        self, environment: TemporalWikipediaEnvironmentV2,
        case: PublicTemporalCaseV2,
        config: LiveSearchConfigV22,
    ):
        self.environment = environment
        self.case = case
        self.config = config
        self._values: dict[tuple[str, int], EnvironmentNodeManifestV22] = {}

    def get(self, snapshot: Any) -> EnvironmentNodeManifestV22:
        key = (_fold(snapshot.title), snapshot.revision_id)
        cached = self._values.get(key)
        if cached is not None:
            return cached
        link_actions: list[EnvironmentActionV2] = []
        link_cursor: str | None = None
        link_pages = 0
        while True:
            page = self.environment.list_links(
                snapshot, cursor=link_cursor,
                page_size=self.config.link_page_size,
            )
            link_actions.extend(page.items)
            link_pages += 1
            link_cursor = page.next_cursor
            if link_cursor is None:
                break
        revision_actions, revision_meta = self.environment.enumerate_revisions(
            snapshot,
            from_date=self.case.cutoff_date,
            to_date=self.case.target_date,
            page_size=self.config.revision_page_size,
        )
        actions = tuple([*link_actions, *revision_actions])
        payload = [row.to_dict() for row in actions]
        digest = _canonical_hash(payload)
        manifest = EnvironmentNodeManifestV22(
            manifest_id="environment_node_" + _canonical_hash([
                snapshot.title, snapshot.revision_id,
                self.case.cutoff_date, self.case.target_date, digest,
            ])[:24],
            page_title=snapshot.title,
            revision_id=snapshot.revision_id,
            time_window=(self.case.cutoff_date, self.case.target_date),
            actions=actions,
            action_count=len(actions),
            actions_sha256=digest,
            link_count=len(link_actions),
            revision_count=len(revision_actions),
            link_pages=link_pages,
            revision_pages=int(revision_meta["pages"]),
        )
        self._values[key] = manifest
        return manifest

    def values(self) -> tuple[EnvironmentNodeManifestV22, ...]:
        return tuple(sorted(
            self._values.values(), key=lambda row: (row.page_title.casefold(), row.revision_id),
        ))


def _merge_actions(
    previous: tuple[EnvironmentActionV2, ...],
    additions: tuple[EnvironmentActionV2, ...],
) -> tuple[EnvironmentActionV2, ...]:
    values = {row.action_id: row for row in previous}
    for row in additions:
        values.setdefault(row.action_id, row)
    return tuple(values.values())


def _merge_entities(
    previous: tuple[str, ...], additions: tuple[str, ...],
) -> tuple[str, ...]:
    values = {value.strip() for value in (*previous, *additions) if value.strip()}
    return tuple(sorted(values, key=str.casefold))


def _submission_public_gate(
    submission: StructuredSubmissionV2,
    evidence: tuple[dict[str, Any], ...],
) -> tuple[bool, str]:
    answer = " ".join(submission.answer.split())
    if not answer:
        return False, "empty_answer"
    if len(answer.split()) > 8:
        return False, "answer_over_8_words"
    if not submission.critical_claims:
        return False, "no_critical_claims"
    by_id = {str(page["evidence_id"]): page for page in evidence}
    claims = [*submission.critical_claims, submission.tail_claim]
    if any(not claim.subject or not claim.relation or not claim.object for claim in claims):
        return False, "empty_claim_field"
    cited = [value for claim in claims for value in claim.supporting_evidence_ids]
    if not cited or any(value not in by_id for value in cited):
        return False, "evidence_not_owned_by_trajectory"
    tail_pages = [
        by_id[value] for value in submission.tail_claim.supporting_evidence_ids
        if value in by_id
    ]
    if not any(_fold(answer) in _fold(page.get("content")) for page in tail_pages):
        return False, "answer_not_literal_in_tail_evidence"
    return True, "public_evidence_ownership_and_literal_gate_passed"


def _solver_retrieved_actions(
    state: LiveBeamStateV22, case: PublicTemporalCaseV2,
    config: LiveSearchConfigV22,
    submission: StructuredSubmissionV2 | None,
) -> tuple[list[EnvironmentActionV2], dict[str, Any]]:
    transitions = [
        row for row in (*state.retrieved_link_actions, *state.retrieved_revision_actions)
        if not (
            row.kind in {"FOLLOW_LINK", "SWITCH_SNAPSHOT"}
            and _prospective_node_key(state, row) in state.visited_nodes
        )
    ]
    controls: list[EnvironmentActionV2] = []
    if state.environment_queries_used < config.max_environment_queries_per_node:
        initial_link, initial_revision = initial_environment_queries_v2(
            link_page_size=config.link_page_size,
            revision_page_size=config.revision_page_size,
            from_date=case.cutoff_date,
            to_date=case.target_date,
        )
        if not state.link_query_started:
            controls.append(initial_link)
        elif not state.links_exhausted and state.link_next_cursor is not None:
            controls.append(EnvironmentActionV2(
                "LIST_LINKS",
                {"cursor": state.link_next_cursor, "page_size": config.link_page_size},
                f"List next hyperlink page from cursor {state.link_next_cursor}",
            ))
        if not state.revision_query_started:
            controls.append(initial_revision)
        elif not state.revisions_exhausted and state.revision_next_cursor is not None:
            controls.append(EnvironmentActionV2(
                "LIST_REVISIONS",
                {
                    "cursor": state.revision_next_cursor,
                    "page_size": config.revision_page_size,
                    "time_window": [case.cutoff_date, case.target_date],
                },
                f"List next revision page from cursor {state.revision_next_cursor}",
            ))
    submit_action = None
    submit_gate = "no_submission_proposed"
    if submission is not None:
        supported, submit_gate = _submission_public_gate(
            submission, state.collected_evidence,
        )
        if supported:
            submit_action = EnvironmentActionV2(
                "SUBMIT_ANSWER",
                submission.to_dict(),
                f'Submit structured answer "{submission.answer}"',
            )
            controls.append(submit_action)
    retrieved = [*transitions, *controls]
    reserved = controls[:config.dense_action_limit]
    kept_transitions = transitions[:max(0, config.dense_action_limit - len(reserved))]
    compacted = [*kept_transitions, *reserved]
    return compacted, {
        "schema_version": "temporal-solver-action-funnel-v2.2",
        "policy": "retrieval_order_first_n_reserving_control_and_submit_v2.2",
        "dense_limit": config.dense_action_limit,
        "solver_retrieved_actions": [row.to_dict() for row in retrieved],
        "compacted_ranker_actions": [row.to_dict() for row in compacted],
        "structured_submission_gate": submit_gate,
        "structured_submit_action_id": submit_action.action_id if submit_action else None,
    }


def _prospective_node_key(
    state: LiveBeamStateV22, action: EnvironmentActionV2,
) -> tuple[str, int]:
    if action.kind == "SWITCH_SNAPSHOT":
        return (_fold(state.current_page), int(action.params["revision_id"]))
    if action.kind == "FOLLOW_LINK":
        # The exact target revision is unknown until fetch; title-only sentinel
        # prevents no action and is not used for final repeated-node checking.
        return (_fold(action.params.get("page_title")), -1)
    return state.node_key


def _normalize_scores(
    actions: list[EnvironmentActionV2], output: LiveRankOutputV22,
) -> list[dict[str, Any]]:
    expected = {row.action_id for row in actions}
    if set(output.scores) != expected:
        raise RankerContractError("live score coverage mismatch")
    raw = {key: float(value) for key, value in output.scores.items()}
    if output.score_kind == "length_normalized_conditional_logprob":
        normalized_scores = raw
    else:
        maximum = max(raw.values(), default=0.0)
        denominator = sum(math.exp(value - maximum) for value in raw.values())
        normalized_scores = {
            key: value - maximum - math.log(denominator)
            for key, value in raw.items()
        }
    return [{
        **action.to_dict(),
        "raw_ranker_score": raw[action.action_id],
        "action_score": normalized_scores[action.action_id],
        "score_kind": output.score_kind,
    } for action in actions]


def _execute_action(
    *, state: LiveBeamStateV22, action: EnvironmentActionV2,
    action_score: float, ranker_output: LiveRankOutputV22,
    environment: TemporalWikipediaEnvironmentV2, backend: Any,
    case: PublicTemporalCaseV2,
) -> tuple[LiveBeamStateV22, dict[str, Any]]:
    snapshot = _snapshot_for_state(state, backend)
    next_page = snapshot
    evidence = state.collected_evidence
    links = state.retrieved_link_actions
    revisions = state.retrieved_revision_actions
    link_started = state.link_query_started
    link_cursor = state.link_next_cursor
    links_exhausted = state.links_exhausted
    revision_started = state.revision_query_started
    revision_cursor = state.revision_next_cursor
    revisions_exhausted = state.revisions_exhausted
    queries_used = state.environment_queries_used
    snapshot_as_of = state.snapshot_as_of
    finished = False
    submitted = None
    stop_reason = ""
    error = ""
    validity: dict[str, bool | None] = {
        "hyperlink_valid": None,
        "revision_valid": None,
        "environment_query_valid": None,
        "structured_submission_public_gate_passed": None,
    }
    try:
        if action.kind in {"LIST_LINKS", "LIST_REVISIONS"}:
            page = execute_environment_query_v2(environment, snapshot, action)
            if page.source_node != (snapshot.title, snapshot.revision_id):
                raise ValueError("environment query returned a different source node")
            queries_used += 1
            validity["environment_query_valid"] = True
            if action.kind == "LIST_LINKS":
                links = _merge_actions(links, page.items)
                link_started = True
                link_cursor = page.next_cursor
                links_exhausted = page.next_cursor is None
            else:
                revisions = _merge_actions(revisions, page.items)
                revision_started = True
                revision_cursor = page.next_cursor
                revisions_exhausted = page.next_cursor is None
        elif action.kind == "FOLLOW_LINK":
            if action.action_id not in {row.action_id for row in links}:
                raise ValueError("FOLLOW_LINK was not retrieved by the solver")
            if not environment.verify_action(snapshot, action):
                raise ValueError("FOLLOW_LINK is absent from the rendered revision")
            next_page = backend.fetch_page(
                str(action.params["page_title"]), as_of=state.snapshot_as_of,
            )
            evidence = (*evidence, _page_evidence(next_page))
            links = ()
            revisions = ()
            link_started = False
            link_cursor = None
            links_exhausted = False
            revision_started = False
            revision_cursor = None
            revisions_exhausted = False
            queries_used = 0
            validity["hyperlink_valid"] = True
        elif action.kind == "SWITCH_SNAPSHOT":
            if action.action_id not in {row.action_id for row in revisions}:
                raise ValueError("SWITCH_SNAPSHOT was not retrieved by the solver")
            if not environment.verify_action(snapshot, action):
                raise ValueError("SWITCH_SNAPSHOT changed page")
            next_page = backend.fetch_revision(int(action.params["revision_id"]))
            evidence = (*evidence, _page_evidence(next_page))
            snapshot_as_of = next_page.timestamp
            links = ()
            revisions = ()
            link_started = False
            link_cursor = None
            links_exhausted = False
            revision_started = False
            revision_cursor = None
            revisions_exhausted = False
            queries_used = 0
            validity["revision_valid"] = True
        elif action.kind == "SUBMIT_ANSWER":
            candidate = structured_submission_from_dict(action.params)
            supported, reason = _submission_public_gate(candidate, evidence)
            if not supported:
                raise ValueError(f"structured submission public gate failed: {reason}")
            submitted = candidate.to_dict()
            finished = True
            stop_reason = "submit_answer"
            validity["structured_submission_public_gate_passed"] = True
        else:
            raise ValueError(f"unsupported live action {action.kind}")
    except (KeyError, TypeError, ValueError, WikipediaError) as exc:
        error = str(exc)
        finished = True
        stop_reason = "action_error"
    origin = state.node_key
    node = (_fold(next_page.title), next_page.revision_id)
    visited = state.visited_nodes if node in state.visited_nodes else (*state.visited_nodes, node)
    trace = {
        "index": len(state.action_trace) + 1,
        "action": action.to_dict(),
        "action_score": action_score,
        "from_node": list(origin),
        "to_node": list(node),
        "result": "error" if error else "ok",
        "error": error,
        **validity,
    }
    child = LiveBeamStateV22(
        current_page=next_page.title,
        current_revision_id=next_page.revision_id,
        current_revision_timestamp=next_page.timestamp,
        snapshot_as_of=snapshot_as_of,
        reasoning_summary=ranker_output.reasoning_summary[:2000],
        extracted_entities=_merge_entities(
            state.extracted_entities, ranker_output.extracted_entities,
        ),
        collected_evidence=evidence,
        retrieved_link_actions=links,
        retrieved_revision_actions=revisions,
        link_query_started=link_started,
        link_next_cursor=link_cursor,
        links_exhausted=links_exhausted,
        revision_query_started=revision_started,
        revision_next_cursor=revision_cursor,
        revisions_exhausted=revisions_exhausted,
        environment_queries_used=queries_used,
        visited_nodes=visited,
        action_trace=(*state.action_trace, trace),
        cumulative_score=state.cumulative_score + action_score,
        finished=finished,
        submitted=submitted,
        stop_reason=stop_reason,
        error=error,
    )
    return child, trace


def _stable_order(state: LiveBeamStateV22, seed: int) -> tuple[float, str]:
    return (-state.cumulative_score, _canonical_hash([seed, state.state_id]))


def run_live_temporal_search_v22(
    *, public_case: PublicTemporalCaseV2, backend: Any,
    environment: TemporalWikipediaEnvironmentV2,
    ranker: LiveActionRankerV22,
    submission_proposer: LiveSubmissionProposerV22,
    config: LiveSearchConfigV22,
) -> LiveSearchResultV22:
    """Run public-only search. No private evaluation object can enter this API."""
    state = initial_live_state_v22(public_case, backend)
    frontier = [state]
    manifests = _EnvironmentManifestCacheV22(environment, public_case, config)
    audit_steps = []
    expansions = 0
    repeated = 0
    iteration = 0
    stop_reason = "exhausted_search"
    while frontier and expansions < config.max_expansions:
        iteration += 1
        proposals: list[tuple[LiveBeamStateV22, int, int]] = []
        iteration_audits: list[dict[str, Any]] = []
        for state in sorted(frontier, key=lambda row: _stable_order(row, config.seed)):
            if state.finished:
                proposals.append((state, -1, -1))
                continue
            manifest = None
            try:
                snapshot = _snapshot_for_state(state, backend)
                manifest = manifests.get(snapshot)
                submission, proposal_raw = submission_proposer.propose(
                    public_case, list(state.collected_evidence), seed=config.seed,
                )
                actions, funnel = _solver_retrieved_actions(
                    state, public_case, config, submission,
                )
                if not actions:
                    proposals.append((replace(
                        state, finished=True, stop_reason="exhausted_no_legal_progress",
                    ), -1, -1))
                    continue
                output = ranker.rank(
                    public_case, state, actions, seed=config.seed,
                )
                scored = sorted(
                    _normalize_scores(actions, output),
                    key=lambda row: (
                        -float(row["action_score"]),
                        _canonical_hash([config.seed, state.state_id, row["action_id"]]),
                    ),
                )
                funnel.update({
                    "parent_page": state.current_page,
                    "parent_revision_id": state.current_revision_id,
                    "environment_legal_actions": [],
                    "environment_legal_action_count": manifest.action_count,
                    "environment_legal_actions_sha256": manifest.actions_sha256,
                    "environment_legal_actions_inline": False,
                    "environment_legal_actions_artifact_reference": manifest.manifest_id,
                    "ranker_scores": output.scores,
                    "ranker_contract_valid": True,
                    "expanded_actions": [],
                })
                audit: dict[str, Any] = {
                    "schema_version": LIVE_TRAJECTORY_SCHEMA_V22,
                    "iteration": iteration,
                    "parent_state": state.to_dict(),
                    "visible_evidence_ids": [
                        page["evidence_id"] for page in state.collected_evidence
                    ],
                    "action_funnel": funnel,
                    "submission_proposal": {
                        "submission": submission.to_dict(),
                        "proposer": proposal_raw,
                    },
                    "ranker": {
                        "name": ranker.ranker_name,
                        "score_kind": output.score_kind,
                        "reasoning_summary": output.reasoning_summary,
                        "extracted_entities": list(output.extracted_entities),
                        "evidence_notes": list(output.evidence_notes),
                        "raw": output.raw,
                    },
                    "candidate_actions": [],
                    "selected_actions": [],
                    "retained_actions": [],
                }
                expanded_for_parent = 0
                for scored_action in scored:
                    candidate = dict(scored_action)
                    candidate.update({
                        "expanded": False,
                        "retained": False,
                        "pruning_reason": "",
                    })
                    action = next(
                        row for row in actions if row.action_id == candidate["action_id"]
                    )
                    if expanded_for_parent >= config.max_actions_per_state:
                        candidate["pruning_reason"] = "local_expansion_cap"
                    elif expansions >= config.max_expansions:
                        candidate["pruning_reason"] = "max_expansions"
                    else:
                        child, transition = _execute_action(
                            state=state,
                            action=action,
                            action_score=float(candidate["action_score"]),
                            ranker_output=output,
                            environment=environment,
                            backend=backend,
                            case=public_case,
                        )
                        expansions += 1
                        expanded_for_parent += 1
                        candidate["expanded"] = True
                        candidate["resulting_state"] = child.to_dict()
                        candidate["transition"] = transition
                        audit["selected_actions"].append(candidate["action_id"])
                        funnel["expanded_actions"].append(candidate["action_id"])
                        revisited = (
                            action.kind in {"FOLLOW_LINK", "SWITCH_SNAPSHOT"}
                            and child.node_key in state.visited_nodes
                        )
                        if revisited:
                            candidate["pruning_reason"] = "repeated_path_node"
                            repeated += 1
                        else:
                            proposals.append((
                                child, len(iteration_audits),
                                len(audit["candidate_actions"]),
                            ))
                    audit["candidate_actions"].append(candidate)
                iteration_audits.append(audit)
            except RankerContractError as exc:
                terminal = replace(
                    state,
                    finished=True,
                    stop_reason="ranker_contract_failure",
                    error=str(exc),
                )
                proposals.append((terminal, -1, -1))
                iteration_audits.append({
                    "schema_version": LIVE_TRAJECTORY_SCHEMA_V22,
                    "iteration": iteration,
                    "parent_state": state.to_dict(),
                    "candidate_actions": [],
                    "selected_actions": [],
                    "retained_actions": [],
                    "pruning_reason": "RANKER_CONTRACT_FAILURE",
                    "error": str(exc),
                    "environment_manifest_reference": (
                        manifest.manifest_id if manifest else None
                    ),
                })
            except (KeyError, TypeError, ValueError, WikipediaError) as exc:
                terminal = replace(
                    state,
                    finished=True,
                    stop_reason="runner_or_environment_error",
                    error=str(exc),
                )
                proposals.append((terminal, -1, -1))
                iteration_audits.append({
                    "schema_version": LIVE_TRAJECTORY_SCHEMA_V22,
                    "iteration": iteration,
                    "parent_state": state.to_dict(),
                    "candidate_actions": [],
                    "selected_actions": [],
                    "retained_actions": [],
                    "pruning_reason": "runner_or_environment_error",
                    "error": str(exc),
                    "environment_manifest_reference": (
                        manifest.manifest_id if manifest else None
                    ),
                })

        best: dict[tuple[Any, ...], tuple[LiveBeamStateV22, int, int]] = {}
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
                    candidate = iteration_audits[audit_index]["candidate_actions"][
                        candidate_index
                    ]
                    candidate["pruning_reason"] = "duplicate_state_lower_score"
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
        if frontier and all(row.finished for row in frontier):
            stop_reason = (
                "all_retained_error" if all(row.error for row in frontier)
                else "all_retained_finished"
            )
            break
    if expansions >= config.max_expansions and not all(
        row.finished for row in frontier
    ):
        stop_reason = "max_expansions"
    elif not frontier:
        stop_reason = "exhausted_search"
    submitted = [row for row in frontier if row.finished and row.submitted]
    pool = submitted or frontier
    if not pool:
        raise RuntimeError("live search produced no retained state")
    final = sorted(pool, key=lambda row: _stable_order(row, config.seed))[0]
    return LiveSearchResultV22(
        public_case=public_case,
        config=config,
        ranker_name=ranker.ranker_name,
        final_state=final,
        retained_states=tuple(frontier),
        audit_steps=tuple(audit_steps),
        environment_manifests=manifests.values(),
        expansions=expansions,
        repeated_state_count=repeated,
        stop_reason=stop_reason,
    )


class _ExclusiveJsonlWriterV22:
    def __init__(self, path: str | Path):
        assert_new_output_path(str(path))
        self.path = Path(path)
        self.handle = self.path.open("x", encoding="utf-8")

    def write(self, **fields: Any) -> None:
        row = {
            "schema_version": LIVE_RUN_SCHEMA_V22,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        self.handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Public-only live temporal runner v2.2")
    parser.add_argument("--public-cases", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-expansions", type=int, default=40)
    parser.add_argument("--max-actions-per-state", type=int, default=4)
    parser.add_argument("--link-page-size", type=int, default=50)
    parser.add_argument("--revision-page-size", type=int, default=50)
    parser.add_argument("--max-environment-queries-per-node", type=int, default=4)
    parser.add_argument("--dense-action-limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--wikipedia-cache", default="live_v22_wikipedia.db")
    parser.add_argument("--ranker-cache", default="live_v22_ranker.db")
    parser.add_argument("--submission-cache", default="live_v22_submission.db")
    parser.add_argument("--request-interval", type=float, default=0.7)
    parser.add_argument("--api-call-budget", type=int, default=2000)
    parser.add_argument("--wikipedia-offline-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is required for live API ranking")
    cases = load_public_manifest_v22(args.public_cases)
    if any(case.model_id != args.model for case in cases):
        parser.error("every public case must be bound to --model")
    config = LiveSearchConfigV22(
        beam_width=args.beam_width,
        max_expansions=args.max_expansions,
        max_actions_per_state=args.max_actions_per_state,
        link_page_size=args.link_page_size,
        revision_page_size=args.revision_page_size,
        max_environment_queries_per_node=args.max_environment_queries_per_node,
        dense_action_limit=args.dense_action_limit,
        seed=args.seed,
    )
    backend = WikipediaPageBackend(
        cache_path=args.wikipedia_cache,
        offline_only=args.wikipedia_offline_only,
        min_request_interval=args.request_interval,
        max_api_calls=args.api_call_budget,
    )
    environment = TemporalWikipediaEnvironmentV2(backend)
    ranker = ApiLiveActionRankerV22(
        args.model,
        cache_path=args.ranker_cache,
        max_dense_actions=args.dense_action_limit,
    )
    proposer = StructuredSubmissionProposerV2(
        args.model, cache_path=args.submission_cache,
    )
    writer = _ExclusiveJsonlWriterV22(args.output)
    try:
        for case in cases:
            result = run_live_temporal_search_v22(
                public_case=case,
                backend=backend,
                environment=environment,
                ranker=ranker,
                submission_proposer=proposer,
                config=config,
            )
            for manifest in result.environment_manifests:
                writer.write(
                    slot="environment_node_manifest",
                    case_id=case.case_id,
                    manifest=manifest.to_dict(),
                )
            for step in result.audit_steps:
                writer.write(
                    slot="live_beam_expansion",
                    case_id=case.case_id,
                    step=step,
                )
            writer.write(
                slot="live_beam_summary",
                case_id=case.case_id,
                result=result.to_dict(),
                wikipedia_request_stats=backend.request_stats(),
                formal_conclusion_allowed=False,
            )
    finally:
        writer.close()
        proposer.close()
        ranker.close()
        backend.close()
    print(f"[done] live v2.2 output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
