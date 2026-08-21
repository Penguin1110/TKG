"""Dense action rankers for the append-only live v2.2 search runner."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from tkg.api.openrouter import call_model
from tkg.experiment.temporal_beam import RankerContractError
from tkg.experiment.temporal_beam_ranker import RankerCache, _extract_json
from tkg.experiment.temporal_environment_v2 import EnvironmentActionV2
from tkg.experiment.temporal_eval_schema_v2 import PublicTemporalCaseV2


LIVE_RANKER_CONTRACT_V22 = "open-world-live-dense-action-ranker-v2.2"


@dataclass(frozen=True)
class LiveRankOutputV22:
    scores: dict[str, float]
    score_kind: str
    reasoning_summary: str = ""
    extracted_entities: tuple[str, ...] = ()
    evidence_notes: tuple[str, ...] = ()
    raw: dict[str, Any] | None = None


class LiveActionRankerV22(Protocol):
    ranker_name: str

    def rank(
        self, public_case: PublicTemporalCaseV2, state: Any,
        actions: list[EnvironmentActionV2], *, seed: int,
    ) -> LiveRankOutputV22:
        ...


class ApiLiveActionRankerV22:
    """API fallback with exact dense IDs; still not decoding integration."""

    ranker_name = "api_live_dense_utility_fallback_not_decoding_v2.2"

    def __init__(
        self, model: str, *, cache_path: str | Path,
        call_model_fn: Callable = call_model, max_dense_actions: int = 30,
        max_contract_attempts: int = 2, max_evidence_chars: int = 20_000,
    ):
        if max_dense_actions <= 0:
            raise ValueError("max_dense_actions must be positive")
        if max_contract_attempts != 2:
            raise ValueError("live API ranker allows exactly one retry")
        self.model = model
        self.cache = RankerCache(cache_path)
        self.call_model_fn = call_model_fn
        self.max_dense_actions = max_dense_actions
        self.max_contract_attempts = max_contract_attempts
        self.max_evidence_chars = max_evidence_chars

    def close(self) -> None:
        self.cache.close()

    def _evidence_text(self, state: Any) -> str:
        remaining = self.max_evidence_chars
        records = []
        for page in reversed(state.collected_evidence):
            header = (
                f"[EVIDENCE_ID {page['evidence_id']} | PAGE {page['title']} | "
                f"revision {page['revision_id']} | {page['timestamp']}]\n"
            )
            content = str(page.get("content") or "")[:max(0, remaining - len(header))]
            if not content:
                break
            records.append(header + content)
            remaining -= len(header) + len(content)
            if remaining <= 0:
                break
        return "\n".join(reversed(records))

    def _prompt(
        self, public_case: PublicTemporalCaseV2, state: Any,
        actions: list[EnvironmentActionV2], seed: int,
    ) -> str:
        candidates = [row.to_dict() for row in actions]
        return f"""Rank every legal/retrieved action for an open-world temporal
Wikipedia graph search. The controller validates and executes actions. Use only the
public question and visible evidence. Do not invent actions, pages, revisions,
claims, dates, or answers. Return JSON only, without hidden chain-of-thought.

Question: {public_case.question}
Knowledge cutoff: {public_case.cutoff_date}
Target date: {public_case.target_date}
Current page: {state.current_page}
Current revision: {state.current_revision_id} ({state.current_revision_timestamp})
Concise state summary: {state.reasoning_summary or '(none)'}
Visible extracted entities: {json.dumps(state.extracted_entities, ensure_ascii=False)}
Tie seed: {seed}

Visible evidence:
{self._evidence_text(state)}

Candidate actions:
{json.dumps(candidates, ensure_ascii=False)}

Return exactly:
{{
  "reasoning_summary": "brief state summary",
  "extracted_entities": ["only entities visible above"],
  "evidence_notes": ["brief visible observations"],
  "action_utilities": [
    {{"action_id": "ACTION_ID", "utility": 0.0}}
  ]
}}

Score every listed action ID exactly once with a finite number from -100 to 100.
LIST_LINKS and LIST_REVISIONS retrieve more environment actions and consume budget.
FOLLOW_LINK and SWITCH_SNAPSHOT move in the graph. SUBMIT_ANSWER already passed a
public evidence-ownership/literal gate; rank it highly only when its structured
critical claims and tail answer the complete question. Utilities are API fallback
scores, not probabilities or decoding logits."""

    def rank(
        self, public_case: PublicTemporalCaseV2, state: Any,
        actions: list[EnvironmentActionV2], *, seed: int,
    ) -> LiveRankOutputV22:
        if len(actions) > self.max_dense_actions:
            raise RankerContractError(
                "live_ranker_candidate_count_exceeds_limit:"
                f"{len(actions)}>{self.max_dense_actions}"
            )
        expected = {row.action_id for row in actions}
        if len(expected) != len(actions):
            raise RankerContractError("duplicate_live_candidate_action_id")
        base_prompt = self._prompt(public_case, state, actions, seed)
        failures = []
        raw: dict[str, Any] = {}
        scores: dict[str, float] = {}
        for attempt in range(self.max_contract_attempts):
            corrective = ""
            if attempt:
                corrective = (
                    "\nCORRECTIVE RETRY: Return exactly every listed action ID once. "
                    "Do not omit, duplicate, or add IDs."
                )
            prompt = base_prompt + corrective
            cache_key = hashlib.sha256(json.dumps({
                "model": self.model,
                "contract": LIVE_RANKER_CONTRACT_V22,
                "attempt": attempt,
                "prompt": prompt,
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )).hexdigest()
            response = self.cache.get(cache_key)
            cache_hit = response is not None
            if response is None:
                response = self.call_model_fn(
                    self.model,
                    [
                        {"role": "system", "content": (
                            "Densely score every supplied action ID. JSON only."
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                self.cache.put(cache_key, response)
            try:
                raw = _extract_json(response)
                raw_scores = raw.get("action_utilities")
                if not isinstance(raw_scores, list):
                    raise RankerContractError("action_utilities is not a list")
                returned_ids = [
                    row.get("action_id") for row in raw_scores
                    if isinstance(row, dict)
                ]
                if len(returned_ids) != len(raw_scores):
                    raise RankerContractError("action utility row is not an object")
                if len(returned_ids) != len(set(returned_ids)):
                    raise RankerContractError("duplicate_live_returned_action_id")
                returned = set(returned_ids)
                if returned != expected:
                    raise RankerContractError(
                        "live_action_id_coverage_mismatch:"
                        f"missing={len(expected - returned)}:"
                        f"unexpected={len(returned - expected)}"
                    )
                score_by_id = {
                    str(row["action_id"]): row.get("utility") for row in raw_scores
                }
                parsed = {}
                for action_id in sorted(expected):
                    value = score_by_id[action_id]
                    if isinstance(value, bool):
                        raise RankerContractError("live utility is not numeric")
                    numeric = float(value)
                    if not math.isfinite(numeric) or not -100 <= numeric <= 100:
                        raise RankerContractError("live utility is invalid")
                    parsed[action_id] = numeric
                scores = parsed
            except (TypeError, ValueError, RankerContractError) as exc:
                failures.append({
                    "attempt": attempt + 1,
                    "cache_key": cache_key,
                    "cache_hit": cache_hit,
                    "error": str(exc),
                })
                continue
            break
        else:
            raise RankerContractError(
                "live_ranker_invalid_after_retry:" + json.dumps(
                    failures, ensure_ascii=False, sort_keys=True,
                )
            )
        entities = raw.get("extracted_entities")
        notes = raw.get("evidence_notes")
        if not isinstance(entities, list) or not all(isinstance(v, str) for v in entities):
            entities = []
        if not isinstance(notes, list) or not all(isinstance(v, str) for v in notes):
            notes = []
        return LiveRankOutputV22(
            scores=scores,
            score_kind="api_fallback_utility_softmax_log_score",
            reasoning_summary=" ".join(str(raw.get("reasoning_summary") or "").split()),
            extracted_entities=tuple(" ".join(value.split()) for value in entities),
            evidence_notes=tuple(" ".join(value.split()) for value in notes),
            raw={
                "contract": LIVE_RANKER_CONTRACT_V22,
                "cache_key": cache_key,
                "cache_hit": cache_hit,
                "contract_attempt": attempt + 1,
                "failures_before_success": failures,
                "expected_action_count": len(expected),
                "returned_action_count": len(scores),
                "action_id_coverage_complete": True,
                "raw_response": response,
                "parsed_response": raw,
            },
        )


class CallableLiveActionRankerV22:
    """Deterministic/open-weight adapter; score each complete supplied action."""

    ranker_name = "callable_live_action_ranker_v2.2"

    def __init__(
        self, score_fn: Callable[[PublicTemporalCaseV2, Any, EnvironmentActionV2], float],
        *, score_kind: str = "length_normalized_conditional_logprob",
    ):
        self.score_fn = score_fn
        self.score_kind = score_kind

    def rank(
        self, public_case: PublicTemporalCaseV2, state: Any,
        actions: list[EnvironmentActionV2], *, seed: int,
    ) -> LiveRankOutputV22:
        del seed
        if len({action.action_id for action in actions}) != len(actions):
            raise RankerContractError("duplicate_live_candidate_action_id")
        scores = {
            action.action_id: float(self.score_fn(public_case, state, action))
            for action in actions
        }
        if any(not math.isfinite(value) for value in scores.values()):
            raise RankerContractError("callable live ranker returned non-finite score")
        return LiveRankOutputV22(scores=scores, score_kind=self.score_kind)
