"""Action rankers for temporal graph-constrained beam search."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Callable

from tkg.api.openrouter import call_model
from tkg.experiment.temporal_beam import (
    AnswerProposal, RankerContractError, RankerOutput, TemporalAction, TemporalBeamState,
    TemporalSearchRequest, evidence_id,
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _extract_json(value: str) -> dict[str, Any]:
    text = value.strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    errors = []

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, item in pairs:
            if key in parsed:
                raise ValueError(f"duplicate JSON key: {key}")
            parsed[key] = item
        return parsed

    for candidate in candidates:
        try:
            parsed = json.loads(candidate, object_pairs_hook=reject_duplicate_keys)
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(str(exc))
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(
        "ranker did not return one valid JSON object: " + "; ".join(errors[:3])
    )


class RankerCache:
    """Content-addressed cache that freezes API fallback decisions."""

    def __init__(self, path: str | Path):
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS ranker_cache ("
            "cache_key TEXT PRIMARY KEY, response TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT response FROM ranker_cache WHERE cache_key=?", (key,),
        ).fetchone()
        return str(row[0]) if row else None

    def put(self, key: str, response: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO ranker_cache(cache_key,response) VALUES(?,?)",
            (key, response),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class ApiUtilityRanker:
    """API fallback: model-reported utilities, not decoding logits."""

    ranker_name = "api_dense_utility_ranker_fallback_not_decoding_integration_v2"

    def __init__(
        self,
        model: str,
        *,
        cache_path: str | Path,
        call_model_fn: Callable = call_model,
        max_evidence_chars: int = 20_000,
        max_dense_actions: int = 30,
        max_contract_attempts: int = 2,
    ):
        self.model = model
        self.call_model_fn = call_model_fn
        self.cache = RankerCache(cache_path)
        self.max_evidence_chars = max_evidence_chars
        self.max_dense_actions = max_dense_actions
        self.max_contract_attempts = max_contract_attempts
        if self.max_dense_actions <= 0:
            raise ValueError("max_dense_actions must be > 0")
        if self.max_contract_attempts != 2:
            raise ValueError("API ranker contract permits exactly one retry")

    def close(self) -> None:
        self.cache.close()

    def _evidence_text(self, state: TemporalBeamState) -> str:
        remaining = self.max_evidence_chars
        evidence = []
        for page in reversed(state.collected_evidence):
            header = (
                f"[EVIDENCE_ID {evidence_id(page)} | PAGE {page['title']} | "
                f"revision {page['revision_id']} | {page['timestamp']} | "
                f"as_of {page.get('as_of')}]\n"
            )
            content = str(page.get("content", ""))[:max(0, remaining - len(header))]
            if not content:
                break
            evidence.append(header + content)
            remaining -= len(header) + len(content)
            if remaining <= 0:
                break
        evidence.reverse()
        return "\n".join(evidence)

    def _prompt(
        self,
        request: TemporalSearchRequest,
        state: TemporalBeamState,
        actions: list[TemporalAction],
        seed: int,
    ) -> str:
        action_payload = [{
            "action_id": action.action_id,
            "kind": action.kind,
            "label": action.label,
            "params": action.params,
        } for action in actions]
        return f"""You rank legal actions for a temporal Wikipedia graph search.
The graph controller, not you, validates and executes every action. Do not invent an
action, page, revision, date, or answer. Use only the supplied visible evidence.
Return one JSON object and no prose. Do not provide hidden chain-of-thought.

Question: {request.question}
Starting cutoff: {request.cutoff_date}
Target answer date: {request.target_date}
Current page: {state.current_page}
Current revision: {state.current_revision_id} ({state.current_revision_date})
Current concise reasoning summary: {state.reasoning_summary or '(none)'}
Previously extracted visible entities: {json.dumps(state.extracted_entities, ensure_ascii=False)}
Deterministic tie seed: {seed}

Visible evidence:
{self._evidence_text(state)}

Legal candidate actions:
{json.dumps(action_payload, ensure_ascii=False)}

Required JSON schema:
{{
  "reasoning_summary": "brief state summary, not chain-of-thought",
  "extracted_entities": ["entities explicitly visible in evidence"],
  "evidence_notes": ["short factual excerpts or observations"],
  "action_utilities": {{"ACTION_ID": 0.0}}
}}
The question is a composed multi-hop task. Evidence that answers only a prefix is
not a final answer. SUBMIT_ANSWER, when present, already passed a literal visible-
evidence support gate; rank it highly only if it answers the entire composed question.
When a visible entity answers the current hop, prefer its FOLLOW_LINK action; on
that entity, use LIST_REVISIONS/SWITCH_SNAPSHOT when post-cutoff evidence is still
needed. Never infer that navigation is unnecessary merely because the current
revision predates the requested event.
Give every listed action_id one finite utility from -100 to 100. These are fallback
ranking utilities, not probabilities or logits."""

    def propose_answer(
        self,
        request: TemporalSearchRequest,
        state: TemporalBeamState,
        visible_evidence: list[dict[str, Any]],
        *,
        seed: int,
    ) -> AnswerProposal:
        del visible_evidence
        prompt = f"""Generate exactly one possible final answer from visible evidence.
This is an answer-candidate stage, separate from graph-action ranking. Use no outside
knowledge. If the complete composed question is not answered, return an empty answer.
The question may spell out intermediate steps. The answer field MUST contain only
the answer to the LAST requested step: one entity name or short noun phrase of at
most 8 words. Do not repeat intermediate answers, sentences, reasoning, dates,
explanations, labels such as "Answer:", or uncertainty phrases.

Question: {request.question}
Starting cutoff: {request.cutoff_date}
Target answer date: {request.target_date}
Current page: {state.current_page}
Deterministic tie seed: {seed}

Visible evidence:
{self._evidence_text(state)}

Return one JSON object only:
{{
  "answer": "concise final answer or empty string",
  "supporting_evidence_ids": ["EVIDENCE_ID containing the answer text"]
}}
The controller will independently reject any answer not literally present in a
claimed visible evidence record. This is not a correctness or gold-answer check.
Before returning, verify that answer contains at most 8 words and no sentence."""
        contract = self.ranker_name + ":answer_candidate_v2"
        cache_key = hashlib.sha256(_canonical({
            "model": self.model, "prompt": prompt, "contract": contract,
        }).encode("utf-8")).hexdigest()
        response = self.cache.get(cache_key)
        cache_hit = response is not None
        if response is None:
            response = self.call_model_fn(
                self.model,
                [
                    {"role": "system", "content": (
                        "Propose one evidence-grounded answer candidate. Return JSON only."
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            self.cache.put(cache_key, response)
        raw = _extract_json(response)
        cited = raw.get("supporting_evidence_ids")
        if not isinstance(cited, list) or not all(isinstance(v, str) for v in cited):
            cited = []
        return AnswerProposal(
            answer=" ".join(str(raw.get("answer", "")).split()),
            supporting_evidence_ids=tuple(cited),
            raw={"cache_key": cache_key, "cache_hit": cache_hit},
        )

    def rank(
        self,
        request: TemporalSearchRequest,
        state: TemporalBeamState,
        visible_evidence: list[dict[str, Any]],
        actions: list[TemporalAction],
        *,
        seed: int,
    ) -> RankerOutput:
        del visible_evidence
        if len(actions) > self.max_dense_actions:
            raise RankerContractError(
                "ranker_call_invalid:candidate_count_exceeds_dense_limit:"
                f"count={len(actions)}:limit={self.max_dense_actions}"
            )
        expected = {action.action_id for action in actions}
        if len(expected) != len(actions):
            raise RankerContractError(
                "ranker_call_invalid:duplicate_legal_candidate_action_id"
            )
        base_prompt = self._prompt(request, state, actions, seed)
        failures = []
        for attempt in range(self.max_contract_attempts):
            corrective = ""
            if attempt:
                corrective = f"""

CORRECTIVE RETRY: The previous response violated the dense scoring contract.
Return exactly {len(expected)} distinct action_utilities keys, one for every listed
action_id and no others. Omitting an action is invalid; do not return top-k only."""
            prompt = base_prompt + corrective
            cache_key = hashlib.sha256(_canonical({
                "model": self.model,
                "prompt": prompt,
                "contract": self.ranker_name,
                "contract_attempt": attempt,
            }).encode("utf-8")).hexdigest()
            response = self.cache.get(cache_key)
            cache_hit = response is not None
            if response is None:
                response = self.call_model_fn(
                    self.model,
                    [
                        {"role": "system", "content": (
                            "Densely score every supplied action ID. Return JSON only."
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                self.cache.put(cache_key, response)
            try:
                raw = _extract_json(response)
                raw_scores = raw.get("action_utilities")
                if not isinstance(raw_scores, dict):
                    raise RankerContractError("action_utilities is not an object")
                returned = set(raw_scores)
                missing = sorted(expected - returned)
                extras = sorted(returned - expected)
                if missing or extras:
                    raise RankerContractError(
                        f"action_id_coverage_mismatch:missing={len(missing)}:"
                        f"unexpected={len(extras)}"
                    )
                scores: dict[str, float] = {}
                for action_id in sorted(expected):
                    value = raw_scores[action_id]
                    if isinstance(value, bool):
                        raise RankerContractError(
                            f"ranker utility for {action_id} is not numeric"
                        )
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError) as exc:
                        raise RankerContractError(
                            f"ranker utility for {action_id} is not numeric"
                        ) from exc
                    if not math.isfinite(numeric) or not -100 <= numeric <= 100:
                        raise RankerContractError(
                            f"ranker utility for {action_id} is out of range"
                        )
                    scores[action_id] = numeric
            except (ValueError, RankerContractError) as exc:
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
                "ranker_call_invalid_after_retry:" + _canonical(failures)
            )
        entities = raw.get("extracted_entities")
        notes = raw.get("evidence_notes")
        if not isinstance(entities, list) or not all(isinstance(v, str) for v in entities):
            entities = []
        if not isinstance(notes, list) or not all(isinstance(v, str) for v in notes):
            notes = []
        return RankerOutput(
            scores=scores,
            score_kind="api_fallback_utility_softmax_log_score",
            reasoning_summary=" ".join(str(raw.get("reasoning_summary", "")).split()),
            extracted_entities=tuple(" ".join(value.split()) for value in entities),
            evidence_notes=tuple(" ".join(value.split()) for value in notes),
            raw={
                "cache_key": cache_key,
                "cache_hit": cache_hit,
                "contract_attempt": attempt + 1,
                "contract_failures_before_success": failures,
                "expected_action_count": len(expected),
                "returned_action_count": len(scores),
                "action_id_coverage_complete": True,
            },
        )


class CallableConditionalLogProbRanker:
    """Adapter for open-weight backends that score complete candidate actions."""

    ranker_name = "length_normalized_conditional_logprob_ranker"

    def __init__(
        self,
        score_fn: Callable[[TemporalSearchRequest, TemporalBeamState, TemporalAction], float],
        answer_fn: Callable[
            [TemporalSearchRequest, TemporalBeamState], AnswerProposal
        ] | None = None,
    ):
        self.score_fn = score_fn
        self.answer_fn = answer_fn

    def propose_answer(
        self,
        request: TemporalSearchRequest,
        state: TemporalBeamState,
        visible_evidence: list[dict[str, Any]],
        *,
        seed: int,
    ) -> AnswerProposal:
        del visible_evidence, seed
        if self.answer_fn is None:
            return AnswerProposal(
                answer="", support_reason="no_open_weight_answer_generator_configured",
            )
        return self.answer_fn(request, state)

    def rank(
        self,
        request: TemporalSearchRequest,
        state: TemporalBeamState,
        visible_evidence: list[dict[str, Any]],
        actions: list[TemporalAction],
        *,
        seed: int,
    ) -> RankerOutput:
        del visible_evidence, seed
        scores = {action.action_id: float(self.score_fn(request, state, action))
                  for action in actions}
        if any(not math.isfinite(value) for value in scores.values()):
            raise ValueError("conditional log-probability ranker returned non-finite score")
        return RankerOutput(
            scores=scores,
            score_kind="length_normalized_conditional_logprob",
        )
