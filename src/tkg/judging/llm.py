"""Independent, structured LLM judge for exposure and answer evaluation."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Callable

from tkg.api.openrouter import call_model
from tkg.experiment.contracts import JudgeResult


ALLOWED_ANSWER_LABELS = {
    "stick_new", "stick_old", "hedge", "unsupported", "irrelevant", "unjudgeable"
}
ALLOWED_TEMPORAL_LABELS = {
    "correct_after", "old_snapshot_answer", "supported_other_time",
    "correct_without_visible_support", "unsupported", "no_answer", "unjudgeable",
}
ALLOWED_PRIOR_KNOWLEDGE_LABELS = {
    "known", "unknown", "wrong", "unjudgeable",
}


def _fold_text(value: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", value).casefold().split()
    )


def _fold_response_span(value: str) -> str:
    """Normalize harmless inline Markdown before checking judge provenance."""
    return _fold_text(re.sub(r"[*_`~]", "", value))


def _mentioned_aliases(text: str, aliases: list[str]) -> list[str]:
    """Return aliases mentioned as complete spans, without asserting correctness."""
    folded = _fold_text(text)
    matches = []
    for alias in aliases:
        candidate = _fold_text(alias)
        if not candidate:
            continue
        if folded == candidate or re.search(
            rf"(?<!\w){re.escape(candidate)}(?!\w)", folded
        ):
            matches.append(alias)
    return matches


def _direct_alias_matches(response: str, aliases: list[str]) -> list[str]:
    """Match only concise affirmative answers; complex language goes to the LLM.

    Presence alone is unsafe: ``not Alice`` and ``the page mentions Alice`` are
    mentions, not answers.  The deterministic path therefore accepts only an
    alias by itself or a small set of unambiguously affirmative wrappers.
    """
    folded = _fold_text(response).strip()
    matches = []
    for alias in aliases:
        candidate = _fold_text(alias)
        if not candidate:
            continue
        escaped = re.escape(candidate)
        patterns = (
            rf"{escaped}[.!]?",
            rf"(?:the\s+)?answer\s*(?:is|was|:)\s*[\"']?{escaped}[\"']?[.!]?",
            rf"(?:it|this|that)\s+(?:is|was)\s+[\"']?{escaped}[\"']?[.!]?",
            rf"my\s+(?:final\s+)?answer\s*(?:is|:)\s*[\"']?{escaped}[\"']?[.!]?",
        )
        if any(re.fullmatch(pattern, folded) for pattern in patterns):
            matches.append(alias)
    return matches


def _deterministic_result(decision: str, response: str, alias: str, gate: str) -> JudgeResult:
    return JudgeResult(
        decision=decision, confidence=1.0,
        reason=f"Deterministic accepted-alias match ({gate}).",
        evidence=alias, answer_extracted=alias,
        raw={"deterministic_gate": gate, "tested_response": response},
    )


def _require_extracted_answer_in_response(
    result: JudgeResult, response: str, positive_labels: set[str],
) -> JudgeResult:
    """Fail closed when a judge invents an answer absent from model output."""
    if result.decision not in positive_labels:
        return result
    extracted = _fold_response_span(result.answer_extracted)
    folded_response = _fold_response_span(response)
    if extracted and extracted in folded_response:
        return result
    result.raw = {
        **result.raw,
        "contract_violation": "answer_extracted_not_in_tested_response",
        "original_decision": result.decision,
    }
    result.decision = "unjudgeable"
    return result


def _require_label_alias_support(
    result: JudgeResult, label_aliases: dict[str, list[str]],
) -> JudgeResult:
    """Reject a stale/current alias label when that alias class is undefined."""
    aliases = label_aliases.get(result.decision)
    if aliases is None:
        return result
    # With configured aliases, the LLM fallback may legitimately recognize a
    # spelling or abbreviation not listed verbatim. The extracted-answer gate
    # still requires that value to occur in the tested response.
    if aliases:
        return result
    original = result.decision
    result.raw = {
        **result.raw,
        "original_decision": original,
        "contract_violation": "positive_label_without_configured_aliases",
    }
    result.decision = "unsupported"
    result.confidence = 1.0
    result.reason = (
        f"Judge returned {original!r}, but that label has no configured aliases."
    )
    return result


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start:end + 1])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    raise ValueError("judge 沒有回傳可解析的 JSON object")


def evidence_text(
    pages: list[dict],
    max_chars: int = 28_000,
    *,
    prioritize_as_of: str | None = None,
) -> str:
    ordered_pages = list(pages)
    if prioritize_as_of is not None:
        # Final-answer judging is about the requested target snapshot.  Put
        # those revisions first so earlier long pages cannot consume the
        # entire evidence budget before the answer-bearing revision appears.
        ordered_pages = [
            page for page in ordered_pages
            if str(page.get("as_of")) == prioritize_as_of
        ] + [
            page for page in ordered_pages
            if str(page.get("as_of")) != prioritize_as_of
        ]
    chunks = []
    used = 0
    for page in ordered_pages:
        as_of = page.get("as_of")
        as_of_label = f" | as_of {as_of}" if as_of is not None else ""
        header = (
            f"[PAGE {page['title']} | revision {page['revision_id']} | "
            f"{page['timestamp']}{as_of_label}]\n"
        )
        remaining = max_chars - used - len(header)
        if remaining <= 0:
            break
        content = page.get("content", "")[:remaining]
        chunks.append(header + content)
        used += len(header) + len(content)
    return "\n\n".join(chunks)


class LLMJudge:
    """A deterministic judge model, kept independent from the tested model."""

    SYSTEM = (
        "You are an independent evaluator for a temporal knowledge browsing experiment. "
        "Judge only from the supplied evidence and rubric. Do not reward the desired research "
        "hypothesis. Return one JSON object and no prose. Confidence must be a JSON number "
        "from 0 through 1, never a word or label. Evidence excerpts must be short."
    )

    def __init__(
        self,
        model: str,
        call_model_fn: Callable = call_model,
        temperature: float = 0.0,
        min_confidence: float = 0.8,
    ):
        self.model = model
        self.call_model_fn = call_model_fn
        self.temperature = temperature
        self.min_confidence = min_confidence

    def _call(self, prompt: str) -> dict:
        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": prompt},
        ]
        last_error = None
        for attempt in range(2):
            response = self.call_model_fn(
                self.model, messages, temperature=self.temperature,
            )
            try:
                return _extract_json(response)
            except ValueError as exc:
                last_error = exc
                if attempt == 0:
                    messages.extend([
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": (
                            "Your previous response was not one valid JSON object. Return only the "
                            "requested JSON object with all required keys."
                        )},
                    ])
        raise ValueError(f"judge JSON parse failed twice: {last_error}")

    @staticmethod
    def _result(raw: dict, default: str = "unjudgeable") -> JudgeResult:
        confidence = raw.get("confidence", 0.0)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.0
        return JudgeResult(
            decision=str(raw.get("decision", default)), confidence=confidence,
            reason=str(raw.get("reason", "")), evidence=str(raw.get("evidence", "")),
            answer_extracted=str(raw.get("answer_extracted", "")), raw=raw,
        )

    def _apply_confidence_gate(self, result: JudgeResult) -> JudgeResult:
        if result.confidence < self.min_confidence:
            result.raw = {**result.raw, "low_confidence_decision": result.decision,
                          "minimum_confidence": self.min_confidence}
            result.decision = "unjudgeable"
        return result

    def judge_visibility(
        self, question: str, new_answers: list[str], old_answers: list[str], pages: list[dict]
    ) -> JudgeResult:
        prompt = f"""Task: decide whether the TARGET/CORRECT answer is explicitly visible and unambiguous in
the pages actually shown to the tested model.

Question: {question}
Updated answer aliases: {json.dumps(new_answers, ensure_ascii=False)}
Stale answer aliases: {json.dumps(old_answers, ensure_ascii=False)}

Evidence:
{evidence_text(pages)}

Rules:
- decision=visible only if a reader can identify the target answer clearly (including its role
  when the question asks about a person holding a role).
- decision=ambiguous if updated and stale answers are both presented as current, or only an ID/
  vague description is shown instead of the answer.
- decision=not_visible if the update is absent.
- decision=unjudgeable if evidence is malformed or insufficient to inspect.
Return keys: decision, confidence, evidence, reason. Use a verbatim short evidence excerpt.
"""
        result = self._apply_confidence_gate(self._result(self._call(prompt)))
        if result.decision not in {"visible", "ambiguous", "not_visible", "unjudgeable"}:
            result.decision = "unjudgeable"
        return result

    def judge_answerability(
        self, question: str, accepted_answers: list[str], pages: list[dict]
    ) -> JudgeResult:
        prompt = f"""Task: decide whether the question can be answered from the pages actually shown,
using explicit statements or a short, uniquely determined inference. General world knowledge is
not evidence.

Question: {question}
Accepted answer aliases (for checking uniqueness, not evidence):
{json.dumps(accepted_answers, ensure_ascii=False)}

Evidence:
{evidence_text(pages)}

Also return support_type=explicit if the answer itself is directly stated,
support_type=inferred if it requires a short uniquely determined inference, otherwise none.
Return keys: decision, confidence, support_type, evidence, reason.
decision must be answerable, not_answerable, ambiguous, or unjudgeable.
"""
        result = self._apply_confidence_gate(self._result(self._call(prompt)))
        if result.decision not in {"answerable", "not_answerable", "ambiguous", "unjudgeable"}:
            result.decision = "unjudgeable"
        return result

    def judge_answer(
        self,
        question: str,
        response: str,
        new_answers: list[str],
        old_answers: list[str],
        pages: list[dict],
    ) -> JudgeResult:
        new_matches = _direct_alias_matches(response, new_answers)
        old_matches = _direct_alias_matches(response, old_answers)
        if new_matches and not old_matches:
            return _deterministic_result(
                "stick_new", response, new_matches[0], "exact_new_alias"
            )
        if old_matches and not new_matches:
            return _deterministic_result(
                "stick_old", response, old_matches[0], "exact_old_alias"
            )
        prompt = f"""Task: classify the tested model's answer.

Question: {question}
Tested model response: {response}
Updated/correct answer aliases: {json.dumps(new_answers, ensure_ascii=False)}
Stale answer aliases: {json.dumps(old_answers, ensure_ascii=False)}

Evidence shown to the tested model:
{evidence_text(pages)}

Labels:
- stick_new: clearly gives the updated/correct answer.
- stick_old: clearly gives a stale answer.
- hedge: explicitly declines or says it is uncertain without a concrete answer.
- unsupported: gives another concrete answer not supported by the evidence.
- irrelevant: does not address the question.
- unjudgeable: malformed or genuinely impossible to classify.

The evidence section records what the tested model saw. It may be empty for a prior-knowledge
probe; an empty evidence section does not prevent matching the response to the supplied aliases.

Return keys: decision, confidence, answer_extracted, evidence, reason.
"""
        result = self._apply_confidence_gate(self._result(self._call(prompt)))
        if result.decision not in ALLOWED_ANSWER_LABELS:
            result.decision = "unjudgeable"
        result = _require_extracted_answer_in_response(
            result, response, {"stick_new", "stick_old"},
        )
        return _require_label_alias_support(result, {
            "stick_new": new_answers,
            "stick_old": old_answers,
        })

    def judge_prior_knowledge(
        self,
        *,
        probe_role: str,
        question: str,
        response: str,
        answer_aliases: list[str],
        context_aliases: list[str] | None = None,
    ) -> JudgeResult:
        """Judge one factorized fact, never the names used as question context."""
        context_aliases = list(context_aliases or [])
        direct = _direct_alias_matches(response, answer_aliases)
        if direct:
            return _deterministic_result(
                "known", response, direct[0], "exact_probe_answer_alias",
            )
        prompt = f"""Task: classify prior knowledge for one isolated factual probe.

Probe role: {probe_role}
Exact probe question: {question}
Tested model response: {response}
Correct answer aliases for THIS probe only:
{json.dumps(answer_aliases, ensure_ascii=False)}
Known context/intermediate aliases that are NOT answers to this probe:
{json.dumps(context_aliases, ensure_ascii=False)}

Labels:
- known: clearly gives the correct answer to this exact probe.
- unknown: explicitly says it does not know the answer and gives no candidate answer.
- wrong: gives a concrete answer to this exact probe, but it is not the correct alias.
- unjudgeable: mixed, malformed, ambiguous, or impossible to classify confidently.

Critical rules:
- Judge only the answer requested by the exact probe.
- Names already present in the question are context, not candidate answers.
- For a composed probe, the response may correctly solve early steps before saying the
  final answer is unknown. Every supplied context/intermediate alias is explicitly not
  the final answer; never label the probe wrong merely because one appears.
- Do not label a response wrong merely because it repeats a person, office, or date from
  the question before saying that the requested answer is unknown.
- answer_extracted must be one plain string copied from the tested response. Use an empty
  string for unknown. Do not return an object, list, summary, or multiple hop answers.

Return keys: decision, confidence, answer_extracted, evidence, reason.
"""
        result = self._apply_confidence_gate(self._result(self._call(prompt)))
        if result.decision not in ALLOWED_PRIOR_KNOWLEDGE_LABELS:
            result.decision = "unjudgeable"
        result = _require_extracted_answer_in_response(
            result, response, {"known", "wrong"},
        )
        if result.decision == "known" and not _mentioned_aliases(
            result.answer_extracted, answer_aliases,
        ):
            result.raw = {
                **result.raw,
                "contract_violation": "known_answer_not_in_probe_aliases",
                "original_decision": "known",
            }
            result.decision = "unjudgeable"
        if result.decision == "wrong" and _mentioned_aliases(
            result.answer_extracted, context_aliases,
        ):
            result.raw = {
                **result.raw,
                "contract_violation": "wrong_answer_is_context_alias",
                "original_decision": "wrong",
            }
            result.decision = "unjudgeable"
        if result.decision == "unknown" and _mentioned_aliases(
            response, answer_aliases,
        ):
            result.raw = {
                **result.raw,
                "contract_violation": "unknown_response_mentions_correct_alias",
                "original_decision": "unknown",
            }
            result.decision = "unjudgeable"
        return result

    def judge_temporal_answer(
        self,
        question: str,
        response: str,
        after_answers: list[str],
        before_answers: list[str],
        pages: list[dict],
        *,
        target_snapshot_as_of: str | None = None,
    ) -> JudgeResult:
        if not response.strip():
            return JudgeResult(
                decision="no_answer",
                confidence=1.0,
                reason="The tested model submitted no answer.",
                evidence="",
                answer_extracted="",
                raw={"deterministic_gate": "blank_response"},
            )
        after_matches = _direct_alias_matches(response, after_answers)
        before_matches = _direct_alias_matches(response, before_answers)
        if after_matches and not before_matches:
            alias = after_matches[0]
            target_support = any(
                (target_snapshot_as_of is None
                 or page.get("as_of") == target_snapshot_as_of)
                and bool(_mentioned_aliases(str(page.get("content", "")), [alias]))
                for page in pages
            )
            return _deterministic_result(
                "correct_after" if target_support else "correct_without_visible_support",
                response, alias,
                "exact_new_alias_with_target_support"
                if target_support else "exact_new_alias_without_target_support",
            )
        if before_matches and not after_matches:
            return _deterministic_result(
                "old_snapshot_answer", response, before_matches[0], "exact_old_alias"
            )
        target_label = target_snapshot_as_of or "CURRENT"
        prompt = f"""Task: evaluate the final answer from a model that could browse multiple
Wikipedia snapshots.

Question: {question}
Target snapshot requested by the task: {target_label}
Tested model response: {response}
Answer aliases supported by the later target revision: {json.dumps(after_answers, ensure_ascii=False)}
Answer aliases supported by the earlier target revision: {json.dumps(before_answers, ensure_ascii=False)}

Page revisions actually shown to the tested model:
{evidence_text(pages, prioritize_as_of=target_snapshot_as_of)}

Labels:
- correct_after: clearly answers with the later target answer.
- old_snapshot_answer: clearly answers with the earlier target answer as if it were current.
- supported_other_time: gives a time-qualified answer that is supported by a shown revision but
  does not answer the requested target time.
- unsupported: gives a concrete answer unsupported by the shown evidence or aliases.
- no_answer: declines, stays vague, or supplies no answer.
- unjudgeable: malformed input or genuinely impossible to classify.

Judge the answer, not whether its browsing strategy was efficient. Return keys: decision,
confidence, answer_extracted, evidence, reason. Evidence must be a short excerpt.
"""
        result = self._apply_confidence_gate(self._result(self._call(prompt)))
        if result.decision not in ALLOWED_TEMPORAL_LABELS:
            result.decision = "unjudgeable"
        result = _require_extracted_answer_in_response(
            result, response,
            {"correct_after", "old_snapshot_answer", "supported_other_time"},
        )
        return _require_label_alias_support(result, {
            "correct_after": after_answers,
            "old_snapshot_answer": before_answers,
        })


def transition_label(previous: str | None, current: str) -> str:
    if previous is None:
        return "initial"
    if previous == "stick_new" and current == "stick_old":
        return "reversion"
    if previous == "stick_old" and current == "stick_new":
        return "recovery"
    if previous == "stick_new" and current == "stick_new":
        return "stable_new"
    if previous == "stick_old" and current == "stick_old":
        return "stable_old"
    return f"{previous}_to_{current}"
