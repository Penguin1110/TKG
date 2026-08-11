"""Independent, structured LLM judge for exposure and answer evaluation."""

from __future__ import annotations

import json
import re
from typing import Callable

from tkg.api.openrouter import call_model
from tkg.experiment.contracts import JudgeResult


ALLOWED_ANSWER_LABELS = {
    "stick_new", "stick_old", "hedge", "unsupported", "irrelevant", "unjudgeable"
}
ALLOWED_TEMPORAL_LABELS = {
    "correct_after", "old_snapshot_answer", "supported_other_time",
    "unsupported", "no_answer", "unjudgeable",
}


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


def evidence_text(pages: list[dict], max_chars: int = 28_000) -> str:
    chunks = []
    used = 0
    for page in pages:
        header = f"[PAGE {page['title']} | revision {page['revision_id']} | {page['timestamp']}]\n"
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
        "hypothesis. Return one JSON object and no prose. Evidence excerpts must be short."
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
        target_label = target_snapshot_as_of or "CURRENT"
        prompt = f"""Task: evaluate the final answer from a model that could browse multiple
Wikipedia snapshots.

Question: {question}
Target snapshot requested by the task: {target_label}
Tested model response: {response}
Answer aliases supported by the later target revision: {json.dumps(after_answers, ensure_ascii=False)}
Answer aliases supported by the earlier target revision: {json.dumps(before_answers, ensure_ascii=False)}

Page revisions actually shown to the tested model:
{evidence_text(pages)}

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
        return result


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
