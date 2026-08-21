"""Evidence-conditioned compact answer generation for open-weight v2.4."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from tkg.experiment.compact_submission_v24 import (
    CompactSubmissionV24, compact_submission_from_dict_v24,
    validate_compact_submission_public_v24,
)
from tkg.experiment.temporal_eval_schema_v2 import PublicTemporalCaseV2


ANSWER_GENERATOR_PROTOCOL_V24 = "evidence-conditioned-compact-answer-v2.4"
ANSWER_SYSTEM_PROMPT_V24 = (
    "You extract answers only from visible Wikipedia evidence. Never guess, and "
    "never use hidden knowledge. Return exactly one JSON object."
)


class TextGenerationBackendV24(Protocol):
    backend_name: str

    def generate_text(
        self, prompt: str, *, max_new_tokens: int = 192,
        system_prompt: str | None = None,
    ) -> str:
        ...


@dataclass(frozen=True)
class CompactPayloadProposalV24:
    submission: CompactSubmissionV24 | None
    status: str
    raw_response: str
    prompt: str
    error: str | None = None

    def audit_dict(self) -> dict[str, Any]:
        return {
            "protocol": ANSWER_GENERATOR_PROTOCOL_V24,
            "status": self.status, "raw_response": self.raw_response,
            "prompt": self.prompt, "error": self.error,
        }


def answer_prompt_v24(case: PublicTemporalCaseV2, state: Any) -> str:
    evidence = [{
        "evidence_id": str(page["evidence_id"]),
        "title": str(page["title"]),
        "revision_id": page.get("revision_id"),
        "timestamp": page.get("timestamp"),
        "content": str(page.get("content") or "")[:12_000],
    } for page in state.collected_evidence[-8:]]
    return (
        "Answer the question only if the visible evidence contains BOTH: "
        "(1) the post-cutoff bridge identifying the target entity and event, and "
        "(2) evidence for the requested final relation. Otherwise return "
        '{"abstain":true}. If complete, return exactly: '
        '{"schema_version":"compact-temporal-evidence-submission-v2.4",'
        '"answer":"1 to 8 word noun phrase",'
        '"bridge_evidence_ids":["ev_id"],"tail_evidence_ids":["ev_id"]}. '
        "Cite only IDs shown below.\n" + json.dumps({
            "question": case.question, "cutoff_date": case.cutoff_date,
            "target_date": case.target_date, "visible_evidence": evidence,
        }, ensure_ascii=False, sort_keys=True)
    )


def _extract_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object in generation")


class EvidenceConditionedAnswerGeneratorV24:
    proposer_name = "open_weight_evidence_conditioned_answer_generator_v2.4"

    def __init__(self, backend: TextGenerationBackendV24, *, max_new_tokens: int = 192):
        self.backend = backend
        self.max_new_tokens = max_new_tokens

    def propose(
        self, case: PublicTemporalCaseV2, state: Any,
    ) -> CompactPayloadProposalV24:
        prompt = answer_prompt_v24(case, state)
        raw = self.backend.generate_text(
            prompt, max_new_tokens=self.max_new_tokens,
            system_prompt=ANSWER_SYSTEM_PROMPT_V24,
        )
        try:
            value = _extract_object(raw)
            if value.get("abstain") is True:
                return CompactPayloadProposalV24(None, "abstained", raw, prompt)
            submission = compact_submission_from_dict_v24(value)
            public = validate_compact_submission_public_v24(
                submission, list(state.collected_evidence),
            )
            if not public.valid:
                return CompactPayloadProposalV24(
                    None, "public_gate_rejected", raw, prompt, public.reason,
                )
            return CompactPayloadProposalV24(
                submission, "valid_compact_candidate", raw, prompt,
            )
        except (TypeError, ValueError) as exc:
            return CompactPayloadProposalV24(
                None, "invalid_generation", raw, prompt, str(exc),
            )
