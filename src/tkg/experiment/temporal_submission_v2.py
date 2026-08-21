"""Gold-free structured-submission generation for v2 inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from tkg.api.openrouter import call_model
from tkg.experiment.temporal_beam_ranker import RankerCache, _extract_json
from tkg.experiment.temporal_eval_schema_v2 import (
    PublicTemporalCaseV2, StructuredSubmissionV2, SUBMISSION_SCHEMA_V2,
    structured_submission_from_dict,
)
from tkg.experiment.temporal_evaluation_v2 import evidence_id_v2
from tkg.experiment.temporal_environment_v2 import EnvironmentActionV2


STRUCTURED_PROPOSER_CONTRACT_V2 = "structured-submission-proposer-v2"


def structured_submit_action_v2(
    submission: StructuredSubmissionV2,
) -> EnvironmentActionV2 | None:
    if not submission.answer:
        return None
    return EnvironmentActionV2(
        kind="SUBMIT_ANSWER",
        params=submission.to_dict(),
        label=f'Submit structured evidence answer "{submission.answer}"',
    )


class StructuredSubmissionProposerV2:
    """Propose claims from visible evidence, without aliases or reference data."""

    def __init__(
        self, model: str, *, cache_path: str | Path,
        call_model_fn: Callable = call_model, max_evidence_chars: int = 20_000,
    ):
        self.model = model
        self.cache = RankerCache(cache_path)
        self.call_model_fn = call_model_fn
        self.max_evidence_chars = max_evidence_chars

    def close(self) -> None:
        self.cache.close()

    def _visible_evidence(self, pages: list[dict[str, Any]]) -> str:
        remaining = self.max_evidence_chars
        records = []
        for page in reversed(pages):
            evidence_id = str(page.get("evidence_id") or evidence_id_v2(page))
            header = (
                f"[EVIDENCE_ID {evidence_id} | PAGE {page.get('title')} | "
                f"revision {page.get('revision_id')} | {page.get('timestamp')}]\n"
            )
            content = str(page.get("content") or "")[:max(0, remaining - len(header))]
            if not content:
                break
            records.append(header + content)
            remaining -= len(header) + len(content)
            if remaining <= 0:
                break
        return "\n".join(reversed(records))

    def prompt(
        self, public_case: PublicTemporalCaseV2,
        evidence_pages: list[dict[str, Any]], *, seed: int,
    ) -> str:
        return f"""Propose one structured final submission using only visible evidence.
Do not use outside knowledge. Return an empty answer and empty claim arrays if the
complete multi-hop question is not supported. Do not expose chain-of-thought.

Question: {public_case.question}
Knowledge cutoff: {public_case.cutoff_date}
Target date: {public_case.target_date}
Deterministic tie seed: {seed}

Visible evidence:
{self._visible_evidence(evidence_pages)}

Return exactly one JSON object:
{{
  "schema_version": "{SUBMISSION_SCHEMA_V2}",
  "answer": "one entity or noun phrase, at most 8 words",
  "critical_claims": [
    {{
      "claim_id": "your stable local identifier or null",
      "subject": "entity",
      "relation": "relation expressed by cited evidence",
      "object": "entity",
      "event_time": "YYYY-MM-DD or null",
      "supporting_evidence_ids": ["EVIDENCE_ID"]
    }}
  ],
  "tail_claim": {{
    "claim_id": "tail",
    "subject": "entity reached by the temporal bridge",
    "relation": "final requested relation",
    "object": "final answer",
    "event_time": null,
    "supporting_evidence_ids": ["EVIDENCE_ID"]
  }}
}}
Every evidence ID must be visible above. The answer alone is insufficient: submit
only when the cited evidence supports the temporal bridge, event time, tail relation,
and their composition."""

    def propose(
        self, public_case: PublicTemporalCaseV2,
        evidence_pages: list[dict[str, Any]], *, seed: int,
    ) -> tuple[StructuredSubmissionV2, dict[str, Any]]:
        prompt = self.prompt(public_case, evidence_pages, seed=seed)
        cache_key = hashlib.sha256(json.dumps({
            "model": self.model,
            "prompt": prompt,
            "contract": STRUCTURED_PROPOSER_CONTRACT_V2,
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
                        "Generate a structured evidence submission. JSON only."
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            self.cache.put(cache_key, response)
        parsed = _extract_json(response)
        submission = structured_submission_from_dict(parsed)
        if submission.answer and len(submission.answer.split()) > 8:
            raise ValueError("structured answer must contain at most 8 words")
        return submission, {
            "contract": STRUCTURED_PROPOSER_CONTRACT_V2,
            "cache_key": cache_key,
            "cache_hit": cache_hit,
            "model": self.model,
            "raw_response": response,
        }
