"""Auditable post-hoc semantic claim judge for non-witness evidence routes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from tkg.api.openrouter import call_model
from tkg.experiment.temporal_beam_ranker import RankerCache, _extract_json
from tkg.experiment.temporal_eval_schema_v2 import RequiredClaimV2, SubmittedClaimV2
from tkg.experiment.temporal_evaluation_v2 import SemanticDecisionV2


SEMANTIC_JUDGE_CONTRACT_V2 = "posthoc-semantic-claim-judge-v2"


class LLMSemanticClaimJudgeV2:
    """Judge relation support after traversal; never available to the solver."""

    def __init__(
        self, model: str, *, version: str, cache_path: str | Path,
        call_model_fn: Callable = call_model,
    ):
        self.model = model
        self.version = version
        self.cache = RankerCache(cache_path)
        self.call_model_fn = call_model_fn

    def close(self) -> None:
        self.cache.close()

    def judge(
        self, required: RequiredClaimV2, submitted: SubmittedClaimV2,
        evidence: list[dict[str, Any]],
    ) -> SemanticDecisionV2:
        judge_input = {
            "required_claim": {
                "claim_id": required.claim_id,
                "subject": required.subject,
                "relation": required.relation,
                "object": required.object,
                "event_time": required.event_time,
            },
            "submitted_claim": submitted.to_dict(),
            "cited_evidence": [{
                "evidence_id": page.get("evidence_id"),
                "title": page.get("title"),
                "revision_id": page.get("revision_id"),
                "timestamp": page.get("timestamp"),
                "content": page.get("content"),
            } for page in evidence],
        }
        prompt = f"""Evaluate whether the cited Wikipedia revision text semantically
supports the submitted relation and the required temporal claim. Do not use outside
knowledge and do not infer from page titles alone. Event time must be expressed or
unambiguously entailed by the cited text. Return JSON only.

Input:
{json.dumps(judge_input, ensure_ascii=False, sort_keys=True)}

Return:
{{
  "decision": "supported|unsupported|unjudgeable",
  "confidence": 0.0,
  "reason": "brief evidence-grounded reason"
}}"""
        cache_key = hashlib.sha256(json.dumps({
            "model": self.model,
            "version": self.version,
            "contract": SEMANTIC_JUDGE_CONTRACT_V2,
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
                        "Judge cited relation support. JSON only."
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            self.cache.put(cache_key, response)
        raw = _extract_json(response)
        decision = str(raw.get("decision") or "unjudgeable").casefold()
        if decision not in {"supported", "unsupported", "unjudgeable"}:
            decision = "unjudgeable"
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if not 0.0 <= confidence <= 1.0:
            confidence = 0.0
            decision = "unjudgeable"
        return SemanticDecisionV2(
            supported={
                "supported": True, "unsupported": False, "unjudgeable": None,
            }[decision],
            confidence=confidence,
            reason=" ".join(str(raw.get("reason") or "").split()),
            model=self.model,
            version=self.version,
            judge_input=judge_input,
            judge_output={
                "parsed": raw,
                "raw_response": response,
                "cache_key": cache_key,
                "cache_hit": cache_hit,
                "contract": SEMANTIC_JUDGE_CONTRACT_V2,
            },
            deterministic_guards={
                "evidence_count": len(evidence),
                "evidence_ids_nonempty": all(
                    bool(page.get("evidence_id")) for page in evidence
                ),
                "temperature": 0.0,
                "judge_is_posthoc_only": True,
                "judge_output_not_available_to_solver": True,
            },
        )
