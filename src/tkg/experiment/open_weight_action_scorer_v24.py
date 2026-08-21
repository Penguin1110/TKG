"""Actual continuation-token conditional log-probability action scoring v2.4."""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from tkg.experiment.joint_controller_v23 import JointCandidateActionV23


OPEN_WEIGHT_SCORER_PROTOCOL_V24 = "open-weight-action-conditional-logprob-v2.4"


class ConditionalTokenLogProbBackendV24(Protocol):
    backend_name: str

    def conditional_token_logprobs(
        self, prompt: str, continuation: str,
    ) -> list[float]:
        ...


@dataclass(frozen=True)
class OpenWeightActionScoresV24:
    scores: dict[str, float]
    token_counts: dict[str, int]
    serialized_actions: dict[str, str]
    backend_name: str
    score_kind: str = "length_normalized_conditional_logprob"
    protocol: str = OPEN_WEIGHT_SCORER_PROTOCOL_V24


def serialize_action_for_scoring_v24(action: JointCandidateActionV23) -> str:
    row = action.to_dict()
    public = {
        "kind": row["kind"],
        "label": row["label"],
        "params": row.get("params", {}),
    }
    return "ACTION " + json.dumps(
        public, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


class OpenWeightConditionalActionScorerV24:
    """Scores every supplied action from model token probabilities, not utilities."""

    def __init__(self, backend: ConditionalTokenLogProbBackendV24):
        self.backend = backend

    def score(
        self, prompt: str, actions: Sequence[JointCandidateActionV23],
    ) -> OpenWeightActionScoresV24:
        if not actions:
            raise ValueError("at least one action is required")
        if len({action.action_id for action in actions}) != len(actions):
            raise ValueError("action IDs must be unique")
        scores = {}
        counts = {}
        serialized = {}
        continuations = [serialize_action_for_scoring_v24(action) for action in actions]
        batch_method = getattr(self.backend, "conditional_token_logprobs_batch", None)
        if callable(batch_method):
            all_logprobs = batch_method(prompt, continuations)
            if len(all_logprobs) != len(actions):
                raise ValueError("backend batch score count mismatch")
        else:
            all_logprobs = [
                self.backend.conditional_token_logprobs(prompt, continuation)
                for continuation in continuations
            ]
        for action, continuation, token_logprobs in zip(
            actions, continuations, all_logprobs, strict=True,
        ):
            if not token_logprobs:
                raise ValueError(f"empty continuation tokenization: {action.action_id}")
            if any(not math.isfinite(value) for value in token_logprobs):
                raise ValueError("backend returned non-finite token log-probability")
            scores[action.action_id] = sum(token_logprobs) / len(token_logprobs)
            counts[action.action_id] = len(token_logprobs)
            serialized[action.action_id] = continuation
        return OpenWeightActionScoresV24(
            scores=scores, token_counts=counts, serialized_actions=serialized,
            backend_name=self.backend.backend_name,
        )


class HuggingFaceCausalLMBackendV24:
    """Lazy optional backend using actual causal-LM logits for each action."""

    backend_name = "huggingface_causal_lm_logits_v2.4"

    def __init__(
        self, model_name_or_path: str, *, device: str = "auto",
        dtype: str = "auto", trust_remote_code: bool = False,
    ):
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFace scoring requires optional torch and transformers"
            ) from exc
        self.torch = torch
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code,
        )
        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, dtype)
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **model_kwargs,
        )
        if device != "auto":
            self.model.to(device)
        self.model.eval()

    def generate_text(
        self, prompt: str, *, max_new_tokens: int = 192,
        system_prompt: str | None = None,
    ) -> str:
        """Greedy evidence-conditioned generation from the same checkpoint."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                rendered = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        else:
            rendered = "\n".join(
                f"{row['role'].upper()}: {row['content']}" for row in messages
            ) + "\nASSISTANT:"
        inputs = self.tokenizer(rendered, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=max_new_tokens,
                pad_token_id=(self.tokenizer.pad_token_id
                              or self.tokenizer.eos_token_id),
            )
        generated = output[0, inputs["input_ids"].shape[1]:]
        return str(self.tokenizer.decode(generated, skip_special_tokens=True)).strip()

    def conditional_token_logprobs(
        self, prompt: str, continuation: str,
    ) -> list[float]:
        torch = self.torch
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=True, return_tensors="pt",
        )["input_ids"]
        continuation_ids = self.tokenizer(
            continuation, add_special_tokens=False, return_tensors="pt",
        )["input_ids"]
        if continuation_ids.shape[1] == 0:
            return []
        input_ids = torch.cat([prompt_ids, continuation_ids], dim=1)
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits
            log_probs = torch.log_softmax(logits, dim=-1)
        prompt_length = prompt_ids.shape[1]
        values = []
        for offset in range(continuation_ids.shape[1]):
            token_id = int(continuation_ids[0, offset])
            prediction_position = prompt_length + offset - 1
            values.append(float(log_probs[0, prediction_position, token_id].item()))
        return values

    def conditional_token_logprobs_batch(
        self, prompt: str, continuations: list[str],
    ) -> list[list[float]]:
        if not continuations:
            return []
        torch = self.torch
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=True, return_tensors="pt",
        )["input_ids"][0]
        continuation_ids = [
            self.tokenizer(
                continuation, add_special_tokens=False, return_tensors="pt",
            )["input_ids"][0]
            for continuation in continuations
        ]
        if any(ids.shape[0] == 0 for ids in continuation_ids):
            return [[] for _ in continuations]
        sequences = [torch.cat([prompt_ids, ids], dim=0) for ids in continuation_ids]
        maximum = max(sequence.shape[0] for sequence in sequences)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        input_ids = torch.full(
            (len(sequences), maximum), int(pad_id), dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, sequence in enumerate(sequences):
            input_ids[index, :sequence.shape[0]] = sequence
            attention_mask[index, :sequence.shape[0]] = 1
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
            ).logits
            log_probs = torch.log_softmax(logits, dim=-1)
        prompt_length = prompt_ids.shape[0]
        result = []
        for batch_index, ids in enumerate(continuation_ids):
            values = []
            for offset in range(ids.shape[0]):
                token_id = int(ids[offset])
                position = prompt_length + offset - 1
                values.append(float(log_probs[batch_index, position, token_id].item()))
            result.append(values)
        return result
