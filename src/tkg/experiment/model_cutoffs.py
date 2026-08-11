"""Versioned model-cutoff hints used to seed temporal candidates.

These dates are a discovery prior, not proof that a model does or does not
know a particular fact.  Formal admission is still decided by fresh-context PK
probes in :mod:`tkg.experiment.temporal_runner`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


SOURCE_URL = "https://github.com/HaoooWang/llm-knowledge-cutoff-dates"
SOURCE_COMMIT = "9fac20a1706bb0d900428c1339b5f1782d47da1a"


@dataclass(frozen=True)
class ModelCutoff:
    model_id: str
    cutoff_date: str
    cutoff_kind: str
    source_url: str = SOURCE_URL
    source_commit: str = SOURCE_COMMIT

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


# Keep exact provider/model IDs.  Do not guess dates for unlisted aliases.
_CUTOFFS = {
    "openai/gpt-4.1": ModelCutoff(
        "openai/gpt-4.1", "2024-06-01", "knowledge_cutoff"
    ),
    "openai/gpt-4.1-mini": ModelCutoff(
        "openai/gpt-4.1-mini", "2024-06-01", "knowledge_cutoff"
    ),
    "openai/gpt-5-mini": ModelCutoff(
        "openai/gpt-5-mini", "2024-05-31", "knowledge_cutoff"
    ),
}


def get_model_cutoff(model_id: str) -> ModelCutoff:
    """Return a pinned cutoff record; fail closed for unknown model IDs."""
    normalized = model_id.strip().casefold()
    try:
        return _CUTOFFS[normalized]
    except KeyError as exc:
        known = ", ".join(sorted(_CUTOFFS))
        raise ValueError(
            f"no pinned knowledge-cutoff record for {model_id!r}; known models: {known}"
        ) from exc


def model_matches_cutoff(case: dict, model_id: str) -> bool:
    """Whether a cutoff-relative case is valid for this exact tested model."""
    cutoff = case.get("knowledge_cutoff")
    if not isinstance(cutoff, dict):
        return True
    allowed = cutoff.get("model_ids", [])
    return isinstance(allowed, list) and model_id.casefold() in {
        str(item).casefold() for item in allowed
    }
