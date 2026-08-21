"""Versioned topology and attribute-tail contract for candidate generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any


TOPOLOGY_REGISTRY_SCHEMA = "candidate-topology-registry-v1"
STAGED_ADAPTERS = {"p39_office_succession", "mixed_leadership"}


@dataclass(frozen=True)
class TailSpec:
    property_id: str
    family: str
    question: str


@dataclass(frozen=True)
class DirectTopologySpec:
    time_modes: tuple[str, ...]
    target_kind: str
    answer_kind: str


@dataclass(frozen=True)
class StagedTopologySpec:
    event_property_id: str
    adapter: str
    required_relation_ids: tuple[str, ...]


@dataclass(frozen=True)
class CandidateTopologyRegistry:
    schema_version: str
    registry_version: str
    direct_topology: DirectTopologySpec
    staged_topologies: tuple[StagedTopologySpec, ...]
    leadership_relation_ids: tuple[str, ...]
    tails: tuple[TailSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
            "direct_topology": {
                "time_modes": list(self.direct_topology.time_modes),
                "target_kind": self.direct_topology.target_kind,
                "answer_kind": self.direct_topology.answer_kind,
            },
            "staged_topologies": [{
                "event_property_id": row.event_property_id,
                "adapter": row.adapter,
                "required_relation_ids": list(row.required_relation_ids),
            } for row in self.staged_topologies],
            "leadership_relation_ids": list(self.leadership_relation_ids),
            "tails": [{
                "property_id": row.property_id,
                "family": row.family,
                "question": row.question,
            } for row in self.tails],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateTopologyRegistry":
        if value.get("schema_version") != TOPOLOGY_REGISTRY_SCHEMA:
            raise ValueError(
                f"topology registry schema must be {TOPOLOGY_REGISTRY_SCHEMA!r}"
            )
        version = str(value.get("registry_version", "")).strip()
        if not version:
            raise ValueError("topology registry needs a registry_version")

        raw_direct = value.get("direct_topology")
        raw_staged = value.get("staged_topologies")
        raw_leadership = value.get("leadership_relation_ids")
        raw_tails = value.get("tails")
        if not isinstance(raw_direct, dict):
            raise ValueError("topology registry needs a direct_topology object")
        direct_modes = tuple(str(item) for item in raw_direct.get("time_modes", []))
        direct_target = str(raw_direct.get("target_kind", "")).strip()
        direct_answer = str(raw_direct.get("answer_kind", "")).strip()
        if (
            not direct_modes or not set(direct_modes) <= {"interval", "either"}
            or not direct_target or direct_answer not in {"Who", "What", "Where", "Which"}
        ):
            raise ValueError("direct_topology has an invalid machine-readable contract")
        if not isinstance(raw_staged, list) or not isinstance(raw_leadership, list):
            raise ValueError("topology registry needs staged and leadership lists")
        if not isinstance(raw_tails, list) or not raw_tails:
            raise ValueError("topology registry needs a non-empty tails list")

        staged: list[StagedTopologySpec] = []
        for raw in raw_staged:
            if not isinstance(raw, dict):
                raise ValueError("staged topology entries must be objects")
            event_property_id = _property_id(raw.get("event_property_id"))
            adapter = str(raw.get("adapter", "")).strip()
            if adapter not in STAGED_ADAPTERS:
                raise ValueError(f"{event_property_id}: unsupported adapter {adapter!r}")
            required = tuple(_property_id(item) for item in raw.get("required_relation_ids", []))
            if event_property_id not in required:
                raise ValueError(
                    f"{event_property_id}: required_relation_ids must include the event property"
                )
            staged.append(StagedTopologySpec(event_property_id, adapter, required))

        leadership = tuple(_property_id(item) for item in raw_leadership)
        tails: list[TailSpec] = []
        for raw in raw_tails:
            if not isinstance(raw, dict):
                raise ValueError("tail entries must be objects")
            property_id = _property_id(raw.get("property_id"))
            family = str(raw.get("family", "")).strip()
            question = " ".join(str(raw.get("question", "")).split())
            if not family or not question.endswith("?") or "Step 3" not in question:
                raise ValueError(f"{property_id}: malformed tail contract")
            tails.append(TailSpec(property_id, family, question))

        for label, values in (
            ("staged event", [item.event_property_id for item in staged]),
            ("leadership", list(leadership)),
            ("tail", [item.property_id for item in tails]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} property IDs")
        return cls(
            schema_version=TOPOLOGY_REGISTRY_SCHEMA,
            registry_version=version,
            direct_topology=DirectTopologySpec(
                direct_modes, direct_target, direct_answer,
            ),
            staged_topologies=tuple(staged),
            leadership_relation_ids=leadership,
            tails=tuple(tails),
        )


def _property_id(value: Any) -> str:
    property_id = str(value or "").strip()
    if not re.fullmatch(r"P[1-9]\d*", property_id):
        raise ValueError(f"invalid Wikidata property ID: {property_id!r}")
    return property_id


def default_topology_registry_path() -> Path:
    return Path(str(files("tkg.experiment").joinpath(
        "data/candidate_topologies.v1.json"
    )))


def load_candidate_topology_registry(
    path: str | Path | None = None,
) -> CandidateTopologyRegistry:
    selected = Path(path) if path is not None else default_topology_registry_path()
    payload = json.loads(selected.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("topology registry must be a JSON object")
    return CandidateTopologyRegistry.from_dict(payload)
