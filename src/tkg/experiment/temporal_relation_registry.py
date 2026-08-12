"""Versioned data contract for renewable temporal relation discovery."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any


REGISTRY_SCHEMA = "temporal-relation-registry-v1"
TIME_MODES = {"interval", "point", "either"}
SELECTION_POLICIES = {"single_active", "latest_start", "latest_point"}
RELATION_STATUSES = {
    "discovered", "bootstrap", "profiled", "validated", "active",
    "quarantined", "deprecated",
}


@dataclass(frozen=True)
class TemporalRelationSpec:
    property_id: str
    label: str
    family: str
    time_mode: str
    source_kind: str
    target_kind: str
    answer_kind: str
    relative_clause: str
    status: str
    semantic_guidance: str = ""
    selection_policy: str = "single_active"
    later_relative_clause: str = ""
    inverse_relative_clause: str = ""
    inverse_answer_kind: str = "What"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TemporalRelationSpec":
        required = (
            "property_id", "label", "family", "time_mode", "source_kind",
            "target_kind", "answer_kind", "relative_clause", "status",
        )
        missing = [field for field in required if not str(value.get(field, "")).strip()]
        if missing:
            raise ValueError(f"temporal relation missing fields: {', '.join(missing)}")
        property_id = str(value["property_id"]).strip()
        if not re.fullmatch(r"P[1-9]\d*", property_id):
            raise ValueError(f"invalid Wikidata property ID: {property_id!r}")
        time_mode = str(value["time_mode"]).strip()
        if time_mode not in TIME_MODES:
            raise ValueError(f"{property_id}: invalid time_mode {time_mode!r}")
        status = str(value["status"]).strip()
        if status not in RELATION_STATUSES:
            raise ValueError(f"{property_id}: invalid status {status!r}")
        clause = str(value["relative_clause"]).strip()
        if clause.count("{source}") != 1:
            raise ValueError(f"{property_id}: relative_clause needs one {{source}}")
        answer_kind = str(value["answer_kind"]).strip()
        if answer_kind not in {"Who", "What", "Where", "Which"}:
            raise ValueError(f"{property_id}: invalid answer_kind {answer_kind!r}")
        selection_policy = str(
            value.get(
                "selection_policy",
                "latest_point" if time_mode == "point" else "single_active",
            )
        ).strip()
        if selection_policy not in SELECTION_POLICIES:
            raise ValueError(
                f"{property_id}: invalid selection_policy {selection_policy!r}"
            )
        later_clause = str(value.get("later_relative_clause", "")).strip()
        inverse_clause = str(value.get("inverse_relative_clause", "")).strip()
        for field_name, optional_clause in (
            ("later_relative_clause", later_clause),
            ("inverse_relative_clause", inverse_clause),
        ):
            if optional_clause and optional_clause.count("{source}") != 1:
                raise ValueError(
                    f"{property_id}: {field_name} needs one {{source}}"
                )
        inverse_answer_kind = str(value.get("inverse_answer_kind", "What")).strip()
        if inverse_answer_kind not in {"Who", "What", "Where", "Which"}:
            raise ValueError(
                f"{property_id}: invalid inverse_answer_kind "
                f"{inverse_answer_kind!r}"
            )
        return cls(
            property_id=property_id,
            label=str(value["label"]).strip(),
            family=str(value["family"]).strip(),
            time_mode=time_mode,
            source_kind=str(value["source_kind"]).strip(),
            target_kind=str(value["target_kind"]).strip(),
            answer_kind=answer_kind,
            relative_clause=clause,
            status=status,
            semantic_guidance=str(value.get("semantic_guidance", "")).strip(),
            selection_policy=selection_policy,
            later_relative_clause=later_clause,
            inverse_relative_clause=inverse_clause,
            inverse_answer_kind=inverse_answer_kind,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalRelationRegistry:
    schema_version: str
    registry_version: str
    relations: tuple[TemporalRelationSpec, ...]
    provenance: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TemporalRelationRegistry":
        schema = str(value.get("schema_version", ""))
        if schema != REGISTRY_SCHEMA:
            raise ValueError(
                f"relation registry schema must be {REGISTRY_SCHEMA!r}, got {schema!r}"
            )
        version = str(value.get("registry_version", "")).strip()
        raw_relations = value.get("relations")
        if not version or not isinstance(raw_relations, list) or not raw_relations:
            raise ValueError("relation registry needs a version and non-empty relations")
        relations = tuple(TemporalRelationSpec.from_dict(row) for row in raw_relations)
        property_ids = [relation.property_id for relation in relations]
        if len(property_ids) != len(set(property_ids)):
            raise ValueError("relation registry contains duplicate property IDs")
        provenance = value.get("provenance")
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError("relation registry provenance must be an object")
        return cls(
            schema_version=schema, registry_version=version,
            relations=relations,
            provenance=dict(provenance) if provenance is not None else None,
        )

    def select(self, property_ids: set[str] | None = None) -> tuple[TemporalRelationSpec, ...]:
        if not property_ids:
            return self.relations
        missing = property_ids - {relation.property_id for relation in self.relations}
        if missing:
            raise ValueError(f"unknown registry properties: {sorted(missing)!r}")
        return tuple(
            relation for relation in self.relations
            if relation.property_id in property_ids
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
            "relations": [relation.to_dict() for relation in self.relations],
        }
        if self.provenance is not None:
            result["provenance"] = self.provenance
        return result


def default_registry_path() -> Path:
    resource = files("tkg.experiment").joinpath(
        "data/temporal_relations.bootstrap.json"
    )
    return Path(str(resource))


def load_temporal_relation_registry(
    path: str | Path | None = None,
) -> TemporalRelationRegistry:
    selected = Path(path) if path is not None else default_registry_path()
    payload = json.loads(selected.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("relation registry must be a JSON object")
    return TemporalRelationRegistry.from_dict(payload)
