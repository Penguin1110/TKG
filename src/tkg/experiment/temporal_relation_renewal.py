"""Materialize discovered and profiled relations into a new registry version.

Profiling artifacts are immutable observations.  This command applies them to
a new registry artifact instead of mutating the bootstrap file, so every paper
run can cite an exact registry hash and review policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_relation_profiler import PROFILE_SCHEMA
from tkg.experiment.temporal_relation_registry import (
    TemporalRelationRegistry,
    TemporalRelationSpec,
    load_temporal_relation_registry,
)


VALIDATED_RECOMMENDATION = "semantic_validated_candidate_human_review_required"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _generic_discovered_spec(candidate: dict[str, Any]) -> TemporalRelationSpec:
    property_id = str(candidate.get("property_id", ""))
    if not re.fullmatch(r"P[1-9]\d*", property_id):
        raise ValueError(f"invalid discovered property ID: {property_id!r}")
    label = " ".join(str(candidate.get("label", property_id)).split())
    if not label or "{" in label or "}" in label:
        label = property_id
    qualifiers = {str(value) for value in candidate.get("qualifiers", [])}
    has_point = "P585" in qualifiers
    has_interval = bool(qualifiers & {"P580", "P582"})
    time_mode = "either" if has_point and has_interval else "point" if has_point else "interval"
    selection = "latest_point" if time_mode == "point" else "single_active"
    return TemporalRelationSpec(
        property_id=property_id, label=label, family="unclassified",
        time_mode=time_mode, source_kind="entity", target_kind="entity",
        answer_kind="What", relative_clause=f"the {label} of {{source}}",
        status="discovered",
        semantic_guidance=(
            "Require the local excerpt to explicitly express the named Wikidata "
            "property in the stated source-to-target direction; reject co-occurrence."
        ),
        selection_policy=selection,
        later_relative_clause=f"the next {label} of {{source}} after that",
    )


def renew_registry(
    base: TemporalRelationRegistry,
    profile_payloads: list[tuple[str, str, dict[str, Any]]],
    *,
    registry_version: str,
    activate_properties: set[str] | None = None,
    activate_validated: bool = False,
    deprecate_properties: set[str] | None = None,
) -> TemporalRelationRegistry:
    """Create one auditable successor registry from immutable profile artifacts."""
    if not registry_version.strip() or registry_version == base.registry_version:
        raise ValueError("registry_version must be non-empty and differ from the base")
    activate = set(activate_properties or ())
    deprecate = set(deprecate_properties or ())
    overlap = activate & deprecate
    if overlap:
        raise ValueError(f"properties cannot be activated and deprecated: {sorted(overlap)}")
    specs = {spec.property_id: spec for spec in base.relations}
    profile_decisions: dict[str, list[str]] = {}
    source_artifacts = []
    for source_path, source_sha256, payload in profile_payloads:
        if payload.get("schema_version") != PROFILE_SCHEMA:
            raise ValueError(f"{source_path}: not a {PROFILE_SCHEMA} artifact")
        profile_registry = payload.get("registry", {})
        if profile_registry.get("schema_version") != base.schema_version:
            raise ValueError(f"{source_path}: registry schema mismatch")
        source_artifacts.append({"path": source_path, "sha256": source_sha256})
        for profile in payload.get("profiles", []):
            if not isinstance(profile, dict):
                continue
            relation = profile.get("relation", {})
            property_id = str(relation.get("property_id", ""))
            if property_id not in specs:
                continue
            recommendation = str(profile.get("recommendation", ""))
            profile_decisions.setdefault(property_id, []).append(recommendation)
        mining = payload.get("property_mining", {})
        for candidate in mining.get("candidates", []) if isinstance(mining, dict) else []:
            if not isinstance(candidate, dict):
                continue
            property_id = str(candidate.get("property_id", ""))
            if property_id not in specs:
                spec = _generic_discovered_spec(candidate)
                specs[property_id] = spec

    for property_id, decisions in profile_decisions.items():
        spec = specs[property_id]
        if spec.status == "deprecated":
            continue
        if any(value.startswith("quarantine_") for value in decisions):
            specs[property_id] = replace(spec, status="quarantined")
        elif VALIDATED_RECOMMENDATION in decisions:
            specs[property_id] = replace(spec, status="validated")

    unknown = (activate | deprecate) - set(specs)
    if unknown:
        raise ValueError(f"unknown properties in review action: {sorted(unknown)}")
    if activate_validated:
        activate.update(
            property_id for property_id, spec in specs.items()
            if spec.status == "validated"
        )
    invalid_activation = {
        property_id for property_id in activate
        if specs[property_id].status not in {"validated", "active"}
    }
    if invalid_activation:
        raise ValueError(
            "activation requires semantic validation: "
            f"{sorted(invalid_activation)}"
        )
    for property_id in activate:
        specs[property_id] = replace(specs[property_id], status="active")
    for property_id in deprecate:
        specs[property_id] = replace(specs[property_id], status="deprecated")

    ordered = tuple(
        specs[property_id]
        for property_id in [
            *(spec.property_id for spec in base.relations),
            *sorted(set(specs) - {spec.property_id for spec in base.relations}),
        ]
    )
    provenance: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_registry_version": base.registry_version,
        "source_profiles": source_artifacts,
        "promotion_policy": (
            "automatic_semantic_threshold" if activate_validated
            else "explicit_property_review"
        ),
        "explicitly_activated": sorted(activate_properties or ()),
        "automatically_activated": sorted(
            activate - set(activate_properties or ())
        ),
        "deprecated": sorted(deprecate),
        "profile_decisions": {
            key: sorted(set(values)) for key, values in sorted(profile_decisions.items())
        },
    }
    return TemporalRelationRegistry(
        schema_version=base.schema_version, registry_version=registry_version,
        relations=ordered, provenance=provenance,
    )


def _parse_properties(value: str | None) -> set[str]:
    if not value:
        return set()
    properties = {item.strip() for item in value.split(",") if item.strip()}
    invalid = {value for value in properties if not re.fullmatch(r"P[1-9]\d*", value)}
    if invalid:
        raise ValueError(f"invalid property IDs: {sorted(invalid)}")
    return properties


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a versioned successor temporal-relation registry"
    )
    parser.add_argument("--registry")
    parser.add_argument("--profile", action="append", required=True)
    parser.add_argument("--registry-version", required=True)
    parser.add_argument("--activate-properties")
    parser.add_argument(
        "--activate-validated", action="store_true",
        help="explicitly choose deterministic auto-activation of all semantic passes",
    )
    parser.add_argument("--deprecate-properties")
    parser.add_argument("--output", default="temporal_relations.renewed.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    assert_new_output_path(args.output)
    if Path(args.output).exists() and not args.overwrite:
        parser.error(f"refusing to overwrite {args.output}; pass --overwrite explicitly")
    try:
        base = load_temporal_relation_registry(args.registry)
        payloads = []
        for value in args.profile:
            path = Path(value)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"{path}: profile must be a JSON object")
            payloads.append((str(path), _sha256(path), payload))
        renewed = renew_registry(
            base, payloads, registry_version=args.registry_version,
            activate_properties=_parse_properties(args.activate_properties),
            activate_validated=args.activate_validated,
            deprecate_properties=_parse_properties(args.deprecate_properties),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    target = Path(args.output)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(renewed.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    counts: dict[str, int] = {}
    for spec in renewed.relations:
        counts[spec.status] = counts.get(spec.status, 0) + 1
    print(
        f"[done] registry={renewed.registry_version} relations={len(renewed.relations)} "
        f"statuses={counts}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
