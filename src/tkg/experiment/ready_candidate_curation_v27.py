"""Curate a fresh candidate pool without exposing sealed-case identities.

The sealed artifact is used only as an exclusion set.  Its case contents and
identifiers are never copied into the output or printed to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tkg.experiment.results import assert_new_output_path


SCHEMA_VERSION = "ready-candidate-curation-v2.7"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_candidate_ids(value: Any) -> set[str]:
    """Collect exclusion IDs recursively without returning any case payload."""
    found: set[str] = set()
    if isinstance(value, dict):
        candidate_id = value.get("source_candidate_id")
        if isinstance(candidate_id, str) and candidate_id.startswith("kgcand_"):
            found.add(candidate_id)
        for child in value.values():
            found.update(_source_candidate_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_source_candidate_ids(child))
    return found


def _spine_key(row: dict[str, Any]) -> str:
    chain = row.get("private_chain", [])
    public_edges = []
    for edge in chain[:3] if isinstance(chain, list) else []:
        if not isinstance(edge, dict):
            continue
        public_edges.append({
            "source_qid": edge.get("source_qid"),
            "target_qid": edge.get("target_qid"),
            "property_id": edge.get("property_id"),
            "event_date": edge.get("event_date"),
            "event_start": edge.get("event_start"),
        })
    return _sha256_bytes(json.dumps(
        public_edges, sort_keys=True, separators=(",", ":"),
    ).encode())


def curate(
    candidates: Iterable[dict[str, Any]], *, excluded_ids: set[str], limit: int,
    excluded_spines: set[str] | None = None,
) -> list[dict[str, Any]]:
    blocked_spines = excluded_spines or set()
    deduplicated: dict[str, dict[str, Any]] = {}
    for row in candidates:
        candidate_id = str(row.get("id") or "")
        fingerprint = str(row.get("question_fingerprint") or candidate_id)
        if candidate_id and candidate_id not in excluded_ids:
            deduplicated.setdefault(fingerprint, row)

    by_spine: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in deduplicated.values():
        key = _spine_key(row)
        if key not in blocked_spines:
            by_spine[key].append(row)
    for rows in by_spine.values():
        rows.sort(key=lambda row: (
            str(row.get("tail_property")), str(row.get("id")),
        ))

    selected: list[dict[str, Any]] = []
    # Round-robin over tail depth: every spine contributes one question before
    # any spine contributes its second wording/tail variant.
    depth = 0
    ordered_spines = sorted(by_spine, key=lambda key: (
        str(by_spine[key][0].get("relation_property")),
        str(by_spine[key][0].get("public_anchor")), key,
    ))
    while len(selected) < limit:
        progress = False
        for key in ordered_spines:
            rows = by_spine[key]
            if depth < len(rows):
                selected.append(rows[depth])
                progress = True
                if len(selected) >= limit:
                    break
        if not progress:
            break
        depth += 1
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--sealed-cases", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--exclude-certified-seeds", action="append", default=[],
        help="exclude every candidate sharing a temporal spine with these seeds",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.limit <= 0 or args.offset < 0:
        parser.error("--limit must be positive and --offset must be non-negative")
    assert_new_output_path(args.output)

    sealed_path = Path(args.sealed_cases)
    sealed_bytes = sealed_path.read_bytes()
    excluded_ids = _source_candidate_ids(json.loads(sealed_bytes))
    candidates: list[dict[str, Any]] = []
    sources = []
    for raw_path in args.input:
        path = Path(raw_path)
        raw = path.read_bytes()
        payload = json.loads(raw)
        rows = payload.get("questions", []) if isinstance(payload, dict) else []
        valid = [dict(row) for row in rows if isinstance(row, dict)]
        candidates.extend(valid)
        sources.append({
            "path": str(path), "sha256": _sha256_bytes(raw), "count": len(valid),
        })

    candidate_by_id = {str(row.get("id") or ""): row for row in candidates}
    excluded_spines: set[str] = set()
    certified_sources = []
    for raw_path in args.exclude_certified_seeds:
        path = Path(raw_path)
        raw = path.read_bytes()
        payload = json.loads(raw)
        rows = payload.get("seeds", []) if isinstance(payload, dict) else []
        matched = 0
        for seed in rows:
            if not isinstance(seed, dict):
                continue
            metadata = seed.get("selection_metadata", {})
            source_id = metadata.get("source_candidate_id") if isinstance(metadata, dict) else None
            candidate = candidate_by_id.get(str(source_id or ""))
            if candidate is not None:
                excluded_spines.add(_spine_key(candidate))
                matched += 1
        certified_sources.append({
            "path": str(path), "sha256": _sha256_bytes(raw),
            "seed_count": len(rows), "matched_source_count": matched,
        })

    ordered = curate(
        candidates, excluded_ids=excluded_ids, excluded_spines=excluded_spines,
        limit=args.offset + args.limit,
    )
    selected = ordered[args.offset:args.offset + args.limit]
    property_counts = Counter(str(row.get("relation_property")) for row in selected)
    domain_counts = Counter(str(row.get("domain_family")) for row in selected)
    tail_counts = Counter(str(row.get("tail_family")) for row in selected)
    spine_count = len({_spine_key(row) for row in selected})
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "fresh_candidates_pending_promotion",
        "source_batches": sources,
        "sealed_exclusion": {
            "artifact_sha256": _sha256_bytes(sealed_bytes),
            "excluded_source_id_count": len(excluded_ids),
            "identities_exported": False,
        },
        "certified_spine_exclusion": {
            "sources": certified_sources,
            "excluded_spine_count": len(excluded_spines),
        },
        "input_candidate_count": len(candidates),
        "selection_offset": args.offset,
        "selected_question_count": len(selected),
        "unique_spine_count": spine_count,
        "counts_by_relation_property": dict(sorted(property_counts.items())),
        "counts_by_domain": dict(sorted(domain_counts.items())),
        "counts_by_tail_family": dict(sorted(tail_counts.items())),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        {"_batch": metadata, "questions": selected},
        ensure_ascii=False, indent=2,
    ) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_question_count": len(selected),
        "unique_spine_count": spine_count,
        "excluded_source_id_count": len(excluded_ids),
        "counts_by_relation_property": dict(sorted(property_counts.items())),
    }, sort_keys=True))
    return 0 if len(selected) >= args.limit else 1


if __name__ == "__main__":
    raise SystemExit(main())
