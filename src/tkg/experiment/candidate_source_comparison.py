"""Compare candidate discovery sources after identical Wikipedia proof gates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "candidate-source-comparison-v1"
KNOWN_SOURCES = {"wikipedia_first", "wikidata_first", "hybrid", "manual"}


def candidate_source(packet: dict[str, Any]) -> str:
    metadata = packet.get("selection_metadata")
    value = metadata.get("candidate_source") if isinstance(metadata, dict) else None
    source = str(value or "unspecified").strip()
    return source if source in KNOWN_SOURCES else "unspecified"


def _counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(value for value in values if value).items()))


def _semantic_contrast_count(chain: list[dict[str, Any]]) -> int:
    return sum(bool(hop.get("prior_relation_contrast_verified")) for hop in chain)


def _redirect_count(chain: list[dict[str, Any]]) -> int:
    return sum(
        hop.get("requested_source_title", hop.get("source_title"))
        != hop.get("source_title")
        or hop.get("requested_target_title", hop.get("target_title"))
        != hop.get("target_title")
        for hop in chain
    )


def compare_candidate_sources(packets: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate gate outcomes without treating a tiny batch as a winner."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for packet in packets:
        grouped.setdefault(candidate_source(packet), []).append(packet)

    sources: dict[str, dict[str, Any]] = {}
    for source, rows in sorted(grouped.items()):
        chains = [
            row.get("chain", []) if isinstance(row.get("chain"), list) else []
            for row in rows
        ]
        verified_hops = [len(chain) for chain in chains]
        deterministic_pass = sum(not row.get("deterministic_errors") for row in rows)
        judge_pass = sum(
            row.get("status") == "machine_pass_human_review_required"
            for row in rows
        )
        sources[source] = {
            "candidate_count": len(rows),
            "status_counts": _counter(str(row.get("status", "unknown")) for row in rows),
            "deterministic_pass_count": deterministic_pass,
            "deterministic_pass_rate": deterministic_pass / len(rows),
            "judge_pass_count": judge_pass,
            "judge_pass_rate": judge_pass / len(rows),
            "deterministic_error_counts": _counter(
                str(error)
                for row in rows
                for error in (
                    row.get("deterministic_errors", [])
                    if isinstance(row.get("deterministic_errors"), list) else []
                )
            ),
            "mean_verified_hops": sum(verified_hops) / len(rows),
            "semantic_contrast_hops": sum(
                _semantic_contrast_count(chain) for chain in chains
            ),
            "verified_redirect_hops": sum(_redirect_count(chain) for chain in chains),
            "categories": _counter(str(row.get("category", "other")) for row in rows),
            "relation_families": _counter(
                str(hop.get("relation_family"))
                for chain in chains for hop in chain if hop.get("relation_family")
            ),
        }

    enough_per_source = bool(sources) and all(
        row["candidate_count"] >= 20 for row in sources.values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_count": len(packets),
        "sources": sources,
        "comparison_is_conclusive": enough_per_source,
        "interpretation": (
            "Rates describe candidates after the same exact Wikipedia revision, evidence, "
            "hyperlink, temporal, shortest-route, and LLM-judge gates. Candidate-source "
            "provenance changes discovery only; Wikipedia remains the formal evidence."
        ),
        "sample_size_warning": (
            None if enough_per_source else
            "Fewer than 20 candidates exist in at least one source group; do not rank sources."
        ),
    }


def _load_packets(paths: list[str]) -> list[dict[str, Any]]:
    packets = []
    for path in paths:
        for line_number, line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: packet must be an object")
            packets.append(value)
    return packets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Wikipedia-first and Wikidata-first candidate gate outcomes"
    )
    parser.add_argument("--packets", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        parser.error(f"refusing to overwrite {output}; pass --overwrite explicitly")
    try:
        report = compare_candidate_sources(_load_packets(args.packets))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote candidate-source comparison: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
