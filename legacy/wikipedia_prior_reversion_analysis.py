"""Archived analysis for the prior-reversion protocol."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict

from tkg.experiment.results import assert_new_output_path


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bootstrap_diff_ci(a: list[int], b: list[int], rng: random.Random, n_boot=2000):
    if not a or not b:
        return None, None
    diffs = []
    for _ in range(n_boot):
        mean_a = sum(rng.choice(a) for _ in a) / len(a)
        mean_b = sum(rng.choice(b) for _ in b) / len(b)
        diffs.append(mean_a - mean_b)
    diffs.sort()
    return round(100 * diffs[int(0.025 * n_boot)], 1), round(100 * diffs[int(0.975 * n_boot)], 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="wikipedia_results.jsonl")
    parser.add_argument("--summary-output", default="wikipedia_summary.csv")
    parser.add_argument("--diff-output", default="wikipedia_conflict_vs_control.csv")
    parser.add_argument("--gate-output", default="wikipedia_gate_summary.csv")
    parser.add_argument("--transition-output", default="wikipedia_transitions.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contract-hash",
                        help="Analyze one contract only; defaults to the latest hash in the JSONL")
    args = parser.parse_args()
    for output in (args.summary_output, args.diff_output, args.gate_output, args.transition_output):
        assert_new_output_path(output)

    rows = load_rows(args.input)
    contract_hash = args.contract_hash
    if contract_hash is None:
        contract_hash = next(
            (row.get("contract_hash") for row in reversed(rows) if row.get("contract_hash")), None
        )
    if contract_hash is not None:
        rows = [row for row in rows if row.get("contract_hash") == contract_hash]
        print(f"Analyzing contract_hash={contract_hash}")
    completed_attempts = {
        row.get("attempt_id") for row in rows
        if row.get("slot") == "checkpoint" and row.get("status") == "complete"
    }
    # Attempt rows are append-only and may include a failed partial retry. Analyze
    # only the exact attempt_id named by a completion checkpoint.
    rows = [
        row for row in rows
        if not row.get("attempt_id") or row.get("attempt_id") in completed_attempts
    ]
    rng = random.Random(args.seed)

    # Browse/gate attrition is part of the result, not discarded preprocessing.
    gates: dict[tuple, dict[str, int]] = defaultdict(
        lambda: {"attempts": 0, "page_hits": 0, "eligible": 0}
    )
    for row in rows:
        key = (row.get("model"), row.get("case_id"), row.get("arm"), row.get("start_distance"))
        if row.get("slot") == "browse_summary":
            gates[key]["attempts"] += 1
            gates[key]["page_hits"] += int(bool(row.get("page_hit")))
        elif row.get("slot") == "exposure_gate":
            gates[key]["eligible"] += int(bool(row.get("gate", {}).get("eligible")))
    with open(args.gate_output, "w", newline="", encoding="utf-8") as fh:
        fields = ["model", "case_id", "arm", "start_distance", "attempts", "page_hits",
                  "eligible", "page_hit_rate_pct", "eligible_rate_pct"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key, count in sorted(gates.items(), key=lambda item: str(item[0])):
            n = count["attempts"]
            writer.writerow(dict(zip(fields[:4], key)) | count | {
                "page_hit_rate_pct": round(100 * count["page_hits"] / n, 1) if n else None,
                "eligible_rate_pct": round(100 * count["eligible"] / n, 1) if n else None,
            })

    followups = [row for row in rows if row.get("slot") == "followup"]
    cells = defaultdict(list)
    transitions: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in followups:
        key = (row["model"], row["arm"], row["distance"], row["occurrence"])
        cells[key].append(int(row.get("label") == "stick_new"))
        transitions[(row["model"], row["arm"])][row.get("transition", "unknown")] += 1

    with open(args.summary_output, "w", newline="", encoding="utf-8") as fh:
        fields = ["model", "arm", "distance", "occurrence", "accuracy_pct", "n"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key, values in sorted(cells.items()):
            writer.writerow(dict(zip(fields[:4], key)) | {
                "accuracy_pct": round(100 * sum(values) / len(values), 1), "n": len(values),
            })

    diff_rows = []
    paired_keys = {(model, distance, occurrence) for model, arm, distance, occurrence in cells
                   if arm == "conflict"}
    for model, distance, occurrence in sorted(paired_keys):
        conflict = cells.get((model, "conflict", distance, occurrence), [])
        control = cells.get((model, "control", distance, occurrence), [])
        if not conflict or not control:
            continue
        lo, hi = bootstrap_diff_ci(conflict, control, rng)
        diff_rows.append({
            "model": model, "distance": distance, "occurrence": occurrence,
            "conflict_accuracy_pct": round(100 * sum(conflict) / len(conflict), 1),
            "control_accuracy_pct": round(100 * sum(control) / len(control), 1),
            "diff_ci_low": lo, "diff_ci_high": hi,
            "significant_reversion": hi is not None and hi < 0,
            "n_conflict": len(conflict), "n_control": len(control),
        })
    with open(args.diff_output, "w", newline="", encoding="utf-8") as fh:
        fields = ["model", "distance", "occurrence", "conflict_accuracy_pct",
                  "control_accuracy_pct", "diff_ci_low", "diff_ci_high",
                  "significant_reversion", "n_conflict", "n_control"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diff_rows)

    transition_names = sorted({name for counts in transitions.values() for name in counts})
    with open(args.transition_output, "w", newline="", encoding="utf-8") as fh:
        fields = ["model", "arm", *transition_names]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for (model, arm), counts in sorted(transitions.items()):
            writer.writerow({"model": model, "arm": arm, **counts})

    print(f"Eligible follow-ups: {len(followups)}")
    print(f"Wrote {args.gate_output}, {args.summary_output}, {args.diff_output}, {args.transition_output}")


if __name__ == "__main__":
    main()
