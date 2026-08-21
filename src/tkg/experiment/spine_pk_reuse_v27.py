"""Plan and apply factorized PK admission once per certified temporal spine."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tkg.experiment.results import assert_new_output_path


SCHEMA_VERSION = "spine-factorized-pk-reuse-v2.7"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def spine_id(seed: dict[str, Any]) -> str:
    hops = seed.get("hops", [])
    if not isinstance(hops, list) or len(hops) < 3:
        raise ValueError(f"{seed.get('id')}: seed lacks three bridge hops")
    identity = {
        "model_id": seed.get("model_id"),
        "cutoff_date": seed.get("cutoff_date"),
        "target_as_of": seed.get("target_as_of"),
        "bridge_hops": hops[:3],
    }
    return _canonical_hash(identity)


def critical_probe_contract_sha256(case: dict[str, Any]) -> str:
    contract = case.get("prior_knowledge_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{case.get('id')}: missing factorized PK contract")
    probes = contract.get("probes")
    if not isinstance(probes, list):
        raise ValueError(f"{case.get('id')}: PK contract lacks probes")
    primary = [
        probe for probe in probes
        if isinstance(probe, dict) and probe.get("objective") == "must_be_unknown"
    ]
    if not primary:
        raise ValueError(f"{case.get('id')}: no critical bridge admission probes")
    normalized = [{
        "id": probe.get("id"),
        "role": probe.get("role"),
        "objective": probe.get("objective"),
        "question": probe.get("question"),
        "answer_aliases": probe.get("answer_aliases"),
        "event_date": probe.get("event_date"),
        "hop_index": probe.get("hop_index"),
    } for probe in primary]
    return _canonical_hash(normalized)


def plan(
    seeds: Iterable[dict[str, Any]], cases: Iterable[dict[str, Any]],
    *, preferred_representatives: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_by_id = {str(case.get("id")): case for case in cases}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen_seed_contracts: dict[str, tuple[str, str]] = {}
    for seed in seeds:
        seed_id = str(seed.get("id") or "")
        if seed_id in case_by_id:
            bridge_sha = spine_id(seed)
            probe_sha = critical_probe_contract_sha256(case_by_id[seed_id])
            contract = (bridge_sha, probe_sha)
            previous = seen_seed_contracts.get(seed_id)
            if previous is not None:
                if previous != contract:
                    raise ValueError(
                        f"{seed_id}: duplicate seed ID has conflicting spine/PK contract"
                    )
                continue
            seen_seed_contracts[seed_id] = contract
            groups.setdefault((bridge_sha, probe_sha), []).append(seed)
    representatives = []
    rows: list[dict[str, Any]] = []
    preferred = preferred_representatives or set()
    for (certificate_sha, probe_sha), members in sorted(groups.items()):
        members.sort(key=lambda row: (
            str(row["id"]) not in preferred, str(row["id"]),
        ))
        representative = members[0]
        representative_id = str(representative["id"])
        representatives.append(case_by_id[representative_id])
        rows.append({
            "spine_certificate_sha256": certificate_sha,
            "critical_probe_contract_sha256": probe_sha,
            "representative_case_id": representative_id,
            "member_case_ids": [str(row["id"]) for row in members],
            "member_count": len(members),
            "model_id": representative.get("model_id"),
            "cutoff_date": representative.get("cutoff_date"),
        })
    mapping = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "admission_basis": "critical_bridge_only",
        "tail_or_composed_correctness_controls_admission": False,
        "spine_count": len(rows),
        "case_count": sum(row["member_count"] for row in rows),
        "groups": rows,
    }
    return {"cases": representatives, "count": len(representatives)}, mapping


def apply(
    mapping: dict[str, Any], pk_rows: Iterable[dict[str, Any]],
    cases: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gates = {
        str(row.get("case_id")): row for row in pk_rows
        if row.get("slot") == "pk_gate"
    }
    case_by_id = {str(case.get("id")): case for case in cases}
    admitted: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for group in mapping.get("groups", []):
        representative = str(group["representative_case_id"])
        gate = gates.get(representative)
        if gate is None:
            status = "pending_missing_representative_gate"
        elif gate.get("passed") is True:
            status = "admitted"
        else:
            status = "rejected"
        for case_id in group["member_case_ids"]:
            ledger.append({
                "case_id": case_id,
                "spine_certificate_sha256": group["spine_certificate_sha256"],
                "critical_probe_contract_sha256": group[
                    "critical_probe_contract_sha256"
                ],
                "representative_case_id": representative,
                "pk_status": status,
                "pk_gate_reused": case_id != representative,
                "admission_basis": "critical_bridge_only",
            })
            if status == "admitted" and case_id in case_by_id:
                admitted.append(case_by_id[case_id])
    return {
        "schema_version": "pk-admitted-certified-spine-cases-v2.7",
        "cases": admitted,
        "count": len(admitted),
    }, ledger


def _load_rows(paths: Iterable[str], key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        rows.extend(dict(row) for row in payload.get(key, []) if isinstance(row, dict))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--seed-batch", action="append", required=True)
    plan_parser.add_argument("--machine-pass-cases", action="append", required=True)
    plan_parser.add_argument("--representative-cases-output", required=True)
    plan_parser.add_argument("--mapping-output", required=True)
    plan_parser.add_argument(
        "--existing-pk-results",
        help="prefer representatives that already have a recorded pk_gate",
    )
    apply_parser = sub.add_parser("apply")
    apply_parser.add_argument("--mapping", required=True)
    apply_parser.add_argument("--pk-results", required=True)
    apply_parser.add_argument("--machine-pass-cases", action="append", required=True)
    apply_parser.add_argument("--admitted-cases-output", required=True)
    apply_parser.add_argument("--reuse-ledger-output", required=True)
    args = parser.parse_args()

    if args.command == "plan":
        for path in (args.representative_cases_output, args.mapping_output):
            assert_new_output_path(path)
        seeds = _load_rows(args.seed_batch, "seeds")
        cases = _load_rows(args.machine_pass_cases, "cases")
        preferred: set[str] = set()
        if args.existing_pk_results:
            preferred = {
                str(row.get("case_id"))
                for line in Path(args.existing_pk_results).read_text(
                    encoding="utf-8"
                ).splitlines() if line.strip()
                for row in [json.loads(line)]
                if isinstance(row, dict) and row.get("slot") == "pk_gate"
            }
        representatives, mapping = plan(
            seeds, cases, preferred_representatives=preferred,
        )
        Path(args.representative_cases_output).write_text(
            json.dumps(representatives, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        Path(args.mapping_output).write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "machine_pass_cases": mapping["case_count"],
            "pk_calls_required": mapping["spine_count"],
        }, sort_keys=True))
        return 0

    for path in (args.admitted_cases_output, args.reuse_ledger_output):
        assert_new_output_path(path)
    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    pk_rows = [
        json.loads(line) for line in Path(args.pk_results).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = _load_rows(args.machine_pass_cases, "cases")
    admitted, ledger = apply(mapping, pk_rows, cases)
    Path(args.admitted_cases_output).write_text(
        json.dumps(admitted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    Path(args.reuse_ledger_output).write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in ledger
    ), encoding="utf-8")
    print(json.dumps({"admitted": admitted["count"], "ledger_rows": len(ledger)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
