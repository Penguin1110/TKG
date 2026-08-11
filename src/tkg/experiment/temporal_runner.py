"""Run the primary page-by-revision temporal Wikipedia exploration task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from tkg.api.openrouter import OpenRouterError, call_model
from tkg.experiment.case_validation import validate_cases, validate_chain_route
from tkg.experiment.model_cutoffs import model_matches_cutoff
from tkg.experiment.results import JsonlResultStore, assert_new_output_path
from tkg.judging.llm import LLMJudge
from tkg.wikipedia.backend import WikipediaError, WikipediaPageBackend
from tkg.wikipedia.browser import run_temporal_browsing
from tkg.wikipedia.snapshot import (
    configured_pivot, page_version_key, temporal_reverse_bfs,
)


CONTRACT_SCHEMA = "temporal-pk-relative-multihop-v4"


def _question(case: dict) -> str:
    return str(case.get("temporal_question") or "")


def _case_endpoints(case: dict) -> tuple[str | None, str | None]:
    generation = case.get("_generation", {})
    before = case.get("wikipedia_before") or generation.get("before", {}).get(
        "requested_as_of"
    )
    after = case.get("wikipedia_as_of") or generation.get("after", {}).get(
        "requested_as_of"
    )
    return before, after


def _pk_prompt(case: dict, target_as_of: str | None) -> str:
    target = target_as_of or "CURRENT"
    cutoff = case.get("knowledge_cutoff", {}).get("cutoff_date")
    cutoff_line = (
        f"Registered knowledge-cutoff snapshot for this tested model: {cutoff}\n"
        if cutoff else ""
    )
    return (
        f"Question: {_question(case)}\n"
        f"Target date: {target}\n\n"
        f"{cutoff_line}"
        "Answer directly using only your existing knowledge. Do not browse, use tools, or "
        "assume access to external sources. If you do not know, say so briefly."
    )


def run_pk_admission(
    *,
    case: dict,
    model: str,
    target_as_of: str | None,
    judge: LLMJudge,
    store: JsonlResultStore,
    repeats: int,
    max_known_rate: float,
    probe_call_model_fn=call_model,
) -> dict:
    """Admit a case-model pair only when the target answer is not already known.

    Every probe is a fresh one-turn conversation.  Probe responses are logged but
    never inserted into the later navigation conversation.
    """
    prompt = _pk_prompt(case, target_as_of)
    labels: list[str] = []
    for probe_repeat in range(repeats):
        response = probe_call_model_fn(
            model, [{"role": "user", "content": prompt}], temperature=0.0,
        )
        judgment = judge.judge_answer(
            _question(case), response,
            case.get("new_answer_keywords", []),
            case.get("old_answer_keywords", []),
            [],
        )
        labels.append(judgment.decision)
        store.write(
            slot="pk_probe", case_id=case["id"], model=model, arm="admission",
            probe_repeat=probe_repeat, target_snapshot_as_of=target_as_of,
            prompt=prompt, response=response, judgment=judgment.to_dict(),
            label=judgment.decision, fresh_context=True, tools_available=False,
        )

    label_counts = {label: labels.count(label) for label in sorted(set(labels))}
    known_count = label_counts.get("stick_new", 0)
    stale_count = label_counts.get("stick_old", 0)
    unjudgeable_count = label_counts.get("unjudgeable", 0)
    known_rate = known_count / repeats
    passed = unjudgeable_count == 0 and known_rate <= max_known_rate
    if unjudgeable_count:
        reason = "unjudgeable_pk_probe"
    elif known_rate > max_known_rate:
        reason = "already_knows_target_answer"
    else:
        reason = "target_answer_not_known"
    gate = {
        "n": repeats,
        "label_counts": label_counts,
        "stick_new_count": known_count,
        "stick_new_rate": known_rate,
        "stick_old_count": stale_count,
        "stick_old_rate": stale_count / repeats,
        "other_count": repeats - known_count - stale_count,
        "max_known_rate": max_known_rate,
        "passed": passed,
        "reason": reason,
        "target_snapshot_as_of": target_as_of,
    }
    store.write(
        slot="pk_gate", case_id=case["id"], model=model, arm="admission", **gate,
    )
    return gate


def _snapshot_values(raw: str | None, case: dict) -> list[str | None]:
    tokens = [item.strip() for item in (raw or "").split(",") if item.strip()]
    if not tokens:
        before, after = _case_endpoints(case)
        required = case.get("required_snapshot_dates", [])
        tokens = [str(value) for value in [before, *required, after] if value]
    values: list[str | None] = []
    for token in tokens:
        value = None if token.upper() == "CURRENT" else token
        if value is not None:
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"invalid snapshot date: {value}") from exc
        if value not in values:
            values.append(value)
    if len(values) < 2:
        raise ValueError(
            "temporal exploration needs at least two --snapshot-dates, or generated "
            "before/after metadata in the case"
        )
    before, after = _case_endpoints(case)
    required_dates = [
        (f"required[{index}]", value)
        for index, value in enumerate(case.get("required_snapshot_dates", []))
    ]
    for label, required in (("before", before), *required_dates, ("target", after)):
        if required is not None and required not in values:
            raise ValueError(
                f"{case.get('id', '<missing-id>')}: {label} snapshot {required!r} "
                "must be included in --snapshot-dates"
            )
    return values


def _contract_hash(cases: list[dict], args) -> str:
    payload = {
        "schema": CONTRACT_SCHEMA,
        "cases": cases,
        "models": args.models,
        "judge_model": args.judge_model,
        "judge_min_confidence": args.judge_min_confidence,
        "snapshot_dates": args.snapshot_dates,
        "repeats": args.repeats,
        "max_steps": args.max_steps,
        "start_distance": args.start_distance,
        "backlink_branch_cap": args.backlink_branch_cap,
        "pk_repeats": args.pk_repeats,
        "pk_max_known_rate": args.pk_max_known_rate,
        "temperature": args.temperature,
        "lang": args.lang,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _choose_start_title(
    navigation: dict, case_id: str, repeat: int, start_distance: int
) -> str:
    candidates = navigation["candidate_titles_by_distance"].get(
        str(start_distance), []
    )
    if not candidates:
        available = sorted(
            int(value) for value, titles in
            navigation["candidate_titles_by_distance"].items() if titles
        )
        raise ValueError(
            f"{case_id}: no start page at exact distance {start_distance}; "
            f"available distances={available}"
        )
    ordered = sorted(candidates, key=str.casefold)
    digest = hashlib.sha256(f"{case_id}|{repeat}".encode("utf-8")).digest()
    return ordered[int.from_bytes(digest[:8], "big") % len(ordered)]


def _navigation_id(navigation: dict) -> str:
    payload = {
        "target_key": navigation["target_key"],
        "allowed_as_of": navigation["allowed_as_of"],
        "states": navigation["states"],
        "arena_edges": navigation["arena_edges"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _navigation_metrics(
    result: dict, navigation: dict, start_title: str
) -> dict:
    distances = navigation["distances"]
    target_key = navigation["target_key"]
    first_state_key = next((
        page_version_key(str(record.get("to_title", "")), record["revision_id"])
        for record in result["trajectory"]
        if record.get("action") == "switch_snapshot"
        and not str(record.get("result", "")).startswith("Error:")
        and record.get("revision_id") is not None
    ), None)
    if first_state_key is None:
        start_state_distances = [
            state["distance_to_pivot"] for state in navigation["states"].values()
            if state["title"].casefold() == start_title.casefold()
        ]
    else:
        first_distance = navigation["distances"].get(first_state_key)
        start_state_distances = [] if first_distance is None else [first_distance]
    if not start_state_distances:
        raise ValueError(f"start title {start_title!r} is absent from navigation graph")
    # The first successful switch_snapshot binds the initially timeless page
    # title to a revision, so it is one navigation step in the measured state graph.
    shortest_navigation_steps = 1 + min(start_state_distances)
    navigation_steps = 0
    seen_versions: set[str] = set()
    revisit_count = 0
    first_target_navigation_steps: int | None = None
    first_target_tool_step: int | None = None
    distance_trace: list[dict] = []
    for record in result["trajectory"]:
        action = record.get("action")
        success = (
            action in {"switch_snapshot", "follow_link"}
            and not str(record.get("result", "")).startswith("Error:")
            and record.get("revision_id") is not None
        )
        if not success:
            continue
        navigation_steps += 1
        key = page_version_key(str(record.get("to_title", "")), record["revision_id"])
        revisited = key in seen_versions
        if revisited:
            revisit_count += 1
        seen_versions.add(key)
        distance = distances.get(key)
        if distance is None:
            raise ValueError(
                f"trajectory entered page-version {key!r} outside navigation arena"
            )
        record["navigation_step"] = navigation_steps
        record["distance_to_pivot"] = distance
        record["revisited"] = revisited
        distance_trace.append({
            "navigation_step": navigation_steps,
            "tool_step": record.get("step"),
            "title": record.get("to_title"),
            "revision_id": record.get("revision_id"),
            "snapshot_token": record.get("snapshot_token"),
            "distance_to_pivot": distance,
            "revisited": revisited,
        })
        if key == target_key and first_target_navigation_steps is None:
            first_target_navigation_steps = navigation_steps
            first_target_tool_step = record.get("step")
    detour_steps = (
        first_target_navigation_steps - shortest_navigation_steps
        if first_target_navigation_steps is not None else None
    )
    if detour_steps is not None and detour_steps < 0:
        raise ValueError(
            "trajectory beat the recorded minimum; navigation arena contract is inconsistent"
        )
    return {
        "pivot_hit": first_target_navigation_steps is not None,
        "shortest_navigation_steps": shortest_navigation_steps,
        "actual_steps_to_first_pivot": first_target_navigation_steps,
        "first_pivot_tool_step": first_target_tool_step,
        "detour_steps": detour_steps,
        "shortest_arrival": detour_steps == 0,
        "navigation_steps_total": navigation_steps,
        "revisit_count": revisit_count,
        "cycle_detected": revisit_count > 0,
        "distance_trace": distance_trace,
    }


def _completed(
    path: str, contract_hash: str
) -> dict[tuple[str, str, int], str | None]:
    completed: dict[tuple[str, str, int], str | None] = {}
    file = Path(path)
    if not file.is_file():
        return completed
    with file.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (row.get("slot") == "checkpoint" and row.get("status") == "complete"
                    and row.get("contract_hash") == contract_hash
                    and row.get("arm") == "temporal"):
                completed[(
                    str(row.get("case_id")), str(row.get("model")), int(row.get("repeat", 0))
                )] = row.get("navigation_id")
    return completed


def _recorded_navigation_ids(path: str, contract_hash: str) -> set[str]:
    recorded: set[str] = set()
    file = Path(path)
    if not file.is_file():
        return recorded
    with file.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (row.get("slot") == "navigation_arena"
                    and row.get("contract_hash") == contract_hash
                    and row.get("navigation_id")):
                recorded.add(str(row["navigation_id"]))
    return recorded


def run_case(
    *,
    case: dict,
    model: str,
    repeat: int,
    backend,
    judge: LLMJudge,
    store: JsonlResultStore,
    snapshot_dates: list[str | None],
    navigation: dict,
    start_distance: int,
    max_steps: int,
    temperature: float,
    browse_call_model_fn=None,
) -> str | None:
    target = configured_pivot(case, backend)
    if not target:
        raise ValueError(f"{case['id']}: no Wikipedia pivot title")
    _, configured_after = _case_endpoints(case)
    target_as_of = configured_after if configured_after is not None else snapshot_dates[-1]
    if target_as_of not in snapshot_dates:
        raise ValueError(f"{case['id']}: target snapshot is unavailable")
    target_page = backend.fetch_page(target, as_of=target_as_of)
    if page_version_key(target_page.title, target_page.revision_id) != navigation["target_key"]:
        raise ValueError(f"{case['id']}: navigation target does not match target revision")
    chain_contract = validate_chain_route(case, navigation)
    if chain_contract:
        start_title = chain_contract["start_title"]
        start_distance = chain_contract["distance"]
    else:
        start_title = _choose_start_title(
            navigation, case["id"], repeat, start_distance
        )
    attempt_id = uuid.uuid4().hex
    kwargs = {}
    if browse_call_model_fn is not None:
        kwargs["call_model_fn"] = browse_call_model_fn
    result = run_temporal_browsing(
        model=model,
        backend=backend,
        start_title=start_title,
        question=_question(case),
        allowed_as_of=snapshot_dates,
        max_steps=max_steps,
        target_title=target_page.title,
        target_as_of=target_as_of,
        allowed_version_keys=set(navigation["allowed_version_keys"]),
        allowed_titles_by_snapshot=navigation["allowed_titles_by_snapshot"],
        reveal_target_title=not bool(case.get("hide_pivot_title")),
        cutoff_reference=case.get("knowledge_cutoff", {}).get("cutoff_date"),
        temperature=temperature,
        **kwargs,
    )
    metrics = _navigation_metrics(result, navigation, start_title)
    for record in result["trajectory"]:
        store.write(
            slot="temporal_step", case_id=case["id"], model=model, arm="temporal",
            attempt_id=attempt_id, repeat=repeat, start_title=start_title, **record,
        )
    store.write(
        slot="temporal_summary", case_id=case["id"], model=model, arm="temporal",
        attempt_id=attempt_id, repeat=repeat, start_title=start_title,
        start_distance=start_distance, target_title=target_page.title,
        target_revision_id=target_page.revision_id, question=_question(case),
        reasoning_hop_count=case.get("reasoning_hop_count"),
        reasoning_chain=case.get("reasoning_chain", []),
        knowledge_cutoff=case.get("knowledge_cutoff"),
        navigation_id=_navigation_id(navigation),
        allowed_as_of=snapshot_dates, target_snapshot_as_of=target_as_of,
        final_title=result["final_title"],
        final_snapshot_token=result["final_snapshot_token"],
        final_answer=result["final_answer"], stop_reason=result["stop_reason"],
        target_title_revealed=result["target_title_revealed"],
        visited_versions=result["visited_versions"],
        **metrics,
        evidence_revisions=[
            {
                "title": page["title"], "revision_id": page["revision_id"],
                "timestamp": page["timestamp"], "as_of": page.get("as_of"),
            }
            for page in result["evidence_pages"]
        ],
    )
    if result["stop_reason"] == "error":
        return None
    judgment = judge.judge_temporal_answer(
        _question(case), result["final_answer"],
        case.get("new_answer_keywords", []), case.get("old_answer_keywords", []),
        result["evidence_pages"], target_snapshot_as_of=target_as_of,
    )
    store.write(
        slot="final_judgment", case_id=case["id"], model=model, arm="temporal",
        attempt_id=attempt_id, repeat=repeat, question=_question(case),
        response=result["final_answer"], judgment=judgment.to_dict(),
        label=judgment.decision, target_snapshot_as_of=target_as_of,
    )
    return attempt_id


def write_scores(result_path: str, output_path: str, contract_hash: str) -> None:
    assert_new_output_path(output_path)
    completed_attempts = set()
    judgments = []
    summaries: dict[str, dict] = {}
    rows = []
    if Path(result_path).is_file():
        with open(result_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("contract_hash") == contract_hash:
                    rows.append(row)
        completed_attempts = {
            row.get("attempt_id") for row in rows
            if row.get("slot") == "checkpoint" and row.get("status") == "complete"
        }
        judgments = [
            row for row in rows
            if row.get("slot") == "final_judgment"
            and row.get("attempt_id") in completed_attempts
        ]
        summaries = {
            str(row.get("attempt_id")): row for row in rows
            if row.get("slot") == "temporal_summary"
            and row.get("attempt_id") in completed_attempts
        }
    pk_gates: dict[tuple[str, str], dict] = {}
    for row in rows:
        if row.get("slot") == "pk_gate":
            pk_gates[(str(row.get("model")), str(row.get("case_id")))] = row
    counts: dict[tuple[str, str], dict[str, int]] = {
        key: {
            "n": 0, "correct_after": 0, "old_snapshot_answer": 0,
            "pivot_hit": 0, "shortest_arrival": 0, "cycle_detected": 0,
            "found_but_wrong": 0, "detour_sum": 0, "detour_n": 0,
        }
        for key in pk_gates
    }
    for row in judgments:
        key = (str(row.get("model")), str(row.get("case_id")))
        bucket = counts.setdefault(key, {
            "n": 0, "correct_after": 0, "old_snapshot_answer": 0,
            "pivot_hit": 0, "shortest_arrival": 0, "cycle_detected": 0,
            "found_but_wrong": 0, "detour_sum": 0, "detour_n": 0,
        })
        bucket["n"] += 1
        label = str(row.get("label"))
        if label in bucket:
            bucket[label] += 1
        summary = summaries.get(str(row.get("attempt_id")), {})
        pivot_hit = bool(summary.get("pivot_hit"))
        bucket["pivot_hit"] += int(pivot_hit)
        bucket["shortest_arrival"] += int(bool(summary.get("shortest_arrival")))
        bucket["cycle_detected"] += int(bool(summary.get("cycle_detected")))
        bucket["found_but_wrong"] += int(pivot_hit and label != "correct_after")
        if summary.get("detour_steps") is not None:
            bucket["detour_sum"] += int(summary["detour_steps"])
            bucket["detour_n"] += 1
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        fields = [
            "model", "case_id", "pk_admitted", "pk_gate_reason", "pk_probe_n",
            "pk_stick_new", "pk_stick_new_rate_pct", "pk_stick_old",
            "pk_stick_old_rate_pct", "pk_other", "n", "correct_after",
            "correct_after_rate_pct",
            "old_snapshot_answer", "old_snapshot_rate_pct",
            "pivot_hit", "pivot_hit_rate_pct", "shortest_arrival",
            "shortest_arrival_rate_pct", "mean_detour_steps_on_hit",
            "cycle_detected", "cycle_rate_pct", "found_but_wrong",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for (model, case_id), bucket in sorted(counts.items()):
            n = bucket["n"]
            pk_gate = pk_gates.get((model, case_id), {})
            pk_n = int(pk_gate.get("n", 0) or 0)
            writer.writerow({
                "model": model, "case_id": case_id, "n": n,
                "pk_admitted": pk_gate.get("passed", ""),
                "pk_gate_reason": pk_gate.get("reason", "missing_pk_gate"),
                "pk_probe_n": pk_n,
                "pk_stick_new": pk_gate.get("stick_new_count", ""),
                "pk_stick_new_rate_pct": (
                    round(100 * float(pk_gate.get("stick_new_rate", 0)), 1)
                    if pk_n else ""
                ),
                "pk_stick_old": pk_gate.get("stick_old_count", ""),
                "pk_stick_old_rate_pct": (
                    round(100 * float(pk_gate.get("stick_old_rate", 0)), 1)
                    if pk_n else ""
                ),
                "pk_other": pk_gate.get("other_count", ""),
                "correct_after": bucket["correct_after"],
                "correct_after_rate_pct": (
                    round(100 * bucket["correct_after"] / n, 1) if n else ""
                ),
                "old_snapshot_answer": bucket["old_snapshot_answer"],
                "old_snapshot_rate_pct": (
                    round(100 * bucket["old_snapshot_answer"] / n, 1) if n else ""
                ),
                "pivot_hit": bucket["pivot_hit"],
                "pivot_hit_rate_pct": (
                    round(100 * bucket["pivot_hit"] / n, 1) if n else ""
                ),
                "shortest_arrival": bucket["shortest_arrival"],
                "shortest_arrival_rate_pct": round(
                    100 * bucket["shortest_arrival"] / n, 1
                ) if n else "",
                "mean_detour_steps_on_hit": (
                    round(bucket["detour_sum"] / bucket["detour_n"], 2)
                    if bucket["detour_n"] else ""
                ),
                "cycle_detected": bucket["cycle_detected"],
                "cycle_rate_pct": (
                    round(100 * bucket["cycle_detected"] / n, 1) if n else ""
                ),
                "found_but_wrong": bucket["found_but_wrong"],
            })


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explore Wikipedia hyperlinks and page revisions across time"
    )
    parser.add_argument("--models", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-min-confidence", type=float, default=0.8)
    parser.add_argument("--cases", default="generated_cases.json")
    parser.add_argument("--case-ids")
    parser.add_argument(
        "--snapshot-dates",
        help="comma-separated dates available to switch_snapshot; CURRENT is allowed",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument(
        "--start-distance", type=int, default=3,
        help="minimum state transitions from the selected start page to target pivot",
    )
    parser.add_argument(
        "--backlink-branch-cap", type=int, default=25,
        help="maximum verified backlink predecessors expanded per page-version state",
    )
    parser.add_argument(
        "--pk-repeats", type=int, default=3,
        help="fresh-context prior-knowledge probes per case-model pair",
    )
    parser.add_argument(
        "--pk-max-known-rate", type=float, default=0.0,
        help="maximum fraction of PK probes allowed to answer with the target value",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--cache-path", default="wikipedia_snapshot.db")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", default="temporal_results.jsonl")
    parser.add_argument("--score-output", default="temporal_scores.csv")
    args = parser.parse_args()

    assert_new_output_path(args.output)
    assert_new_output_path(args.score_output)
    if args.repeats <= 0 or args.max_steps <= 0 or args.pk_repeats <= 0:
        parser.error("--repeats, --max-steps, and --pk-repeats must be > 0")
    if args.start_distance <= 0 or args.backlink_branch_cap <= 0:
        parser.error("--start-distance and --backlink-branch-cap must be > 0")
    if args.max_steps < args.start_distance + 2:
        parser.error(
            "--max-steps must allow the initial time selection, shortest path, and answer"
        )
    if not 0 <= args.judge_min_confidence <= 1:
        parser.error("--judge-min-confidence must be between 0 and 1")
    if not 0 <= args.pk_max_known_rate <= 1:
        parser.error("--pk-max-known-rate must be between 0 and 1")
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if not models:
        parser.error("--models must not be empty")
    if args.judge_model in models:
        parser.error("--judge-model must differ from every tested model")
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[error] OPENROUTER_API_KEY is required", file=sys.stderr)
        return 2
    with open(args.cases, "r", encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]
    if args.case_ids:
        wanted = {item.strip() for item in args.case_ids.split(",") if item.strip()}
        cases = [case for case in cases if case.get("id") in wanted]
    errors = validate_cases(cases)
    if errors:
        print("\n".join(f"[config error] {error}" for error in errors), file=sys.stderr)
        return 2
    required_max_steps = max(
        (int(case.get("expected_navigation_distance", args.start_distance)) + 2
         for case in cases),
        default=args.start_distance + 2,
    )
    if args.max_steps < required_max_steps:
        parser.error(
            f"--max-steps must be >= {required_max_steps} for the deepest declared "
            "reasoning chain, initial time selection, and answer"
        )
    # Validate all per-case date sets before opening the append-only result file.
    dates_by_case = {case["id"]: _snapshot_values(args.snapshot_dates, case) for case in cases}
    contract_hash = _contract_hash(cases, args)
    completed = _completed(args.output, contract_hash)
    recorded_navigation_ids = _recorded_navigation_ids(args.output, contract_hash)
    backend = WikipediaPageBackend(
        cache_path=args.cache_path, lang=args.lang, offline_only=args.offline
    )
    judge = LLMJudge(args.judge_model, min_confidence=args.judge_min_confidence)
    store = JsonlResultStore(args.output, metadata={
        "contract_hash": contract_hash,
        "judge_model": args.judge_model,
        "experiment": "pk_admitted_temporal_shortest_path_navigation",
        "schema_version": CONTRACT_SCHEMA,
    })
    try:
        for case in cases:
            _, configured_after = _case_endpoints(case)
            target_as_of = (
                configured_after if configured_after is not None
                else dates_by_case[case["id"]][-1]
            )
            compatible_models = [
                model for model in models if model_matches_cutoff(case, model)
            ]
            for model in models:
                if model not in compatible_models:
                    print(
                        f"[skip] {case['id']}/{model}: cutoff-relative case is bound "
                        "to another exact model ID"
                    )
            if not compatible_models:
                print(f"[skip] {case['id']}: no model matches its cutoff contract")
                continue
            # Build and validate the graph contract before any paid PK/judge calls.
            # A disconnected or non-shortest relation chain is not a valid question.
            target = configured_pivot(case, backend)
            if not target:
                print(f"[error] {case['id']}: no Wikipedia pivot title")
                continue
            try:
                graph_depth = max(
                    args.start_distance,
                    int(case.get("expected_navigation_distance", 0)),
                )
                navigation = temporal_reverse_bfs(
                    backend, target, target_as_of, dates_by_case[case["id"]],
                    graph_depth, branch_cap=args.backlink_branch_cap,
                )
                if case.get("reasoning_chain"):
                    validate_chain_route(case, navigation)
                else:
                    _choose_start_title(
                        navigation, case["id"], 0, args.start_distance
                    )
            except (WikipediaError, ValueError) as exc:
                print(f"[error] {case['id']}: navigation graph unavailable: {exc}")
                continue
            admitted_models = []
            for model in compatible_models:
                cached_gate = JsonlResultStore.latest_pk_gate(
                    args.output, case["id"], model, contract_hash,
                )
                if cached_gate is None:
                    try:
                        gate = run_pk_admission(
                            case=case, model=model, target_as_of=target_as_of,
                            judge=judge, store=store, repeats=args.pk_repeats,
                            max_known_rate=args.pk_max_known_rate,
                        )
                        admitted = bool(gate["passed"])
                        print(
                            f"[PK {'pass' if admitted else 'reject'}] "
                            f"{case['id']}/{model}: new={gate['stick_new_count']}/"
                            f"{gate['n']} old={gate['stick_old_count']}/{gate['n']} "
                            f"reason={gate['reason']}"
                        )
                    except (OpenRouterError, ValueError) as exc:
                        print(f"[error] PK admission {case['id']}/{model}: {exc}")
                        admitted = False
                else:
                    admitted = cached_gate
                    print(
                        f"[PK checkpoint] {case['id']}/{model}: "
                        f"{'admitted' if admitted else 'rejected'}"
                    )
                if admitted:
                    admitted_models.append(model)
                else:
                    print(f"[PK reject] {case['id']}/{model}: unknown-knowledge gate failed")
            if not admitted_models:
                print(f"[skip] {case['id']}: no model passed PK admission")
                continue
            navigation_id = _navigation_id(navigation)
            if navigation_id not in recorded_navigation_ids:
                store.write(
                    slot="navigation_arena", case_id=case["id"], model="__shared__",
                    arm="temporal", navigation_id=navigation_id,
                    target=navigation["target"],
                    allowed_as_of=navigation["allowed_as_of"],
                    states=list(navigation["states"].values()),
                    arena_edges=navigation["arena_edges"],
                    admitted_models=admitted_models,
                    coverage_note=navigation["coverage_note"],
                )
                recorded_navigation_ids.add(navigation_id)
            for model in admitted_models:
                for repeat in range(args.repeats):
                    checkpoint = (case["id"], model, repeat)
                    if completed.get(checkpoint) == navigation_id:
                        print(f"[checkpoint] skip {checkpoint}")
                        continue
                    try:
                        attempt_id = run_case(
                            case=case, model=model, repeat=repeat, backend=backend,
                            judge=judge, store=store,
                            snapshot_dates=dates_by_case[case["id"]],
                            navigation=navigation,
                            start_distance=args.start_distance,
                            max_steps=args.max_steps, temperature=args.temperature,
                        )
                    except (WikipediaError, ValueError) as exc:
                        print(f"[error] {checkpoint}: {exc}")
                        attempt_id = None
                    if attempt_id:
                        store.write(
                            slot="checkpoint", case_id=case["id"], model=model,
                            arm="temporal", repeat=repeat, status="complete",
                            attempt_id=attempt_id, navigation_id=navigation_id,
                        )
    finally:
        store.close()
        backend.close()
    write_scores(args.output, args.score_output, contract_hash)
    print(f"Done: {args.output}; scores: {args.score_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
