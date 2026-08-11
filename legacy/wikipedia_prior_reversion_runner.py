"""Archived Wikipedia prior-reversion runner; not used by the primary CLI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import uuid
from collections import defaultdict
from pathlib import Path

from tkg.api.openrouter import OpenRouterError, call_model
from tkg.experiment.case_validation import validate_cases
from tkg.experiment.results import JsonlResultStore, assert_new_output_path
from legacy.wikipedia_prior_reversion_gates import answerability_for_item, evaluate_exposure
from legacy.wikipedia_prior_reversion_schedule import build_round_schedule
from tkg.judging.llm import LLMJudge, transition_label
from tkg.wikipedia.backend import WikipediaError, WikipediaPageBackend, reverse_bfs_frontier
from tkg.wikipedia.browser import run_snapshot_selection, run_wikipedia_browsing
from tkg.wikipedia.snapshot import configured_target


NEUTRAL_TASK = (
    "Browse Wikipedia to gather useful background information. Choose links based on what seems "
    "informative. You are not told that any particular page or fact is a target."
)


def _seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return base + int.from_bytes(digest[:8], "big")


def _contract_hash(cases: list[dict], args) -> str:
    contract = {
        "schema": "wikipedia-v1", "cases": cases, "judge_model": args.judge_model,
        "judge_min_confidence": args.judge_min_confidence,
        "as_of": args.as_of, "arms": args.arms, "distances": args.distances,
        "n_starts_per_distance": args.n_starts_per_distance,
        "repeats_per_start": args.repeats_per_start, "max_steps": args.max_steps,
        "temperature": args.temperature, "rounds": args.rounds,
        "distractor_every": args.distractor_every, "pk_repeats": args.pk_repeats,
        "branch_cap": args.branch_cap, "lang": args.lang,
        "snapshot_mode": args.snapshot_mode, "snapshot_dates": args.snapshot_dates,
    }
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _first_question(pool: dict) -> dict | None:
    for distance in sorted(pool, key=int):
        if pool[distance]:
            return pool[distance][0]
    return None


def case_for_arm(case: dict, arm: str) -> dict:
    if arm == "conflict":
        return case
    first = _first_question(case.get("control", {}))
    if not first:
        raise ValueError("control arm 沒有 control questions")
    return {
        **case,
        "pk_question": case.get("control_pk_question", first["question"]),
        "new_answer_keywords": case.get(
            "control_pk_answer_keywords", first.get("answer_keywords", [])
        ),
        "old_answer_keywords": [],
    }


def _pool(case: dict, arm: str) -> dict:
    return case["ripples"] if arm == "conflict" else case["control"]


def run_pk_gate(case, model, judge, store, repeats: int) -> bool:
    labels = []
    for repeat in range(repeats):
        response = call_model(model, [{"role": "user", "content": case["pk_question"]}])
        judgment = judge.judge_answer(
            case["pk_question"], response, case.get("new_answer_keywords", []),
            case.get("old_answer_keywords", []), [],
        )
        labels.append(judgment.decision)
        store.write(
            slot="pk_probe", case_id=case["id"], model=model, arm="conflict",
            repeat=repeat, question=case["pk_question"], response=response,
            judgment=judgment.to_dict(), label=judgment.decision,
        )
    rate = labels.count("stick_old") / len(labels) if labels else 0.0
    passed = rate >= case.get("pk_threshold", 0.8)
    store.write(
        slot="pk_gate", case_id=case["id"], model=model, arm="conflict",
        n=len(labels), stick_old_rate=rate, threshold=case.get("pk_threshold", 0.8),
        passed=passed,
    )
    return passed


def _log_browsing(
    store, case, model, arm, distance, repeat, result, attempt_id, as_of,
    snapshot_selection=None, target_snapshot_as_of=None, target_title=None,
):
    for record in result["trajectory"]:
        store.write(
            slot="browse_step", case_id=case["id"], model=model, arm=arm,
            start_distance=distance, start_title=result["start_title"], repeat=repeat,
            attempt_id=attempt_id,
            **record,
        )
    store.write(
        slot="browse_summary", case_id=case["id"], model=model, arm=arm,
        start_distance=distance, start_title=result["start_title"], repeat=repeat,
        attempt_id=attempt_id,
        final_title=result["final_title"], page_hit=result["page_hit"],
        stop_reason=result["stop_reason"], visited_titles=result["visited_titles"],
        target_title=target_title,
        snapshot_as_of=as_of,
        snapshot_selection_id=(
            snapshot_selection.get("selection_id") if snapshot_selection else None
        ),
        snapshot_mode=(
            snapshot_selection.get("snapshot_mode") if snapshot_selection else "prebound"
        ),
        target_snapshot_as_of=target_snapshot_as_of,
        evidence_revisions=[
            {"title": page["title"], "revision_id": page["revision_id"],
             "timestamp": page["timestamp"]} for page in result["evidence_pages"]
        ],
    )


def _answerable_items(case, arm, pages, judge, store, model, trajectory_key, attempt_id):
    eligible = defaultdict(list)
    for distance_text, items in _pool(case, arm).items():
        distance = int(distance_text)
        for index, item in enumerate(items):
            judgment = answerability_for_item(judge, item, pages, arm)
            item_key = f"d{distance}_i{index}"
            store.write(
                slot="answerability_gate", case_id=case["id"], model=model, arm=arm,
                trajectory_key=trajectory_key, distance=distance, item_key=item_key,
                attempt_id=attempt_id,
                question=item["question"], judgment=judgment.to_dict(),
                passed=judgment.decision == "answerable",
                pre_seen_direct=judgment.raw.get("support_type") == "explicit",
            )
            if judgment.decision == "answerable":
                eligible[distance].append((item_key, item))
    return eligible


def _run_followups(
    case, arm, model, result, gate, eligible, judge, store, trajectory_key,
    distractors, rounds, distractor_every, rng, pivot_question,
    attempt_id, plain_call_model_fn=call_model,
):
    messages = list(result["messages"])
    immediate_prompt = (
        "Browsing is now paused. Based on the pages you just read, answer this question directly "
        f"and briefly: {pivot_question}"
    )
    messages.extend([
        {"role": "user", "content": immediate_prompt},
        {"role": "assistant", "content": gate.comprehension_response},
    ])
    distances = sorted(eligible)
    schedule = build_round_schedule(distances, rounds, distractor_every, rng)
    occurrence: dict[int, int] = defaultdict(int)
    previous_by_item: dict[str, str] = {}

    for round_index, slot in enumerate(schedule, start=1):
        if slot == "distractor":
            question = rng.choice(distractors)
            messages.append({"role": "user", "content": question})
            response = plain_call_model_fn(model, messages)
            messages.append({"role": "assistant", "content": response})
            store.write(
                slot="distractor", case_id=case["id"], model=model, arm=arm,
                trajectory_key=trajectory_key, round=round_index,
                attempt_id=attempt_id,
                question=question, response=response,
            )
            continue

        distance = int(slot)
        occurrence[distance] += 1
        item_key, item = rng.choice(eligible[distance])
        question = rng.choice([item["question"], *item.get("paraphrases", [])])
        messages.append({"role": "user", "content": question})
        response = plain_call_model_fn(model, messages)
        messages.append({"role": "assistant", "content": response})
        if arm == "conflict":
            new_answers = item.get("new_keywords", [])
            old_answers = item.get("old_keywords", [])
        else:
            new_answers = item.get("answer_keywords", [])
            old_answers = []
        judgment = judge.judge_answer(
            question, response, new_answers, old_answers, result["evidence_pages"]
        )
        previous = previous_by_item.get(item_key)
        transition = transition_label(previous, judgment.decision)
        previous_by_item[item_key] = judgment.decision
        store.write(
            slot="followup", case_id=case["id"], model=model, arm=arm,
            trajectory_key=trajectory_key, round=round_index, distance=distance,
            attempt_id=attempt_id,
            occurrence=occurrence[distance], item_key=item_key, question=question,
            response=response, label=judgment.decision, transition=transition,
            judgment=judgment.to_dict(),
        )


def run_trajectory(
    *, case, raw_case, arm, model, backend, target, as_of, start_title, distance, repeat,
    args, judge, store, rng, browse_call_model_fn=None,
    plain_call_model_fn=call_model,
    snapshot_selection=None, target_snapshot_as_of=None,
):
    attempt_id = uuid.uuid4().hex
    trajectory_key = f"{case['id']}|{model}|{arm}|d{distance}|{start_title}|r{repeat}"
    browse_kwargs = {}
    if browse_call_model_fn is not None:
        browse_kwargs["call_model_fn"] = browse_call_model_fn
    if snapshot_selection is not None:
        browse_kwargs["initial_messages"] = snapshot_selection["messages"]
    result = run_wikipedia_browsing(
        model=model, backend=backend, start_title=start_title, max_steps=args.max_steps,
        task_prompt=NEUTRAL_TASK, target_title=target, as_of=as_of,
        temperature=args.temperature, **browse_kwargs,
    )
    _log_browsing(
        store, case, model, arm, distance, repeat, result, attempt_id, as_of,
        snapshot_selection=snapshot_selection,
        target_snapshot_as_of=target_snapshot_as_of,
        target_title=target,
    )
    if result["stop_reason"] == "error":
        return None

    gate = evaluate_exposure(
        page_hit=result["page_hit"], case=case, pages=result["evidence_pages"],
        messages=result["messages"], tested_model=model, judge=judge,
        call_model_fn=plain_call_model_fn,
    )
    store.write(
        slot="exposure_gate", case_id=case["id"], model=model, arm=arm,
        trajectory_key=trajectory_key, start_distance=distance,
        start_title=start_title, repeat=repeat, gate=gate.to_dict(),
        attempt_id=attempt_id,
    )
    if gate.eligible:
        eligible = _answerable_items(
            raw_case, arm, result["evidence_pages"], judge, store, model, trajectory_key,
            attempt_id,
        )
        if eligible:
            _run_followups(
                raw_case, arm, model, result, gate, eligible, judge, store,
                trajectory_key, args.distractors, args.rounds, args.distractor_every, rng,
                pivot_question=case["pk_question"],
                attempt_id=attempt_id,
                plain_call_model_fn=plain_call_model_fn,
            )
        else:
            store.write(
                slot="line_b_skipped", case_id=case["id"], model=model, arm=arm,
                trajectory_key=trajectory_key, reason="no_answerable_followups",
                attempt_id=attempt_id,
            )
    return attempt_id


def write_hit_rates(result_path: str, csv_path: str, contract_hash: str | None = None):
    assert_new_output_path(csv_path)
    counts: dict[tuple, list[int]] = defaultdict(
        lambda: [0, 0, 0]
    )  # attempts, hits, eligible exposure
    if Path(result_path).exists():
        rows = []
        with open(result_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if contract_hash is not None and row.get("contract_hash") != contract_hash:
                    continue
                rows.append(row)
        completed_attempts = {
            row.get("attempt_id") for row in rows
            if row.get("slot") == "checkpoint" and row.get("status") == "complete"
        }
        for row in rows:
            if row.get("slot") in {"browse_summary", "exposure_gate"} and (
                not row.get("attempt_id") or row.get("attempt_id") not in completed_attempts
            ):
                continue
            key = (row.get("model"), row.get("case_id"), row.get("arm"),
                   row.get("start_distance"))
            if row.get("slot") == "browse_summary":
                counts[key][0] += 1
                counts[key][1] += int(bool(row.get("page_hit")))
            elif row.get("slot") == "exposure_gate" and row.get("gate", {}).get("eligible"):
                counts[key][2] += 1
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        fields = ["model", "case_id", "arm", "start_distance", "attempts", "page_hits",
                  "page_hit_rate_pct", "eligible_exposures", "eligible_rate_pct"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key, (attempts, hits, eligible) in sorted(counts.items(), key=lambda item: str(item[0])):
            writer.writerow(dict(zip(fields[:4], key)) | {
                "attempts": attempts, "page_hits": hits,
                "page_hit_rate_pct": round(100 * hits / attempts, 1) if attempts else None,
                "eligible_exposures": eligible,
                "eligible_rate_pct": round(100 * eligible / attempts, 1) if attempts else None,
            })


def main():
    parser = argparse.ArgumentParser(description="Wikipedia hyperlink prior-reversion experiment")
    parser.add_argument("--models", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-min-confidence", type=float, default=0.8)
    parser.add_argument("--cases", default="cases.json")
    parser.add_argument("--case-ids")
    parser.add_argument("--arms", default="conflict,control")
    parser.add_argument("--as-of")
    parser.add_argument(
        "--snapshot-mode", choices=("assigned", "agent_selected"), default="assigned",
        help="assigned binds one controller date; agent_selected lets the model choose first",
    )
    parser.add_argument(
        "--snapshot-dates",
        help="comma-separated allowed dates for --snapshot-mode agent_selected",
    )
    parser.add_argument("--distances", default="1,2,3")
    parser.add_argument("--n-starts-per-distance", type=int, default=5)
    parser.add_argument("--repeats-per-start", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--distractor-every", type=int, default=3)
    parser.add_argument("--pk-repeats", type=int, default=5)
    parser.add_argument("--branch-cap", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--cache-path", default="wikipedia_snapshot.db")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", default="wikipedia_results.jsonl")
    parser.add_argument("--hit-rate-output", default="wikipedia_hit_rates.csv")
    args = parser.parse_args()

    assert_new_output_path(args.output)
    assert_new_output_path(args.hit_rate_output)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if args.judge_model in models:
        parser.error("--judge-model 必須和所有被測模型不同，避免 self-judging")
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[error] OPENROUTER_API_KEY is required", file=sys.stderr)
        return 2
    with open(args.cases, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    cases = data["cases"]
    if args.case_ids:
        wanted = set(args.case_ids.split(","))
        cases = [case for case in cases if case["id"] in wanted]
    args.distractors = data["distractor_questions"]
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    distances = [int(item) for item in args.distances.split(",") if item.strip()]
    snapshot_dates = list(dict.fromkeys(
        item.strip() for item in (args.snapshot_dates or "").split(",") if item.strip()
    ))
    if args.snapshot_mode == "agent_selected":
        if args.as_of:
            parser.error("--as-of cannot be combined with agent_selected; use --snapshot-dates")
        if len(snapshot_dates) < 2:
            parser.error("agent_selected requires at least two --snapshot-dates")
    elif snapshot_dates:
        parser.error("--snapshot-dates is only valid with --snapshot-mode agent_selected")
    args.snapshot_dates = snapshot_dates
    errors = validate_cases(cases, allow_legacy=True)
    if errors:
        print("\n".join(f"[config error] {error}" for error in errors), file=sys.stderr)
        return 2
    if not distances or any(distance < 1 for distance in distances):
        parser.error("--distances must contain positive integers")
    for name in ("n_starts_per_distance", "repeats_per_start", "max_steps", "rounds",
                 "pk_repeats", "branch_cap"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    if not 0 <= args.judge_min_confidence <= 1:
        parser.error("--judge-min-confidence must be between 0 and 1")

    contract_hash = _contract_hash(cases, args)
    backend = WikipediaPageBackend(
        cache_path=args.cache_path, lang=args.lang, offline_only=args.offline
    )
    judge = LLMJudge(args.judge_model, min_confidence=args.judge_min_confidence)
    store = JsonlResultStore(args.output, metadata={
        "contract_hash": contract_hash, "judge_model": args.judge_model,
    })
    completed = store.completed(args.output, contract_hash=contract_hash)
    try:
        for raw_case in cases:
            candidate_dates = (
                list(args.snapshot_dates) if args.snapshot_mode == "agent_selected"
                else [args.as_of or raw_case.get("wikipedia_as_of")]
            )
            contexts_by_date = {}
            # Every selectable date is preflighted before a paid selection call. A date is
            # offered only when every requested arm has a usable target and start frontier.
            for candidate_as_of in candidate_dates:
                arm_contexts = {}
                frontier_cache = {}
                for arm in arms:
                    try:
                        case = case_for_arm(raw_case, arm)
                        target = configured_target(raw_case, arm, backend)
                        if not target:
                            raise ValueError("no explicit Wikipedia target")
                        target = backend.fetch_page(target, as_of=candidate_as_of).title
                        frontier_key = (target.casefold(), candidate_as_of)
                        if frontier_key not in frontier_cache:
                            frontier_cache[frontier_key] = reverse_bfs_frontier(
                                backend, target, max(distances), as_of=candidate_as_of,
                                branch_cap=args.branch_cap,
                            )
                        frontier = frontier_cache[frontier_key]
                        if not any(frontier.get(distance) for distance in distances):
                            raise ValueError("no candidate start pages at requested distances")
                        arm_contexts[arm] = {
                            "case": case, "target": target, "as_of": candidate_as_of,
                            "frontier": frontier,
                        }
                    except (ValueError, WikipediaError) as exc:
                        print(
                            f"[exclude snapshot] {raw_case['id']}/{arm}/"
                            f"{candidate_as_of}: {exc}"
                        )
                if set(arm_contexts) == set(arms):
                    contexts_by_date[candidate_as_of] = arm_contexts
            usable_dates = [date for date in candidate_dates if date in contexts_by_date]
            if not usable_dates:
                print(f"[skip] {raw_case['id']}: no snapshot has all requested arms")
                continue
            if args.snapshot_mode == "agent_selected" and len(usable_dates) < 2:
                print(
                    f"[skip] {raw_case['id']}: agent_selected needs at least two usable snapshots"
                )
                continue
            for model in models:
                snapshot_selection = store.latest_snapshot_selection(
                    args.output, raw_case["id"], model, contract_hash
                )
                if snapshot_selection is None:
                    snapshot_selection = run_snapshot_selection(
                        model, usable_dates, snapshot_mode=args.snapshot_mode,
                        task_prompt=NEUTRAL_TASK,
                    )
                    store.write(
                        slot="temporal_selection", case_id=raw_case["id"], model=model,
                        arm="shared", selection=snapshot_selection,
                        target_snapshot_as_of=raw_case.get("wikipedia_as_of"),
                    )
                else:
                    print(
                        f"[checkpoint] reuse snapshot selection {raw_case['id']}/{model}: "
                        f"{snapshot_selection.get('selection_token')}"
                    )
                if snapshot_selection.get("status") != "selected":
                    print(f"[TIME FAIL] {raw_case['id']}/{model}: no valid snapshot selected")
                    continue
                selected_as_of = snapshot_selection.get("selected_as_of")
                arm_contexts = contexts_by_date.get(selected_as_of, {})
                if not arm_contexts:
                    print(
                        f"[TIME FAIL] {raw_case['id']}/{model}: selected snapshot is not usable"
                    )
                    continue
                if "conflict" in arm_contexts:
                    previous_pk = store.latest_pk_gate(
                        args.output, raw_case["id"], model, contract_hash
                    )
                    if previous_pk is None:
                        conflict_allowed = run_pk_gate(
                            raw_case, model, judge, store, args.pk_repeats
                        )
                    else:
                        conflict_allowed = previous_pk
                        print(f"[checkpoint] reuse PK gate {raw_case['id']}/{model}: {previous_pk}")
                else:
                    conflict_allowed = True
                if "conflict" in arm_contexts and not conflict_allowed:
                    print(f"[PK FAIL] {raw_case['id']}/{model}: skip conflict trajectories")
                for arm, context in arm_contexts.items():
                    if arm == "conflict" and not conflict_allowed:
                        continue
                    case = context["case"]
                    target = context["target"]
                    as_of = context["as_of"]
                    frontier = context["frontier"]
                    rng = random.Random(_seed(args.seed, raw_case["id"], model, arm))
                    for distance in distances:
                        candidates = frontier.get(distance, [])
                        starts = rng.sample(
                            candidates, min(args.n_starts_per_distance, len(candidates))
                        )
                        for start in starts:
                            for repeat in range(args.repeats_per_start):
                                checkpoint = (raw_case["id"], model, arm, distance, start, repeat)
                                if checkpoint in completed:
                                    print(f"[checkpoint] skip {checkpoint}")
                                    continue
                                try:
                                    attempt_id = run_trajectory(
                                        case=case, raw_case=raw_case, arm=arm, model=model,
                                        backend=backend, target=target, as_of=as_of,
                                        start_title=start, distance=distance, repeat=repeat,
                                        args=args, judge=judge, store=store, rng=rng,
                                        snapshot_selection=snapshot_selection,
                                        target_snapshot_as_of=raw_case.get("wikipedia_as_of"),
                                    )
                                except (OpenRouterError, WikipediaError, ValueError) as exc:
                                    print(f"[error] {checkpoint}: {exc}")
                                    attempt_id = None
                                if attempt_id:
                                    trajectory_key = (
                                        f"{raw_case['id']}|{model}|{arm}|d{distance}|{start}|r{repeat}"
                                    )
                                    store.write(
                                        slot="checkpoint", case_id=raw_case["id"], model=model,
                                        arm=arm, start_distance=distance, start_title=start,
                                        repeat=repeat, status="complete",
                                        attempt_id=attempt_id, trajectory_key=trajectory_key,
                                        snapshot_selection_id=snapshot_selection["selection_id"],
                                        snapshot_as_of=as_of,
                                    )
    finally:
        store.close()
        backend.close()
    write_hit_rates(args.output, args.hit_rate_output, contract_hash=contract_hash)
    print(f"Done: {args.output}; hit rates: {args.hit_rate_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
