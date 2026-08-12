"""Run the primary page-by-revision temporal Wikipedia exploration task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

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


CONTRACT_SCHEMA = "temporal-pk-relative-multihop-v7"


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


PK_PROBE_VARIANTS = (
    "Answer with your best direct recall.",
    "Try to retrieve the specific name before deciding that you do not know.",
    "Give one concise best-effort answer; uncertainty is allowed.",
    "Independently reconsider the question and state the most likely answer.",
    "Make a final fresh attempt from memory only, without tools.",
)


def _pk_prompt(case: dict, target_as_of: str | None, variant_index: int = 0) -> str:
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
        f"Probe instruction: {PK_PROBE_VARIANTS[variant_index % len(PK_PROBE_VARIANTS)]}\n"
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
    temperatures: list[float] | None = None,
    probe_call_model_fn=call_model,
) -> dict:
    """Admit a case-model pair only when the target answer is not already known.

    Every probe is a fresh one-turn conversation.  Probe responses are logged but
    never inserted into the later navigation conversation.
    """
    if temperatures is None:
        base = [0.0, 0.2, 0.5, 0.7, 1.0]
        temperatures = [base[index % len(base)] for index in range(repeats)]
    if len(temperatures) != repeats:
        raise ValueError("PK temperature schedule must have one value per repeat")
    if any(not 0 <= value <= 2 for value in temperatures):
        raise ValueError("PK temperatures must be between 0 and 2")
    labels: list[str] = []
    deterministic_judgments = 0
    for probe_repeat in range(repeats):
        prompt = _pk_prompt(case, target_as_of, probe_repeat)
        probe_temperature = temperatures[probe_repeat]
        response = probe_call_model_fn(
            model, [{"role": "user", "content": prompt}],
            temperature=probe_temperature,
        )
        judgment = judge.judge_answer(
            _question(case), response,
            case.get("new_answer_keywords", []),
            case.get("old_answer_keywords", []),
            [],
        )
        labels.append(judgment.decision)
        deterministic = bool(judgment.raw.get("deterministic_gate"))
        deterministic_judgments += int(deterministic)
        store.write(
            slot="pk_probe", case_id=case["id"], model=model, arm="admission",
            probe_repeat=probe_repeat, target_snapshot_as_of=target_as_of,
            prompt=prompt, response=response, judgment=judgment.to_dict(),
            label=judgment.decision, fresh_context=True, tools_available=False,
            probe_variant=probe_repeat % len(PK_PROBE_VARIANTS),
            probe_temperature=probe_temperature,
            judge_mode="deterministic_alias" if deterministic else "llm_fallback",
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
        "temperature_schedule": temperatures,
        "probe_method": "fresh_context_varied_prompt_temperature_ensemble",
        "deterministic_judgments": deterministic_judgments,
        "llm_fallback_judgments": repeats - deterministic_judgments,
    }
    store.write(
        slot="pk_gate", case_id=case["id"], model=model, arm="admission", **gate,
    )
    return gate


def _snapshot_values(raw: str | None, case: dict) -> list[str | None]:
    """Return the oracle dates, or an explicit legacy fixed-date allowlist."""
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


def _snapshot_range(case: dict) -> tuple[str, str]:
    """Return the public cutoff-to-target range without intermediate oracle dates."""
    before, after = _case_endpoints(case)
    if before is None or after is None:
        raise ValueError(
            f"{case.get('id', '<missing-id>')}: agent-selected time requires dated "
            "before and target snapshots"
        )
    try:
        start = date.fromisoformat(before)
        end = date.fromisoformat(after)
    except ValueError as exc:
        raise ValueError(
            f"{case.get('id', '<missing-id>')}: before and target must use YYYY-MM-DD"
        ) from exc
    if start.isoformat() != before or end.isoformat() != after:
        raise ValueError(
            f"{case.get('id', '<missing-id>')}: before and target must use YYYY-MM-DD"
        )
    if start >= end:
        raise ValueError(
            f"{case.get('id', '<missing-id>')}: target must be after the cutoff snapshot"
        )
    return before, after


def _contract_hash(cases: list[dict], args) -> str:
    payload = {
        "schema": CONTRACT_SCHEMA,
        "cases": cases,
        "models": args.models,
        "judge_model": args.judge_model,
        "judge_min_confidence": args.judge_min_confidence,
        "snapshot_dates": args.snapshot_dates,
        "snapshot_policy": (
            "fixed_allowlist" if args.snapshot_dates else "agent_selected_range"
        ),
        "repeats": args.repeats,
        "max_steps": args.max_steps,
        "start_distance": args.start_distance,
        "backlink_branch_cap": args.backlink_branch_cap,
        "arena_node_cap": args.arena_node_cap,
        "arena_discovery_mode": "offline_cache" if args.offline else "live_api",
        "pk_repeats": args.pk_repeats,
        "pk_temperature_schedule": args.pk_temperature_schedule,
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
    result: dict, navigation: dict, start_title: str, *, strict_arena: bool = True
) -> dict:
    distances = navigation["distances"]
    target_key = navigation["target_key"]
    initial_state = result.get("initial_state") or {}
    initial_state_key = (
        page_version_key(str(initial_state.get("title", "")), initial_state["revision_id"])
        if initial_state.get("revision_id") is not None else None
    )
    initial_distance = distances.get(initial_state_key) if initial_state_key else None
    if initial_distance is None:
        start_state_distances = [
            state["distance_to_pivot"] for state in navigation["states"].values()
            if state["title"].casefold() == start_title.casefold()
        ]
    else:
        start_state_distances = [initial_distance]
    if not start_state_distances:
        raise ValueError(f"start title {start_title!r} is absent from navigation graph")
    # Initial state is now a real cutoff revision and costs no tool action.  If
    # that exact revision is outside the bounded reference arena, one temporal
    # move is needed to enter the nearest represented state for the same title.
    shortest_navigation_steps = min(start_state_distances) + int(initial_distance is None)
    navigation_steps = 0
    seen_versions: set[str] = set()
    revisit_count = 0
    initial_is_target = initial_state_key == target_key
    first_target_navigation_steps: int | None = 0 if initial_is_target else None
    first_target_tool_step: int | None = None
    distance_trace: list[dict] = []
    unknown_distance_count = 0
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
            unknown_distance_count += 1
            if strict_arena:
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
    if detour_steps is not None and detour_steps < 0 and strict_arena:
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
        "reference_distance_coverage_rate": (
            (navigation_steps - unknown_distance_count) / navigation_steps
            if navigation_steps else 1.0
        ),
        "outside_reference_arena_count": unknown_distance_count,
        "reference_shortcut_detected": (
            detour_steps is not None and detour_steps < 0
        ),
        "initial_state_in_reference_arena": initial_distance is not None,
    }


def _semantic_waypoint_metrics(result: dict, case: dict) -> dict:
    """Compare against one generated proof route without making it mandatory.

    Hyperlink steps must match the declared entity relation and must really
    have succeeded in the rendered revision.  Intermediate temporal steps may
    use any strictly later model-selected date; only the first cutoff and final
    target dates are fixed by the question contract.  Exact generated
    revisions remain useful reference evidence, not a secret answer the model
    has to guess.
    """
    waypoints = case.get("temporal_waypoints") or []
    if not waypoints:
        return {
            "semantic_route_complete": None,
            "semantic_waypoints_completed": None,
            "semantic_waypoint_count": 0,
            "semantic_completion_rate": None,
            "semantic_shortest_navigation_steps": None,
            "semantic_actual_steps_to_complete": None,
            "semantic_detour_steps": None,
            "semantic_shortest_arrival": None,
            "required_temporal_switches": 0,
            "actual_required_temporal_switches": 0,
            "actual_required_snapshot_visits": 0,
            "reference_route_match": None,
            "reference_waypoints_completed": None,
            "reference_waypoint_count": 0,
            "reference_route_coverage": None,
            "reference_route_steps": None,
        }

    required_temporal = sum(
        item.get("incoming_edge") == "temporal" for item in waypoints[1:]
    )
    progress = 0
    temporal_completed = 0
    reference_revision_visits: set[int] = set()
    completion_navigation_step: int | None = None
    route_date: date | None = None
    _, target_as_of = _case_endpoints(case)
    target_date = date.fromisoformat(target_as_of) if target_as_of else None

    initial = result.get("initial_state") or {}
    if waypoints:
        first = waypoints[0]
        if (
            str(initial.get("title", "")).casefold()
            == str(first.get("title", "")).casefold()
            and initial.get("snapshot_token") == first.get("as_of")
        ):
            progress = 1
            route_date = date.fromisoformat(str(first["as_of"]))
    initial_waypoint_loaded = progress == 1

    def successful_navigation(record: dict) -> bool:
        return (
            record.get("action") in {"switch_snapshot", "follow_link"}
            and not str(record.get("result", "")).startswith("Error:")
            and record.get("revision_id") is not None
        )

    def record_date(record: dict, expected: dict) -> date | None:
        token = record.get("snapshot_token")
        if isinstance(token, str):
            try:
                parsed = date.fromisoformat(token)
            except ValueError:
                parsed = None
            if parsed is not None and parsed.isoformat() == token:
                return parsed
        # Compatibility for old exact-waypoint artifacts that predate explicit
        # snapshot tokens in test fixtures.
        if record.get("revision_id") == expected.get("revision_id"):
            as_of = expected.get("as_of")
            if isinstance(as_of, str):
                try:
                    return date.fromisoformat(as_of)
                except ValueError:
                    return None
        return None

    for record in result.get("trajectory", []):
        if not successful_navigation(record):
            continue
        action = record.get("action")
        destination_title = str(record.get("to_title", ""))
        destination_revision = int(record["revision_id"])
        if any(
            destination_title.casefold() == str(item.get("title", "")).casefold()
            and destination_revision == item.get("revision_id")
            for item in waypoints
        ):
            reference_revision_visits.add(destination_revision)
        if progress == 0:
            expected = waypoints[0]
            selected_date = record_date(record, expected)
            expected_cutoff = date.fromisoformat(str(expected["as_of"]))
            if (
                action == "switch_snapshot"
                and destination_title.casefold()
                == str(expected["title"]).casefold()
                and selected_date == expected_cutoff
            ):
                progress = 1
                route_date = selected_date
                record["semantic_waypoint_index"] = 0
                record["semantic_progress"] = progress
            continue
        if progress >= len(waypoints):
            continue
        expected = waypoints[progress]
        expected_action = (
            "switch_snapshot"
            if expected.get("incoming_edge") == "temporal" else "follow_link"
        )
        previous = waypoints[progress - 1]
        source_matches = (
            str(record.get("from_title", "")).casefold()
            == str(previous["title"]).casefold()
        )
        destination_matches = (
            destination_title.casefold() == str(expected["title"]).casefold()
        )
        advances = False
        if action == expected_action and source_matches and destination_matches:
            if expected_action == "follow_link":
                from_token = record.get("from_snapshot_token")
                to_token = record.get("snapshot_token")
                advances = (
                    from_token is None or to_token is None or from_token == to_token
                )
            else:
                selected_date = record_date(record, expected)
                is_final_temporal_switch = not any(
                    item.get("incoming_edge") == "temporal"
                    for item in waypoints[progress + 1:]
                )
                advances = (
                    selected_date is not None
                    and route_date is not None
                    and selected_date > route_date
                    and (
                        selected_date == target_date
                        if is_final_temporal_switch else
                        target_date is None or selected_date < target_date
                    )
                )
                if advances:
                    route_date = selected_date
                    temporal_completed += 1
        if advances:
            record["semantic_waypoint_index"] = progress
            progress += 1
            record["semantic_progress"] = progress
            if progress == len(waypoints):
                completion_navigation_step = record.get("navigation_step")

    count = len(waypoints)
    complete = progress == count
    semantic_shortest_steps = max(0, count - int(initial_waypoint_loaded))
    semantic_detour = (
        int(completion_navigation_step) - semantic_shortest_steps
        if completion_navigation_step is not None else None
    )
    metrics = {
        "semantic_route_complete": complete,
        "semantic_waypoints_completed": progress,
        "semantic_waypoint_count": count,
        "semantic_completion_rate": progress / count,
        "semantic_shortest_navigation_steps": semantic_shortest_steps,
        "semantic_actual_steps_to_complete": completion_navigation_step,
        "semantic_detour_steps": semantic_detour,
        "semantic_shortest_arrival": semantic_detour == 0,
        "required_temporal_switches": required_temporal,
        "actual_required_temporal_switches": temporal_completed,
        "actual_required_snapshot_visits": progress,
        "reference_waypoint_revision_visits": len(reference_revision_visits),
    }
    metrics.update({
        "reference_route_match": complete,
        "reference_waypoints_completed": progress,
        "reference_waypoint_count": count,
        "reference_route_coverage": progress / count,
        "reference_route_steps": completion_navigation_step,
    })
    return metrics


def _capability_metrics(
    result: dict, *, target_title: str, target_as_of: str | None,
    cutoff_reference: str | None, accepted_answers: list[str],
) -> dict:
    """Decompose browsing into observable abilities without using an oracle path."""
    trajectory = result.get("trajectory", [])

    def rows(action: str) -> list[dict]:
        return [row for row in trajectory if row.get("action") == action]

    def successful(row: dict) -> bool:
        return not str(row.get("result", "")).startswith("Error:")

    revisions = rows("list_revisions")
    switches = rows("switch_snapshot")
    follows = rows("follow_link")
    errors = [row for row in trajectory if not successful(row)]
    error_counts: dict[str, int] = {}
    for row in errors:
        action = str(row.get("action") or "unknown")
        error_counts[action] = error_counts.get(action, 0) + 1
    visited = result.get("visited_versions", [])
    evidence = result.get("evidence_pages", [])
    initial = result.get("initial_state") or {}
    target_page_seen = any(
        str(page.get("title", "")).casefold() == target_title.casefold()
        for page in evidence
    )
    target_snapshot_page_seen = any(
        str(page.get("title", "")).casefold() == target_title.casefold()
        and page.get("as_of") == target_as_of
        for page in evidence
    )
    normalized_answers = [
        " ".join(value.casefold().split()) for value in accepted_answers if value.strip()
    ]
    target_snapshot_evidence_seen = any(
        page.get("as_of") == target_as_of
        and any(
            alias in " ".join(str(page.get("content", "")).casefold().split())
            for alias in normalized_answers
        )
        for page in evidence
    )
    return {
        "cutoff_state_loaded": (
            cutoff_reference is None
            or initial.get("snapshot_token") == cutoff_reference
        ),
        "revision_discovery_attempts": len(revisions),
        "revision_discovery_successes": sum(successful(row) for row in revisions),
        "temporal_switch_attempts": len(switches),
        "temporal_switch_successes": sum(successful(row) for row in switches),
        "hyperlink_follow_attempts": len(follows),
        "hyperlink_follow_successes": sum(successful(row) for row in follows),
        "unique_pages_visited": len({
            str(page.get("title", "")).casefold() for page in visited
        }),
        "unique_revisions_visited": len({
            (str(page.get("title", "")).casefold(), page.get("revision_id"))
            for page in visited
        }),
        "target_page_seen": target_page_seen,
        "target_snapshot_page_seen": target_snapshot_page_seen,
        "target_snapshot_evidence_seen": target_snapshot_evidence_seen,
        "answer_submitted": bool(str(result.get("final_answer") or "").strip()),
        "tool_error_count": len(errors),
        "tool_errors_by_action": error_counts,
    }


def _failure_mode(capabilities: dict, label: str, stop_reason: str) -> str:
    if label == "correct_after":
        return (
            "success_with_target_evidence"
            if capabilities["target_snapshot_evidence_seen"]
            else "correct_without_visible_target_evidence"
        )
    if not capabilities["answer_submitted"]:
        return "no_answer_before_step_budget" if stop_reason == "max_steps" else "no_answer"
    if not capabilities["temporal_switch_successes"]:
        return "temporal_exploration_failure"
    if capabilities["revision_discovery_attempts"] and not capabilities[
        "revision_discovery_successes"
    ]:
        return "revision_discovery_failure"
    if capabilities["hyperlink_follow_attempts"] and not capabilities[
        "hyperlink_follow_successes"
    ]:
        return "hyperlink_navigation_failure"
    if not capabilities["target_page_seen"]:
        return "target_page_not_found"
    if not capabilities["target_snapshot_evidence_seen"]:
        return "target_time_not_grounded"
    if label == "old_snapshot_answer":
        return "stale_answer_after_target_evidence"
    return "wrong_answer_after_target_evidence"


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
    snapshot_date_range: tuple[str, str] | None = None,
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
        snapshot_date_range=snapshot_date_range,
        allowed_version_keys=(
            set(navigation["allowed_version_keys"])
            if snapshot_date_range is None else None
        ),
        allowed_titles_by_snapshot=(
            navigation["allowed_titles_by_snapshot"]
            if snapshot_date_range is None else None
        ),
        reveal_target_title=not bool(case.get("hide_pivot_title")),
        cutoff_reference=case.get("knowledge_cutoff", {}).get("cutoff_date"),
        semantic_route_contract=(
            {
                "relation_hops": case.get("reasoning_hop_count", 0),
                "temporal_switches": case.get("required_temporal_switches", 0),
            }
            if case.get("temporal_waypoints") else None
        ),
        temperature=temperature,
        **kwargs,
    )
    metrics = _navigation_metrics(
        result, navigation, start_title,
        strict_arena=snapshot_date_range is None,
    )
    semantic_metrics = _semantic_waypoint_metrics(result, case)
    capabilities = _capability_metrics(
        result, target_title=target_page.title, target_as_of=target_as_of,
        cutoff_reference=case.get("knowledge_cutoff", {}).get("cutoff_date"),
        accepted_answers=case.get("new_answer_keywords", []),
    )
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
        relation_families=case.get("relation_families", []),
        selection_metadata=case.get("selection_metadata"),
        temporal_waypoints=case.get("temporal_waypoints", []),
        knowledge_cutoff=case.get("knowledge_cutoff"),
        navigation_id=_navigation_id(navigation),
        snapshot_mode=(
            "agent_selected_range" if snapshot_date_range is not None
            else "fixed_allowlist"
        ),
        snapshot_range=(
            {"start": snapshot_date_range[0], "end": snapshot_date_range[1]}
            if snapshot_date_range is not None else None
        ),
        reference_snapshot_dates=snapshot_dates,
        allowed_as_of=(snapshot_dates if snapshot_date_range is None else []),
        target_snapshot_as_of=target_as_of,
        final_title=result["final_title"],
        final_snapshot_token=result["final_snapshot_token"],
        final_answer=result["final_answer"], stop_reason=result["stop_reason"],
        agent_initial_messages=result["initial_messages"],
        agent_tool_contract=result["tool_contract"],
        target_title_revealed=result["target_title_revealed"],
        initial_state=result["initial_state"],
        visited_versions=result["visited_versions"],
        **metrics,
        evidence_revisions=[
            {
                "title": page["title"], "revision_id": page["revision_id"],
                "timestamp": page["timestamp"], "as_of": page.get("as_of"),
            }
            for page in result["evidence_pages"]
        ],
        raw_shortest_navigation_steps=metrics["shortest_navigation_steps"],
        **capabilities,
        **semantic_metrics,
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
        judge_mode=(
            "deterministic_alias" if judgment.raw.get("deterministic_gate")
            else "llm_fallback"
        ),
        failure_mode=_failure_mode(
            capabilities, judgment.decision, result["stop_reason"]
        ),
        capabilities=capabilities,
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
    counts: dict[tuple[str, str], dict[str, Any]] = {
        key: {
            "n": 0, "correct_after": 0, "old_snapshot_answer": 0,
            "correct_without_visible_support": 0,
            "no_answer": 0, "blank_answer_judge_overrides": 0,
            "pivot_hit": 0, "shortest_arrival": 0, "cycle_detected": 0,
            "found_but_wrong": 0, "detour_sum": 0, "detour_n": 0,
            "semantic_route_complete": 0, "semantic_progress_sum": 0,
            "revision_discovery_used": 0, "temporal_explored": 0,
            "hyperlink_navigation_succeeded": 0,
            "target_snapshot_evidence_seen": 0, "answer_submitted": 0,
            "failure_modes": {},
        }
        for key in pk_gates
    }
    for row in judgments:
        key = (str(row.get("model")), str(row.get("case_id")))
        bucket = counts.setdefault(key, {
            "n": 0, "correct_after": 0, "old_snapshot_answer": 0,
            "correct_without_visible_support": 0,
            "no_answer": 0, "blank_answer_judge_overrides": 0,
            "pivot_hit": 0, "shortest_arrival": 0, "cycle_detected": 0,
            "found_but_wrong": 0, "detour_sum": 0, "detour_n": 0,
            "semantic_route_complete": 0, "semantic_progress_sum": 0,
            "revision_discovery_used": 0, "temporal_explored": 0,
            "hyperlink_navigation_succeeded": 0,
            "target_snapshot_evidence_seen": 0, "answer_submitted": 0,
            "failure_modes": {},
        })
        bucket["n"] += 1
        summary = summaries.get(str(row.get("attempt_id")), {})
        blank_answer = (
            "final_answer" in summary
            and not str(summary.get("final_answer") or "").strip()
        )
        label = "no_answer" if blank_answer else str(row.get("label"))
        if blank_answer and row.get("label") != "no_answer":
            bucket["blank_answer_judge_overrides"] += 1
        if label in bucket:
            bucket[label] += 1
        pivot_hit = bool(summary.get("pivot_hit"))
        bucket["pivot_hit"] += int(pivot_hit)
        bucket["shortest_arrival"] += int(bool(summary.get("shortest_arrival")))
        bucket["cycle_detected"] += int(bool(summary.get("cycle_detected")))
        bucket["found_but_wrong"] += int(
            pivot_hit and label not in {
                "correct_after", "correct_without_visible_support",
            }
        )
        bucket["revision_discovery_used"] += int(
            int(summary.get("revision_discovery_attempts", 0) or 0) > 0
        )
        bucket["temporal_explored"] += int(
            int(summary.get("temporal_switch_successes", 0) or 0) > 0
        )
        bucket["hyperlink_navigation_succeeded"] += int(
            int(summary.get("hyperlink_follow_successes", 0) or 0) > 0
        )
        bucket["target_snapshot_evidence_seen"] += int(
            bool(summary.get("target_snapshot_evidence_seen"))
        )
        bucket["answer_submitted"] += int(bool(summary.get("answer_submitted")))
        failure_mode = str(row.get("failure_mode") or "unclassified")
        failure_modes = bucket["failure_modes"]
        failure_modes[failure_mode] = failure_modes.get(failure_mode, 0) + 1
        bucket["semantic_route_complete"] += int(
            bool(summary.get("semantic_route_complete"))
        )
        if summary.get("semantic_completion_rate") is not None:
            bucket["semantic_progress_sum"] += float(
                summary["semantic_completion_rate"]
            )
        if summary.get("detour_steps") is not None:
            bucket["detour_sum"] += int(summary["detour_steps"])
            bucket["detour_n"] += 1
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        fields = [
            "model", "case_id", "pk_admitted", "pk_gate_reason", "pk_probe_n",
            "pk_stick_new", "pk_stick_new_rate_pct", "pk_stick_old",
            "pk_stick_old_rate_pct", "pk_other", "n", "correct_after",
            "pk_deterministic_judgments", "pk_llm_fallback_judgments",
            "correct_after_rate_pct",
            "correct_without_visible_support",
            "old_snapshot_answer", "old_snapshot_rate_pct",
            "no_answer", "no_answer_rate_pct", "blank_answer_judge_overrides",
            "pivot_hit", "pivot_hit_rate_pct", "shortest_arrival",
            "shortest_arrival_rate_pct", "mean_detour_steps_on_hit",
            "semantic_route_complete", "semantic_route_complete_rate_pct",
            "mean_semantic_completion_pct",
            "revision_discovery_used", "temporal_explored",
            "hyperlink_navigation_succeeded", "target_snapshot_evidence_seen",
            "answer_submitted", "failure_modes_json",
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
                "pk_deterministic_judgments": pk_gate.get(
                    "deterministic_judgments", ""
                ),
                "pk_llm_fallback_judgments": pk_gate.get(
                    "llm_fallback_judgments", ""
                ),
                "correct_after": bucket["correct_after"],
                "correct_after_rate_pct": (
                    round(100 * bucket["correct_after"] / n, 1) if n else ""
                ),
                "correct_without_visible_support": bucket[
                    "correct_without_visible_support"
                ],
                "old_snapshot_answer": bucket["old_snapshot_answer"],
                "old_snapshot_rate_pct": (
                    round(100 * bucket["old_snapshot_answer"] / n, 1) if n else ""
                ),
                "no_answer": bucket["no_answer"],
                "no_answer_rate_pct": (
                    round(100 * bucket["no_answer"] / n, 1) if n else ""
                ),
                "blank_answer_judge_overrides": bucket[
                    "blank_answer_judge_overrides"
                ],
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
                "semantic_route_complete": bucket["semantic_route_complete"],
                "semantic_route_complete_rate_pct": (
                    round(100 * bucket["semantic_route_complete"] / n, 1)
                    if n else ""
                ),
                "mean_semantic_completion_pct": (
                    round(100 * bucket["semantic_progress_sum"] / n, 1)
                    if n else ""
                ),
                "revision_discovery_used": bucket["revision_discovery_used"],
                "temporal_explored": bucket["temporal_explored"],
                "hyperlink_navigation_succeeded": bucket[
                    "hyperlink_navigation_succeeded"
                ],
                "target_snapshot_evidence_seen": bucket[
                    "target_snapshot_evidence_seen"
                ],
                "answer_submitted": bucket["answer_submitted"],
                "failure_modes_json": json.dumps(
                    bucket["failure_modes"], sort_keys=True
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
        help=(
            "legacy fixed allowlist for switch_snapshot; omit it to let the model choose "
            "any date from the case cutoff through its target"
        ),
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
        "--arena-node-cap", type=int, default=500,
        help="maximum page-version nodes in the bounded reference arena",
    )
    parser.add_argument(
        "--pk-repeats", type=int, default=5,
        help="fresh-context varied-prompt prior-answer probes per case-model pair",
    )
    parser.add_argument(
        "--pk-temperatures",
        help="comma-separated PK sampling temperatures; count must equal --pk-repeats",
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
    if min(args.start_distance, args.backlink_branch_cap, args.arena_node_cap) <= 0:
        parser.error(
            "--start-distance, --backlink-branch-cap, and --arena-node-cap must be > 0"
        )
    if args.max_steps < args.start_distance + 1:
        parser.error(
            "--max-steps must allow the shortest navigation path and answer submission"
        )
    if not 0 <= args.judge_min_confidence <= 1:
        parser.error("--judge-min-confidence must be between 0 and 1")
    if not 0 <= args.pk_max_known_rate <= 1:
        parser.error("--pk-max-known-rate must be between 0 and 1")
    if args.pk_temperatures:
        try:
            args.pk_temperature_schedule = [
                float(value.strip()) for value in args.pk_temperatures.split(",")
                if value.strip()
            ]
        except ValueError:
            parser.error("--pk-temperatures must be comma-separated numbers")
        if len(args.pk_temperature_schedule) != args.pk_repeats:
            parser.error("--pk-temperatures count must equal --pk-repeats")
    else:
        base_temperatures = [0.0, 0.2, 0.5, 0.7, 1.0]
        args.pk_temperature_schedule = [
            base_temperatures[index % len(base_temperatures)]
            for index in range(args.pk_repeats)
        ]
    if any(not 0 <= value <= 2 for value in args.pk_temperature_schedule):
        parser.error("--pk-temperatures values must be between 0 and 2")
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
        (int(case.get("expected_navigation_distance", args.start_distance)) + 1
         for case in cases),
        default=args.start_distance + 1,
    )
    if args.max_steps < required_max_steps:
        parser.error(
            f"--max-steps must be >= {required_max_steps} for the deepest declared "
            "reasoning chain and answer submission"
        )
    # Validate all per-case date sets before opening the append-only result file.
    dates_by_case = {
        case["id"]: _snapshot_values(args.snapshot_dates, case) for case in cases
    }
    ranges_by_case = {
        case["id"]: (None if args.snapshot_dates else _snapshot_range(case))
        for case in cases
    }
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
                    max_nodes=args.arena_node_cap,
                    required_waypoints=case.get("temporal_waypoints") or None,
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
                            temperatures=args.pk_temperature_schedule,
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
            snapshot_range = ranges_by_case[case["id"]]
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
                    discovery_mode=navigation["discovery_mode"],
                    branch_cap=navigation["branch_cap"],
                    max_nodes=navigation["max_nodes"],
                    arena_truncated=navigation["arena_truncated"],
                    snapshot_mode=(
                        "agent_selected_range"
                        if snapshot_range is not None
                        else "fixed_allowlist"
                    ),
                    snapshot_range=(
                        {
                            "start": snapshot_range[0],
                            "end": snapshot_range[1],
                        }
                        if snapshot_range is not None else None
                    ),
                    reference_only=snapshot_range is not None,
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
                            snapshot_date_range=snapshot_range,
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
