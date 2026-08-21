"""Run external-agent and graph-constrained search arms on public case views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tkg.experiment.case_validation import validate_cases
from tkg.experiment.human_review import resolve_human_reviews
from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_beam import (
    BEAM_TRAJECTORY_SCHEMA, BeamSearchConfig, BeamSearchResult,
    TemporalSearchRequest, run_temporal_beam_search,
)
from tkg.experiment.temporal_beam_ranker import ApiUtilityRanker
from tkg.wikipedia.backend import WikipediaPageBackend
from tkg.wikipedia.browser import run_temporal_browsing


PUBLIC_CASE_SCHEMA = "temporal-beam-public-cases-v1"
RUN_SCHEMA = "temporal-graph-constrained-search-run-v1"
ARM_WIDTHS = {"A": None, "B": 1, "C": 3, "D": 5}
PUBLIC_CASE_FIELDS = {
    "id", "model_id", "question", "start_page", "cutoff_date", "target_date",
}
FORBIDDEN_FIELD_PATTERN = re.compile(
    r"(?:qid|private|gold|reference|route|waypoint|answer|pivot|intermediate)",
    re.IGNORECASE,
)


def public_request_from_record(record: dict[str, Any]) -> tuple[TemporalSearchRequest, str]:
    if not isinstance(record, dict):
        raise ValueError("public case must be an object")
    unknown = set(record) - PUBLIC_CASE_FIELDS
    dangerous = sorted(field for field in unknown if FORBIDDEN_FIELD_PATTERN.search(field))
    if dangerous:
        raise ValueError(f"public case contains forbidden private fields: {dangerous!r}")
    if unknown:
        raise ValueError(f"public case contains unknown fields: {sorted(unknown)!r}")
    required = [field for field in PUBLIC_CASE_FIELDS if not str(record.get(field, "")).strip()]
    if required:
        raise ValueError(f"public case missing fields: {sorted(required)!r}")
    request = TemporalSearchRequest(
        case_id=str(record["id"]),
        question=str(record["question"]),
        start_page=str(record["start_page"]),
        cutoff_date=str(record["cutoff_date"]),
        target_date=str(record["target_date"]),
    )
    if request.cutoff_date >= request.target_date:
        raise ValueError(f"{request.case_id}: cutoff_date must precede target_date")
    if re.search(r"\b[QP][1-9]\d*\b", request.question):
        raise ValueError(f"{request.case_id}: public question exposes a Wikidata identifier")
    return request, str(record["model_id"])


def load_public_cases(path: str | Path) -> list[tuple[TemporalSearchRequest, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != PUBLIC_CASE_SCHEMA:
        raise ValueError(f"public manifest must use {PUBLIC_CASE_SCHEMA}")
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("public manifest needs non-empty cases")
    projected = [public_request_from_record(row) for row in rows]
    ids = [request.case_id for request, _ in projected]
    if len(ids) != len(set(ids)):
        raise ValueError("public manifest contains duplicate case IDs")
    return projected


def _latest_pk_gate(
    path: str | Path, case_id: str, model: str, contract_hash: str,
) -> dict[str, Any] | None:
    result = None
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("slot") == "pk_gate"
                and row.get("case_id") == case_id
                and row.get("model") == model
                and row.get("contract_hash") == contract_hash
            ):
                result = row
    return result


def validate_formal_admission(
    *, case: dict[str, Any], model: str, pk_results: str | Path,
    contract_hash: str,
) -> dict[str, Any]:
    errors = validate_cases([case])
    if errors:
        raise ValueError("formal case validation failed: " + "; ".join(errors))
    generation = case.get("_generation", {})
    if generation.get("schema_version") != "wikipedia-cutoff-relative-multihop-v6":
        raise ValueError("formal beam runs require a v6 generated case")
    gate = _latest_pk_gate(pk_results, str(case["id"]), model, contract_hash)
    if gate is None:
        raise ValueError("no matching frozen PK gate")
    labels = gate.get("label_counts")
    if (
        not gate.get("passed")
        or not isinstance(labels, dict)
        or int(labels.get("unjudgeable", 0)) != 0
        or float(gate.get("critical_bridge_known_rate", 1.0))
        > float(gate.get("max_known_rate", 0.0))
    ):
        raise ValueError("critical-bridge PK admission did not pass")
    return {
        "status": "admitted",
        "contract_hash": contract_hash,
        "known_rate": gate.get("critical_bridge_known_rate"),
        "unjudgeable_count": labels.get("unjudgeable"),
        "gate_reason": gate.get("reason"),
    }


class _JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = self.path.open("x", encoding="utf-8")

    def write(self, **fields: Any) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": RUN_SCHEMA,
            **fields,
        }
        self._fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _evidence_id(page: dict[str, Any]) -> str:
    payload = {
        "title": page.get("title"),
        "revision_id": page.get("revision_id"),
        "timestamp": page.get("timestamp"),
        "as_of": page.get("as_of"),
        "content": page.get("content"),
        "links": page.get("links"),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return "evidence_" + hashlib.sha256(encoded).hexdigest()[:24]


def _is_evidence_page(value: dict[str, Any]) -> bool:
    return (
        "title" in value and "revision_id" in value
        and "content" in value and "links" in value
    )


def _collect_evidence(value: Any, found: dict[str, dict[str, Any]]) -> None:
    if isinstance(value, dict):
        if _is_evidence_page(value):
            found.setdefault(_evidence_id(value), value)
            return
        for child in value.values():
            _collect_evidence(child, found)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _collect_evidence(child, found)


def _compact_audit_value(value: Any) -> Any:
    if isinstance(value, dict):
        if _is_evidence_page(value):
            content = str(value.get("content", ""))
            raw_links = value.get("links")
            links: list[Any] = raw_links if isinstance(raw_links, list) else []
            return {
                "evidence_id": _evidence_id(value),
                "title": value.get("title"),
                "revision_id": value.get("revision_id"),
                "timestamp": value.get("timestamp"),
                "as_of": value.get("as_of"),
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "link_count": len(links),
            }
        return {key: _compact_audit_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact_audit_value(child) for child in value]
    return value


def _beam_browser_view(result: BeamSearchResult) -> dict[str, Any]:
    state = result.final_state
    trajectory = []
    action_names = {
        "FOLLOW_LINK": "follow_link",
        "LIST_REVISIONS": "list_revisions",
        "SWITCH_SNAPSHOT": "switch_snapshot",
        "SUBMIT_ANSWER": "submit_answer",
    }
    for row in state.action_trace:
        action = row["action"]
        trajectory.append({
            "step": row["index"],
            "action": action_names[action["kind"]],
            "args": action["params"],
            "from_title": row["from_node"][0],
            "to_title": row["to_node"][0],
            "from_revision_id": row["from_node"][1],
            "revision_id": row["to_node"][1],
            "result": "Error: " + row["error"] if row["error"] else "ok",
        })
    return {
        "trajectory": trajectory,
        "evidence_pages": list(state.collected_evidence),
        "visited_versions": [
            {"title": title, "revision_id": revision}
            for title, revision in state.visited_nodes
        ],
        "final_answer": state.submitted_answer,
        "stop_reason": state.stop_reason or result.stop_reason,
    }


def engineering_metrics(result: BeamSearchResult) -> dict[str, Any]:
    state = result.final_state
    trace = list(state.action_trace)
    follows = [row for row in trace if row["action"]["kind"] == "FOLLOW_LINK"]
    switches = [row for row in trace if row["action"]["kind"] == "SWITCH_SNAPSHOT"]
    errors = [row for row in trace if row.get("error")]
    target_revision_seen = any(
        page.get("as_of") == result.request.target_date
        for page in state.collected_evidence
    )
    return {
        "critical_post_cutoff_bridge_acquired": "not_evaluated_without_formal_case",
        "target_date_wikipedia_revision_seen": target_revision_seen,
        "target_date_wikipedia_evidence_seen": "not_evaluated_without_answer_contract",
        "tail_composition_complete": "not_evaluated_without_answer_contract",
        "hyperlink_validity": (
            all(row.get("hyperlink_valid") is True for row in follows)
            if follows else "not_attempted"
        ),
        "revision_validity": (
            all(row.get("revision_valid") is True for row in switches)
            if switches else "not_attempted"
        ),
        "temporal_consistency": all(
            row["from_node"][0] == row["to_node"][0] for row in switches
        ),
        "visited_unique_nodes": len(set(state.visited_nodes)),
        "expansions": result.expansions,
        "action_count": len(trace),
        "repeated_state_count": result.repeated_state_count,
        "wrong_page_failure": "not_evaluated_without_formal_case",
        "wrong_revision_failure": "not_evaluated_without_formal_case",
        "entity_state_tracking_failure": False,
        "entity_state_tracking_policy": "monotonic_union_of_visible_extractions",
        "tool_api_cache_error": bool(errors or state.error),
        "submitted_answer": state.submitted_answer,
        "answer_status": "submitted" if state.submitted_answer else "no_answer",
        "search_stop_reason": result.stop_reason,
        "state_stop_reason": state.stop_reason,
        "formal_success": "not_evaluated_engineering_smoke",
        "score_recomputable": result.to_dict()["score_recomputable"],
    }


def external_engineering_metrics(result: dict[str, Any], target_date: str) -> dict[str, Any]:
    trajectory = list(result.get("trajectory") or [])
    follows = [row for row in trajectory if row.get("action") == "follow_link"]
    switches = [row for row in trajectory if row.get("action") == "switch_snapshot"]
    errors = [
        row for row in trajectory
        if str(row.get("result", "")).startswith("Error:")
    ]
    nodes = [
        (str(row.get("to_title", "")).casefold(), row.get("revision_id"))
        for row in trajectory if row.get("revision_id") is not None
    ]
    return {
        "critical_post_cutoff_bridge_acquired": "not_evaluated_without_formal_case",
        "target_date_wikipedia_revision_seen": any(
            page.get("as_of") == target_date
            for page in result.get("evidence_pages", [])
        ),
        "target_date_wikipedia_evidence_seen": "not_evaluated_without_answer_contract",
        "tail_composition_complete": "not_evaluated_without_answer_contract",
        "hyperlink_validity": (
            all(not str(row.get("result", "")).startswith("Error:") for row in follows)
            if follows else "not_attempted"
        ),
        "revision_validity": (
            all(not str(row.get("result", "")).startswith("Error:") for row in switches)
            if switches else "not_attempted"
        ),
        "temporal_consistency": all(
            str(row.get("from_title", "")).casefold()
            == str(row.get("to_title", "")).casefold()
            for row in switches
        ),
        "visited_unique_nodes": len(set(nodes)),
        "expansions": len(trajectory),
        "action_count": len(trajectory),
        "repeated_state_count": len(nodes) - len(set(nodes)),
        "wrong_page_failure": "not_evaluated_without_formal_case",
        "wrong_revision_failure": "not_evaluated_without_formal_case",
        "entity_state_tracking_failure": "not_evaluated_for_external_agent",
        "tool_api_cache_error": bool(errors),
        "submitted_answer": str(result.get("final_answer") or ""),
        "answer_status": (
            "submitted" if str(result.get("final_answer") or "").strip()
            else "no_answer"
        ),
        "search_stop_reason": result.get("stop_reason"),
        "formal_success": "not_evaluated_engineering_smoke",
    }


def posthoc_private_metrics(
    result: BeamSearchResult, case: dict[str, Any],
) -> dict[str, Any]:
    # Imported only inside the offline scorer boundary. The search API above has
    # no parameter through which this case can reach the ranker.
    from tkg.experiment.temporal_runner import (  # noqa: PLC0415
        _capability_metrics, _critical_bridge_evidence_metrics,
    )

    browser_view = _beam_browser_view(result)
    target_title = str(case.get("wikipedia_title") or "")
    capabilities = _capability_metrics(
        browser_view,
        target_title=target_title,
        target_as_of=str(case.get("wikipedia_as_of") or ""),
        cutoff_reference=str(case.get("wikipedia_before") or ""),
        accepted_answers=list(case.get("new_answer_keywords") or []),
    )
    bridge = _critical_bridge_evidence_metrics(browser_view, case)
    action_funnel = _posthoc_private_action_funnel(result, case)
    accepted = {
        " ".join(str(value).casefold().split())
        for value in case.get("new_answer_keywords", [])
    }
    submitted = " ".join(result.final_state.submitted_answer.casefold().split())
    expected_nodes = {
        (str(hop.get("source_title", "")).casefold(), hop.get("source_revision_id"))
        for hop in case.get("reasoning_chain", [])
    }
    expected_titles = {title for title, _ in expected_nodes}
    visited = set(result.final_state.visited_nodes)
    exhausted_without_answer = not submitted and result.stop_reason in {
        "max_expansions", "exhausted_search", "all_retained_finished",
    }
    return {
        **bridge,
        **action_funnel,
        "target_date_wikipedia_evidence_seen": capabilities[
            "target_snapshot_evidence_seen"
        ],
        "tail_composition_complete": "requires_independent_answer_judge",
        "wrong_page_failure": bool(
            exhausted_without_answer
            and any(title not in expected_titles for title, _ in visited)
        ),
        "wrong_revision_failure": bool(
            exhausted_without_answer
            and any(title in expected_titles and node not in expected_nodes
                    for node in visited for title in [node[0]])
        ),
        "final_answer_alias_match": submitted in accepted,
        "acquisition_candidate_before_independent_judge": bool(
            submitted in accepted and bridge["critical_bridge_evidence_complete"]
        ),
        "formal_success": "requires_independent_answer_judge",
    }


def _action_matches_expected(
    action: dict[str, Any], expected: dict[str, Any],
) -> bool:
    if action.get("kind") != expected["kind"]:
        return False
    params = action.get("params")
    if not isinstance(params, dict):
        return False
    if expected["kind"] == "FOLLOW_LINK":
        return (
            str(params.get("page_title", "")).casefold()
            == str(expected["page_title"]).casefold()
        )
    if expected["kind"] == "SWITCH_SNAPSHOT":
        revision_id = params.get("revision_id")
        if revision_id is None:
            return False
        try:
            return int(revision_id) == int(expected["revision_id"])
        except (TypeError, ValueError):
            return False
    if expected["kind"] == "SUBMIT_ANSWER":
        answer = " ".join(str(params.get("answer", "")).casefold().split())
        return answer in expected["accepted_answers"]
    return False


def _metric_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _posthoc_private_action_funnel(
    result: BeamSearchResult, case: dict[str, Any],
) -> dict[str, Any]:
    """Score the private route after search without affecting its decisions."""
    expected_actions: list[dict[str, Any]] = []
    chain = case.get("reasoning_chain")
    if not isinstance(chain, list):
        chain = []
    for hop in chain:
        if not isinstance(hop, dict):
            continue
        source_title = str(hop.get("source_title", ""))
        source_revision = hop.get("source_revision_id")
        prior_revision = hop.get("prior_revision_id")
        if source_title and source_revision is not None and prior_revision is not None:
            expected_actions.append({
                "kind": "SWITCH_SNAPSHOT",
                "parent_page": source_title,
                "parent_revision_id": int(prior_revision),
                "revision_id": int(source_revision),
                "hop_index": hop.get("index"),
            })
        target_title = str(hop.get("target_title", ""))
        if source_title and target_title and source_revision is not None:
            expected_actions.append({
                "kind": "FOLLOW_LINK",
                "parent_page": source_title,
                "parent_revision_id": int(source_revision),
                "page_title": target_title,
                "hop_index": hop.get("index"),
            })
    if chain:
        tail = chain[-1]
        aliases = {
            " ".join(str(value).casefold().split())
            for value in case.get("new_answer_keywords", []) if str(value).strip()
        }
        if aliases and isinstance(tail, dict) and tail.get("source_revision_id") is not None:
            expected_actions.append({
                "kind": "SUBMIT_ANSWER",
                "parent_page": str(tail.get("source_title", "")),
                "parent_revision_id": int(tail["source_revision_id"]),
                "accepted_answers": aliases,
                "hop_index": "submit",
            })

    details: list[dict[str, Any]] = []
    for expected in expected_actions:
        matching_steps = []
        for step in result.audit_steps:
            parent = step.get("parent_state")
            if not isinstance(parent, dict):
                continue
            if (
                str(parent.get("current_page", "")).casefold()
                == str(expected["parent_page"]).casefold()
                and parent.get("current_revision_id") == expected["parent_revision_id"]
            ):
                matching_steps.append(step)
        detail: dict[str, Any] = {
            "kind": expected["kind"],
            "hop_index": expected["hop_index"],
            "parent_page": expected["parent_page"],
            "parent_revision_id": expected["parent_revision_id"],
            "state_observed": bool(matching_steps),
            "legal_candidate_present": False,
            "post_compaction_present": False,
            "ranker_covered": False,
            "beam_retained": False,
        }
        if expected["kind"] == "FOLLOW_LINK":
            detail["expected_page_title"] = expected["page_title"]
        elif expected["kind"] == "SWITCH_SNAPSHOT":
            detail["expected_revision_id"] = expected["revision_id"]
        else:
            detail["accepted_answer_count"] = len(expected["accepted_answers"])
        for step in matching_steps:
            compaction = step.get("action_compaction")
            if not isinstance(compaction, dict):
                continue
            pre = compaction.get("pre_compaction_actions", [])
            post = compaction.get("post_compaction_actions", [])
            pre_matches = [row for row in pre if _action_matches_expected(row, expected)]
            post_matches = [row for row in post if _action_matches_expected(row, expected)]
            detail["legal_candidate_present"] |= bool(pre_matches)
            detail["post_compaction_present"] |= bool(post_matches)
            post_ids = {row.get("action_id") for row in post_matches}
            scored_ids = {
                row.get("action_id") for row in step.get("candidate_actions", [])
            }
            step_covered = bool(
                post_ids and post_ids.issubset(scored_ids)
            )
            detail["ranker_covered"] |= step_covered
            if step_covered:
                detail["beam_retained"] |= any(
                    row.get("action_id") in post_ids and row.get("retained") is True
                    for row in step.get("candidate_actions", [])
                )
        details.append(detail)

    observed = [row for row in details if row["state_observed"]]
    legal = [row for row in observed if row["legal_candidate_present"]]
    compacted = [row for row in legal if row["post_compaction_present"]]
    covered = [row for row in compacted if row["ranker_covered"]]
    retained = [row for row in covered if row["beam_retained"]]
    failure = "NO_OBSERVED_PRIVATE_ROUTE_OPPORTUNITY"
    if observed:
        if len(legal) < len(observed):
            failure = "LEGAL_CANDIDATE_RECALL_FAILURE"
        elif len(compacted) < len(legal):
            failure = "COMPACTION_RECALL_FAILURE"
        elif len(covered) < len(compacted):
            failure = "RANKER_OUTPUT_COMPLETENESS_FAILURE"
        elif len(retained) < len(covered):
            failure = "BEAM_PRUNING_FAILURE"
        else:
            failure = "ACTION_FUNNEL_PASSED_FOR_OBSERVED_OPPORTUNITIES"
    return {
        "legal_candidate_recall": _metric_ratio(len(legal), len(observed)),
        "post_compaction_recall@30": _metric_ratio(len(compacted), len(legal)),
        "ranker_coverage": _metric_ratio(len(covered), len(compacted)),
        "beam_recall@k": _metric_ratio(len(retained), len(covered)),
        "private_route_observed_opportunity_count": len(observed),
        "private_route_expected_action_count": len(details),
        "private_route_action_audit": details,
        "action_funnel_failure_class": failure,
        "private_route_usage": "posthoc_scoring_only_not_available_to_search",
    }


def _run_external(
    request: TemporalSearchRequest, model: str, backend: Any, max_actions: int,
) -> dict[str, Any]:
    return run_temporal_browsing(
        model=model,
        backend=backend,
        start_title=request.start_page,
        question=request.question,
        allowed_as_of=[request.cutoff_date, request.target_date],
        max_steps=max_actions,
        target_title="__HIDDEN_TARGET__",
        target_as_of=request.target_date,
        snapshot_date_range=(request.cutoff_date, request.target_date),
        reveal_target_title=False,
        cutoff_reference=request.cutoff_date,
        temperature=0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Temporal graph-constrained beam-search engineering runner"
    )
    parser.add_argument("--public-cases", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arms", default="A,B,C,D")
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--private-cases")
    parser.add_argument("--pk-results")
    parser.add_argument("--pk-contract-hash")
    parser.add_argument("--human-review-file")
    parser.add_argument(
        "--waive-human-review", action="store_true",
        help="explicit review waiver, applied only after a case passes frozen PK",
    )
    parser.add_argument("--max-expansions", type=int, default=16)
    parser.add_argument("--max-actions-per-state", type=int, default=4)
    parser.add_argument(
        "--max-links", type=int, default=20,
        help="gold-free document-order compaction before the <=30-action API scorer",
    )
    parser.add_argument("--revision-limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-path", default="wikipedia_snapshot.db")
    parser.add_argument("--ranker-cache", default="temporal_beam_ranker.db")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--request-interval", type=float, default=0.1)
    parser.add_argument("--output", default="temporal_beam_results.jsonl")
    args = parser.parse_args()
    assert_new_output_path(args.output)
    if Path(args.output).exists():
        parser.error(f"refusing to append to existing output {args.output}")
    try:
        public_cases = load_public_cases(args.public_cases)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if any(model != args.model for _, model in public_cases):
        parser.error("every public case must be bound to --model")
    arms = [value.strip().upper() for value in args.arms.split(",") if value.strip()]
    if not arms or set(arms) - set(ARM_WIDTHS):
        parser.error("--arms must be a comma-separated subset of A,B,C,D")
    if not args.engineering_smoke and not (
        args.private_cases and args.pk_results and args.pk_contract_hash
    ):
        parser.error(
            "formal mode requires --private-cases, --pk-results, and --pk-contract-hash"
        )
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is required")

    private_by_id: dict[str, dict[str, Any]] = {}
    if args.private_cases:
        payload = json.loads(Path(args.private_cases).read_text(encoding="utf-8"))
        private_by_id = {str(row["id"]): row for row in payload.get("cases", [])}
    admission_by_id: dict[str, dict[str, Any]] = {}
    review_by_id: dict[str, dict[str, Any]] = {}
    if not args.engineering_smoke:
        # Frozen PK is checked before review resolution and before opening any
        # graph/API backend.  Rejected cases therefore cannot enter A/B/C/D.
        admitted_private = []
        try:
            for request, model in public_cases:
                private_case = private_by_id.get(request.case_id)
                if private_case is None:
                    raise ValueError(f"missing private case {request.case_id}")
                admission_by_id[request.case_id] = validate_formal_admission(
                    case=private_case, model=model, pk_results=args.pk_results,
                    contract_hash=args.pk_contract_hash,
                )
                admitted_private.append(private_case)
            review_by_id = resolve_human_reviews(
                admitted_private, review_file=args.human_review_file,
                waive_human_review=args.waive_human_review,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
    backend = WikipediaPageBackend(
        cache_path=args.cache_path,
        offline_only=args.offline,
        min_request_interval=args.request_interval,
    )
    ranker = ApiUtilityRanker(
        args.model, cache_path=args.ranker_cache,
    )
    writer = _JsonlWriter(args.output)
    written_evidence_ids: set[str] = set()
    try:
        for request, model in public_cases:
            private_case = private_by_id.get(request.case_id)
            if args.engineering_smoke:
                admission = {
                    "status": "not_run_engineering_smoke",
                    "formal_eligible": False,
                }
                review = {
                    "decision": "not_run_engineering_smoke",
                    "formal_eligible": False,
                }
            else:
                admission = admission_by_id[request.case_id]
                review = review_by_id[request.case_id]
            for arm in arms:
                if arm == "A":
                    baseline = _run_external(
                        request, model, backend, args.max_expansions,
                    )
                    writer.write(
                        slot="external_agent_summary",
                        case_id=request.case_id,
                        model=model,
                        arm=arm,
                        arm_name="external_tool_agent_baseline",
                        admission=admission,
                        human_review=review,
                        request=request.to_dict(),
                        max_actions=args.max_expansions,
                        result=baseline,
                        metrics=external_engineering_metrics(
                            baseline, request.target_date,
                        ),
                        formal_conclusion_allowed=False,
                    )
                    continue
                width = ARM_WIDTHS[arm]
                assert width is not None
                config = BeamSearchConfig(
                    beam_width=width,
                    max_expansions=args.max_expansions,
                    max_actions_per_state=args.max_actions_per_state,
                    max_links=args.max_links,
                    revision_limit=args.revision_limit,
                    seed=args.seed,
                )
                result = run_temporal_beam_search(request, backend, ranker, config)
                evidence: dict[str, dict[str, Any]] = {}
                _collect_evidence(result.audit_steps, evidence)
                _collect_evidence(result.final_state.to_dict(), evidence)
                for evidence_id, page in sorted(evidence.items()):
                    if evidence_id in written_evidence_ids:
                        continue
                    writer.write(
                        slot="beam_evidence",
                        evidence_id=evidence_id,
                        case_id=request.case_id,
                        model=model,
                        arm=arm,
                        page=page,
                    )
                    written_evidence_ids.add(evidence_id)
                for step in result.audit_steps:
                    writer.write(
                        slot="beam_expansion",
                        beam_schema_version=BEAM_TRAJECTORY_SCHEMA,
                        case_id=request.case_id,
                        model=model,
                        arm=arm,
                        beam_width=width,
                        admission=admission,
                        human_review=review,
                        **_compact_audit_value(step),
                    )
                metrics = engineering_metrics(result)
                if private_case is not None and not args.engineering_smoke:
                    metrics.update(posthoc_private_metrics(result, private_case))
                writer.write(
                    slot="beam_summary",
                    beam_schema_version=BEAM_TRAJECTORY_SCHEMA,
                    case_id=request.case_id,
                    model=model,
                    arm=arm,
                    arm_name=(
                        "temporal_constrained_greedy" if width == 1
                        else "temporal_constrained_beam"
                    ),
                    beam_width=width,
                    admission=admission,
                    human_review=review,
                    request=request.to_dict(),
                    config=asdict(config),
                    final_state=_compact_audit_value(result.final_state.to_dict()),
                    retained_state_ids=[state.state_id for state in result.retained_states],
                    metrics=metrics,
                    formal_conclusion_allowed=False,
                )
    finally:
        writer.close()
        ranker.close()
        backend.close()
    print(f"[done] temporal beam results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
