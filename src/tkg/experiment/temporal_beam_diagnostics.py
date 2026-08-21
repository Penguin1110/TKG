"""Non-scoring forced-state diagnostics for temporal graph search.

Private chains are used only by this offline controller to construct and score a
forced state.  The model receives Wikipedia-derived text, a public/direct probe,
and unlabelled legal actions; expected actions and answers are never put in its
context.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_beam import (
    AnswerProposal, RevisionOption, TemporalAction, TemporalBeamState,
    TemporalSearchRequest, _fold, _score_actions,
    submit_action_for_proposal, validate_answer_proposal,
)
from tkg.experiment.temporal_beam_ranker import ApiUtilityRanker
from tkg.wikipedia.backend import normalize_title


DIAGNOSTIC_SCHEMA = "temporal-forced-state-diagnostic-v1"


def _question(case: dict[str, Any]) -> str:
    writer = case.get("_generation", {}).get("question_writer", {})
    value = writer.get("question") or case.get("question")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{case.get('id')}: missing public question")
    return " ".join(value.split())


def _pages(case: dict[str, Any]) -> list[dict[str, Any]]:
    pages = case.get("frozen_wikipedia_evidence", {}).get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError(f"{case.get('id')}: missing frozen Wikipedia evidence")
    return pages


def _page(
    pages: list[dict[str, Any]], title: str, revision_id: int,
) -> dict[str, Any]:
    for page in pages:
        if _fold(str(page.get("title", ""))) == _fold(title) and int(
            page.get("revision_id", -1)
        ) == revision_id:
            return page
    raise ValueError(f"frozen page not found: {title}@{revision_id}")


def _visible_excerpt(page: dict[str, Any], excerpt: str) -> dict[str, Any]:
    """Bind a validated excerpt to its public revision metadata."""
    return {
        "title": page["title"],
        "revision_id": page["revision_id"],
        "timestamp": page["timestamp"],
        "as_of": page["as_of"],
        "content": excerpt,
        "links": page.get("links", []),
        "diagnostic_slice_policy": "validated_chain_evidence_excerpt",
    }


def _state(
    page: dict[str, Any], evidence: list[dict[str, Any]], request: TemporalSearchRequest,
    *, known_revisions: tuple[RevisionOption, ...] = (),
) -> TemporalBeamState:
    nodes = tuple(dict.fromkeys(
        (_fold(str(row["title"])), int(row["revision_id"])) for row in evidence
    ))
    return TemporalBeamState(
        current_page=str(page["title"]),
        current_revision_id=int(page["revision_id"]),
        current_revision_date=str(page["timestamp"])[:10],
        snapshot_as_of=str(page["as_of"]),
        reasoning_summary="",
        extracted_entities=(),
        temporal_constraints={
            "range_start": request.cutoff_date,
            "range_end": request.target_date,
            "event_time_policy": "world_event_or_tenure_onset_not_observation_time",
        },
        collected_evidence=tuple(evidence),
        known_revisions=known_revisions,
        visited_nodes=nodes,
        action_trace=(),
        cumulative_score=0.0,
        finished=False,
        submitted_answer="",
    )


def _link_actions(page: dict[str, Any], *, max_links: int) -> list[TemporalAction]:
    content = str(page.get("content", ""))
    links: dict[str, tuple[str, str]] = {}
    ordered_links = []
    for index, link in enumerate(page.get("links", [])):
        target = normalize_title(str(link["target"]))
        anchor = str(link.get("anchor", target))
        marker = f"[{anchor} -> {link['target']}]"
        position = content.find(marker)
        ordered_links.append((position if position >= 0 else len(content) + index, link))
    for _, link in sorted(ordered_links, key=lambda row: row[0]):
        target = normalize_title(str(link["target"]))
        links.setdefault(_fold(target), (target, str(link.get("anchor", target))))
    return [
        TemporalAction(
            "FOLLOW_LINK", {"page_title": target},
            f'Follow hyperlink "{anchor}" to {target}',
        )
        for target, anchor in list(links.values())[:max_links]
    ]


def _compact_to_dense_limit(
    link_actions: list[TemporalAction], trailing_actions: list[TemporalAction],
    *, limit: int = 30,
) -> tuple[list[TemporalAction], dict[str, Any]]:
    kept_links = link_actions[:max(0, limit - len(trailing_actions))]
    compacted = [*kept_links, *trailing_actions]
    full = [*link_actions, *trailing_actions]
    return compacted, {
        "policy": "visible_document_order_first_n_reserving_nonlink_actions_v1",
        "dense_limit": limit,
        "pre_compaction_count": len(full),
        "post_compaction_count": len(compacted),
        "pre_compaction_actions": [action.to_dict() for action in full],
        "post_compaction_actions": [action.to_dict() for action in compacted],
    }


def _funnel_fields(
    expected: TemporalAction | None, compaction: dict[str, Any],
    scored: list[dict[str, Any]], ranker_error: str,
) -> dict[str, Any]:
    expected_id = expected.action_id if expected else None
    pre_ids = {row["action_id"] for row in compaction["pre_compaction_actions"]}
    post_ids = {row["action_id"] for row in compaction["post_compaction_actions"]}
    legal = bool(expected_id and expected_id in pre_ids)
    post = bool(expected_id and expected_id in post_ids)
    covered = bool(post and not ranker_error and any(
        row["action_id"] == expected_id for row in scored
    ))
    return {
        "action_compaction": compaction,
        "legal_candidate_recall": legal,
        "post_compaction_recall@30": post,
        "ranker_coverage": covered,
        "beam_recall@k": None,
        "posthoc_failure_class": (
            "LEGAL_CANDIDATE_RECALL_FAILURE" if not legal
            else "COMPACTION_RECALL_FAILURE" if not post
            else "RANKER_OUTPUT_COMPLETENESS_FAILURE" if not covered
            else "FORCED_RANKING_OBSERVED_BEAM_NOT_RUN"
        ),
    }


def _rank(
    ranker: ApiUtilityRanker, request: TemporalSearchRequest,
    state: TemporalBeamState, actions: list[TemporalAction], seed: int,
) -> tuple[list[dict[str, Any]], Any | None, str]:
    try:
        output = ranker.rank(
            request, state, list(state.collected_evidence), actions, seed=seed,
        )
    except ValueError as exc:
        return [], None, str(exc)
    scored = sorted(
        _score_actions(actions, output),
        key=lambda row: (-float(row["action_score"]), row["action_id"]),
    )
    return scored, output, ""


def _answer_result(
    proposal: AnswerProposal, expected_aliases: list[str],
) -> dict[str, Any]:
    expected = {_fold(value).strip(".?!") for value in expected_aliases}
    actual = _fold(proposal.answer).strip(".?!")
    return {
        "answer_candidate": asdict(proposal),
        "literal_support_gate_passed": proposal.supported,
        "semantic_relation_support": "not_evaluated",
        "posthoc_expected_aliases": expected_aliases,
        "posthoc_alias_match": actual in expected,
    }


def run_case_diagnostics(
    case: dict[str, Any], ranker: ApiUtilityRanker, *, seed: int, max_links: int,
) -> list[dict[str, Any]]:
    chain = case.get("reasoning_chain")
    if not isinstance(chain, list) or len(chain) < 4:
        raise ValueError(f"{case.get('id')}: diagnostics require a four-hop chain")
    pages = _pages(case)
    public_request = TemporalSearchRequest(
        case_id=str(case["id"]),
        question=_question(case),
        start_page=str(chain[0]["source_title"]),
        cutoff_date=str(chain[0]["as_of"]),
        target_date=str(case["wikipedia_as_of"]),
    )
    bridge = chain[-2]
    target_page = _page(
        pages, str(bridge["source_title"]), int(bridge["source_revision_id"]),
    )
    prior_page = _page(
        pages, str(bridge["source_title"]), int(bridge["prior_revision_id"]),
    )
    target_option = RevisionOption(
        revision_id=int(target_page["revision_id"]),
        revision_date=str(target_page["timestamp"])[:10],
        revision_timestamp=str(target_page["timestamp"]),
        as_of=str(target_page["as_of"]),
    )
    prior_option = RevisionOption(
        revision_id=int(prior_page["revision_id"]),
        revision_date=str(prior_page["timestamp"])[:10],
        revision_timestamp=str(prior_page["timestamp"]),
        as_of=str(prior_page["as_of"]),
    )
    rows: list[dict[str, Any]] = []

    # 1a. Forced correct page, old revision: can the ranker choose the revision?
    revision_state = _state(
        prior_page, [prior_page], public_request,
        known_revisions=(prior_option, target_option),
    )
    revision_links = _link_actions(prior_page, max_links=max_links)
    expected_revision_action = TemporalAction(
        "SWITCH_SNAPSHOT",
        {
            "revision_id": target_option.revision_id,
            "revision_date": target_option.revision_date,
            "as_of": target_option.as_of,
        },
        f"Switch this page to revision {target_option.revision_id} "
        f"({target_option.revision_date})",
    )
    revision_actions, revision_compaction = _compact_to_dense_limit(
        revision_links, [expected_revision_action],
    )
    scored, output, ranker_error = _rank(
        ranker, public_request, revision_state, revision_actions, seed,
    )
    expected_id = expected_revision_action.action_id
    rows.append({
        "diagnostic": "navigation_revision",
        "model_visible": {
            "question": public_request.question,
            "state": revision_state.to_dict(),
            "candidate_actions": [action.to_dict() for action in revision_actions],
        },
        "ranked_actions": scored,
        "ranker_call_valid": not ranker_error,
        "ranker_error": ranker_error,
        "ranker_raw": output.raw if output else None,
        "posthoc_expected_action_id": expected_id,
        "posthoc_expected_selected": bool(scored and scored[0]["action_id"] == expected_id),
        **_funnel_fields(
            expected_revision_action, revision_compaction, scored, ranker_error,
        ),
    })

    # 1b. Forced correct target revision: can it choose the linked next entity?
    page_state = _state(target_page, [target_page], public_request)
    full_page_actions = _link_actions(target_page, max_links=max_links)
    page_actions, page_compaction = _compact_to_dense_limit(full_page_actions, [])
    scored, output, ranker_error = _rank(
        ranker, public_request, page_state, page_actions, seed,
    )
    expected_title = normalize_title(str(bridge["target_title"]))
    expected = next(
        (action for action in full_page_actions
         if _fold(str(action.params.get("page_title", ""))) == _fold(expected_title)),
        None,
    )
    rows.append({
        "diagnostic": "navigation_page",
        "model_visible": {
            "question": public_request.question,
            "state": page_state.to_dict(),
            "candidate_actions": [action.to_dict() for action in page_actions],
        },
        "ranked_actions": scored,
        "ranker_call_valid": not ranker_error,
        "ranker_error": ranker_error,
        "ranker_raw": output.raw if output else None,
        "posthoc_expected_action_id": expected.action_id if expected else None,
        "posthoc_expected_action_present": expected is not None,
        "posthoc_expected_selected": bool(
            expected and scored and scored[0]["action_id"] == expected.action_id
        ),
        **_funnel_fields(expected, page_compaction, scored, ranker_error),
    })

    # 2. Direct extraction from the correct target-revision evidence excerpt.
    bridge_excerpt = _visible_excerpt(target_page, str(bridge["evidence"]))
    extraction_request = TemporalSearchRequest(
        case_id=public_request.case_id + ":extraction",
        question=(
            "According only to the visible target revision, who is shown as the "
            "incumbent officeholder?"
        ),
        start_page=public_request.start_page,
        cutoff_date=public_request.cutoff_date,
        target_date=public_request.target_date,
    )
    extraction_state = _state(target_page, [bridge_excerpt], extraction_request)
    extraction = validate_answer_proposal(
        ranker.propose_answer(
            extraction_request, extraction_state, [bridge_excerpt], seed=seed,
        ),
        extraction_state.collected_evidence,
    )
    rows.append({
        "diagnostic": "evidence_extraction",
        "model_visible": {
            "question": extraction_request.question,
            "evidence": [bridge_excerpt],
        },
        **_answer_result(extraction, list(bridge.get("target_aliases", []))),
    })

    # 3. Full chain excerpts: can it compose, generate, and rank SUBMIT_ANSWER?
    composed_evidence: list[dict[str, Any]] = []
    for hop in chain:
        source = _page(
            pages, str(hop["source_title"]), int(hop["source_revision_id"]),
        )
        excerpt = _visible_excerpt(source, str(hop["evidence"]))
        key = (excerpt["title"], excerpt["revision_id"])
        if key not in {(row["title"], row["revision_id"]) for row in composed_evidence}:
            composed_evidence.append(excerpt)
    current = composed_evidence[-1]
    composition_state = _state(current, composed_evidence, public_request)
    composition = validate_answer_proposal(
        ranker.propose_answer(
            public_request, composition_state, composed_evidence, seed=seed,
        ),
        composition_state.collected_evidence,
    )
    graph_actions = _link_actions(current, max_links=max_links)
    submit = submit_action_for_proposal(composition)
    combined, composition_compaction = _compact_to_dense_limit(
        graph_actions, [submit] if submit else [],
    )
    scored, output, ranker_error = _rank(
        ranker, public_request, composition_state, combined, seed,
    )
    submit_rank = next(
        (index + 1 for index, row in enumerate(scored)
         if row["kind"] == "SUBMIT_ANSWER"),
        None,
    )
    rows.append({
        "diagnostic": "composition_termination",
        "model_visible": {
            "question": public_request.question,
            "evidence": composed_evidence,
            "candidate_actions": [action.to_dict() for action in combined],
        },
        **_answer_result(composition, list(case.get("new_answer_keywords", []))),
        "submit_candidate_created": submit is not None,
        "submit_rank": submit_rank,
        "submit_selected": submit_rank == 1,
        "ranker_call_valid": not ranker_error,
        "ranker_error": ranker_error,
        "ranked_actions": scored,
        "ranker_raw": output.raw if output else None,
        **_funnel_fields(submit, composition_compaction, scored, ranker_error),
    })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run non-scoring forced-state diagnostics")
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ranker-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-links", type=int, default=500)
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is required")
    assert_new_output_path(args.output)
    cases = []
    for path in args.cases:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cases.extend(payload.get("cases", []))
    ranker = ApiUtilityRanker(args.model, cache_path=args.ranker_cache)
    try:
        with Path(args.output).open("x", encoding="utf-8") as fh:
            for case in cases:
                for row in run_case_diagnostics(
                    case, ranker, seed=args.seed, max_links=args.max_links,
                ):
                    record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "schema_version": DIAGNOSTIC_SCHEMA,
                        "non_scoring": True,
                        "case_id": case["id"],
                        "model": args.model,
                        "ranker_boundary": ranker.ranker_name,
                        **row,
                    }
                    fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    fh.flush()
    finally:
        ranker.close()
    print(f"[done] non-scoring forced diagnostics: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
