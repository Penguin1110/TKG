"""Five-case open-weight A/B/C/D end-to-end engineering pilot.

Arm A is a free external-tool loop driven by the same local checkpoint. Arms
B/C/D use the frozen hierarchical conditional-logprob controller at beam widths
1/3/5. Private cases enter only post-hoc evaluation and failure diagnosis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from tkg.experiment.compact_submission_v24 import (
    compact_submission_from_dict_v24, evaluate_compact_submission_posthoc_v24,
)
from tkg.experiment.open_weight_action_scorer_v24 import (
    HuggingFaceCausalLMBackendV24, OpenWeightConditionalActionScorerV24,
)
from tkg.experiment.open_weight_answer_generator_v24 import (
    EvidenceConditionedAnswerGeneratorV24,
)
from tkg.experiment.open_weight_live_controller_v24 import (
    HierarchicalOpenWeightLiveControllerV24,
)
from tkg.experiment.temporal_environment_v2 import TemporalWikipediaEnvironmentV2
from tkg.experiment.temporal_eval_schema_v2 import (
    EvaluationCaseV2, StructuredSubmissionV2, SubmittedClaimV2, load_cases_v2,
    normalized,
)
from tkg.experiment.temporal_evaluation_v2 import (
    evidence_id_v2, validate_structured_submission_v2,
)
from tkg.experiment.temporal_live_runner_v23 import LiveSearchConfigV23
from tkg.experiment.temporal_live_runner_v24 import run_live_temporal_search_v24
from tkg.experiment.temporal_semantic_judge_v2 import LLMSemanticClaimJudgeV2
from tkg.wikipedia.backend import WikipediaPageBackend
from tkg.wikipedia.browser import run_temporal_browsing


SCHEMA_VERSION = "open-weight-abcd-engineering-pilot-v2.5"
DEFAULT_CASE_IDS = (
    "promoted_c4e2822024602c2c4345",       # bounded no-shortcut signal
    "promoted_00f4be76d3b2d0377623",       # temporal-valid alternatives only
    "promoted_237ae89c2f939b87444c_p19",  # answer-only + valid alternative
    "promoted_4645b1b1444731a0ddc8",       # answer-only-heavy, citizenship tail
    "promoted_590bebda8b580ddf5dd5",       # answer-only + award tail
)
ARM_WIDTHS = {"A": None, "B": 1, "C": 3, "D": 5}
ARM_LOCAL_EXPANSIONS = {"A": None, "B": 1, "C": 3, "D": 5}
FROZEN_SOURCE_HASHES = {
    "src/tkg/experiment/open_weight_action_scorer_v24.py": "d20e21f380b9d1388d2d547836d0219a0c43855e7efc7563d63ee2fa29ae1471",
    "src/tkg/experiment/open_weight_answer_generator_v24.py": "c0fcecee3629a926b053b980f1861ca788357f2ae4f1c01584c7f2046f6b2539",
    "src/tkg/experiment/open_weight_live_controller_v24.py": "21a62af769e548da560893e7174e8ad3f187c925c69a052bf664c69aa212b978",
    "src/tkg/experiment/temporal_live_runner_v24.py": "3785a6f38300548b72c871fbbfae369f65a8ba8740ffd9ab76120d5e15bc0817",
    "src/tkg/experiment/open_weight_independent_gate_v24.py": "e92800a8ae2768e42d37b2dda8880be460d321c3b0f5390a5e5f877932aaf684",
}
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_frozen_method(repo_root: Path) -> dict[str, str]:
    actual = {name: _sha(repo_root / name) for name in FROZEN_SOURCE_HASHES}
    if actual != FROZEN_SOURCE_HASHES:
        changed = sorted(name for name in actual if actual[name] != FROZEN_SOURCE_HASHES[name])
        raise ValueError(f"frozen method sources changed: {changed}")
    return actual


def _extract_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


class MicrobatchedLogprobBackendV25:
    """Memory-bounded execution wrapper with unchanged per-action scores."""

    def __init__(self, backend: HuggingFaceCausalLMBackendV24, batch_size: int):
        if batch_size <= 0:
            raise ValueError("log-probability microbatch size must be positive")
        self.backend = backend
        self.batch_size = batch_size
        self.backend_name = (
            f"{backend.backend_name}_microbatch_{batch_size}_v2.5"
        )

    def generate_text(
        self, prompt: str, *, max_new_tokens: int = 192,
        system_prompt: str | None = None,
    ) -> str:
        return self.backend.generate_text(
            prompt, max_new_tokens=max_new_tokens, system_prompt=system_prompt,
        )

    def conditional_token_logprobs(
        self, prompt: str, continuation: str,
    ) -> list[float]:
        return self.backend.conditional_token_logprobs(prompt, continuation)

    def conditional_token_logprobs_batch(
        self, prompt: str, continuations: list[str],
    ) -> list[list[float]]:
        result = []
        for start in range(0, len(continuations), self.batch_size):
            result.extend(self.backend.conditional_token_logprobs_batch(
                prompt, continuations[start:start + self.batch_size],
            ))
        return result


class OpenWeightToolCallerV25:
    """OpenAI-tool-shaped adapter around the same checkpoint's text decoder."""

    def __init__(self, backend: HuggingFaceCausalLMBackendV24):
        self.backend = backend
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _compact_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept = messages[:2] + messages[-10:] if len(messages) > 12 else messages
        result = []
        for row in kept:
            content = row.get("content")
            result.append({
                "role": row.get("role"),
                "content": str(content or "")[:10_000],
            })
        return result

    def __call__(
        self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        del model, temperature
        public_tools = [{
            "name": row["function"]["name"],
            "description": row["function"].get("description", ""),
            "parameters": row["function"].get("parameters", {}),
        } for row in tools]
        prompt = (
            "Choose exactly one tool call. Use only the visible conversation and tool contract. "
            "Do not invent a hyperlink or revision. Return exactly one JSON object: "
            '{"tool":"tool_name","arguments":{},"content":"brief reason"}.\n'
            "TOOLS:\n" + json.dumps(public_tools, ensure_ascii=False, sort_keys=True) +
            "\nCONVERSATION:\n" + json.dumps(
                self._compact_messages(messages), ensure_ascii=False, sort_keys=True,
            )
        )
        raw = self.backend.generate_text(
            prompt, max_new_tokens=192,
            system_prompt="You are a strict Wikipedia tool-calling controller. JSON only.",
        )
        parsed = _extract_object(raw)
        names = {row["name"] for row in public_tools}
        result: dict[str, Any]
        if not parsed or parsed.get("tool") not in names:
            result = {"role": "assistant", "content": raw, "tool_calls": []}
        else:
            arguments = parsed.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            result = {
                "role": "assistant", "content": str(parsed.get("content") or ""),
                "tool_calls": [{
                    "id": "local_" + uuid.uuid4().hex,
                    "type": "function",
                    "function": {
                        "name": parsed["tool"],
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }],
            }
        self.calls.append({
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "raw_response": raw, "parsed": parsed,
            "contract_valid": bool(result["tool_calls"]),
        })
        return result


def _with_evidence_ids(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for original in pages:
        page = dict(original)
        page["evidence_id"] = str(page.get("evidence_id") or evidence_id_v2(page))
        if page["evidence_id"] not in seen:
            seen.add(page["evidence_id"])
            result.append(page)
    return result


def _oracle_evidence_probe(
    case: EvaluationCaseV2, evidence: list[dict[str, Any]], *, actions_valid: bool,
    judge: LLMSemanticClaimJudgeV2,
) -> dict[str, Any]:
    ids = tuple(page["evidence_id"] for page in evidence)
    answer = case.accepted_final_answer_aliases[0]
    submission = StructuredSubmissionV2(
        answer=answer,
        critical_claims=tuple(SubmittedClaimV2(
            subject=claim.subject, relation=claim.relation, object=claim.object,
            event_time=claim.event_time, supporting_evidence_ids=ids,
            claim_id=claim.claim_id,
        ) for claim in case.critical_claims),
        tail_claim=SubmittedClaimV2(
            subject=case.tail_relation.subject, relation=case.tail_relation.relation,
            object=answer, event_time=case.tail_relation.event_time,
            supporting_evidence_ids=ids, claim_id=case.tail_relation.claim_id,
        ),
    )
    return validate_structured_submission_v2(
        case=case, submission=submission, trajectory_evidence=evidence,
        trajectory_actions_valid=actions_valid, semantic_judge=judge,
    )


def _actions_valid_live(trace: list[dict[str, Any]]) -> bool:
    return not any(row.get("error") for row in trace) and all(
        row.get("hyperlink_valid") is not False
        and row.get("revision_valid") is not False
        and row.get("environment_query_valid") is not False
        for row in trace
    )


def _actions_valid_external(trajectory: list[dict[str, Any]]) -> bool:
    return not any(str(row.get("result") or "").startswith("Error:") for row in trajectory)


def _action_matches(action: dict[str, Any], expected: Any) -> bool:
    if action.get("kind") != expected.kind:
        return False
    params = action.get("params") or {}
    if expected.kind == "FOLLOW_LINK":
        return normalized(params.get("page_title")) == normalized(expected.page_title)
    if expected.kind == "SWITCH_SNAPSHOT":
        return params.get("revision_id") == expected.revision_id
    return False


def _failure_stage_live(result: Any, case: EvaluationCaseV2, metrics: dict[str, Any]) -> dict[str, Any]:
    if metrics["complete_evidence_submitted"]:
        return {"stage": "none", "reason": "end_to_end_success"}
    manifests = {row.manifest_id: row.to_dict() for row in result.environment_manifests}
    opportunities = []
    for expected in case.reference_routes[0].actions:
        expected_opportunities: list[dict[str, Any]] = []
        pagination_opportunities: list[dict[str, Any]] = []
        for step in result.audit_steps:
            parent = step.get("parent_state") or {}
            if (
                normalized(parent.get("current_page")) != normalized(expected.parent_page)
                or parent.get("current_revision_id") != expected.parent_revision_id
            ):
                continue
            funnel = step.get("action_funnel") or {}
            manifest = manifests.get(funnel.get("environment_legal_actions_artifact_reference"), {})
            legal = any(_action_matches(row, expected) for row in manifest.get("actions", []))
            retrieved = any(_action_matches(row, expected) for row in funnel.get("solver_retrieved_actions", []))
            compacted = any(_action_matches(row, expected) for row in funnel.get("compacted_ranker_actions", []))
            matching = [row for row in step.get("candidate_actions", []) if _action_matches(row, expected)]
            selected = any(row.get("expanded") is True for row in matching)
            retained = any(row.get("retained") is True for row in matching)
            graph_candidates = [
                row for row in step.get("candidate_actions", [])
                if row.get("kind") != "SUBMIT_SLOT"
            ]
            ranked_graph = sorted(
                graph_candidates,
                key=lambda row: float(row.get("action_score", float("-inf"))),
                reverse=True,
            )
            ranks = [
                index for index, row in enumerate(ranked_graph, start=1)
                if _action_matches(row, expected)
            ]
            opportunity = {
                "kind": expected.kind, "parent_page": expected.parent_page,
                "parent_revision_id": expected.parent_revision_id,
                "legal": legal, "retrieved": retrieved, "compacted": compacted,
                "ranked": bool(matching), "selected": selected, "retained": retained,
                "graph_rank": min(ranks) if ranks else None,
            }
            expected_opportunities.append(opportunity)
            pagination_kind = {
                "FOLLOW_LINK": "LIST_LINKS",
                "SWITCH_SNAPSHOT": "LIST_REVISIONS",
            }.get(expected.kind)
            if pagination_kind:
                pagination_opportunities.extend({
                    "kind": row.get("kind"),
                    "expanded": row.get("expanded") is True,
                    "retained": row.get("retained") is True,
                    "iteration": step.get("iteration"),
                } for row in step.get("candidate_actions", [])
                    if row.get("kind") == pagination_kind)
        opportunities.extend(expected_opportunities)
        if any(row["retained"] for row in expected_opportunities):
            continue
        expanded = [row for row in expected_opportunities if row["selected"]]
        if expanded:
            return {
                "stage": "beam", "reason": "expanded_progress_state_pruned",
                "opportunity": expanded[0],
            }
        ranked = [row for row in expected_opportunities if row["ranked"]]
        if ranked:
            best = min(
                ranked,
                key=lambda row: (
                    float(row["graph_rank"])
                    if isinstance(row.get("graph_rank"), (int, float))
                    else float("inf")
                ),
            )
            return {
                "stage": "ranking", "reason": "progress_action_below_expansion_cutoff",
                "opportunity": best,
            }
        pruned_pagination = [
            row for row in pagination_opportunities
            if row["expanded"] and not row["retained"]
        ]
        if pruned_pagination:
            return {
                "stage": "beam",
                "reason": "pagination_state_expanded_then_pruned_before_action_retrieval",
                "opportunity": expected_opportunities[-1] if expected_opportunities else None,
                "pagination": pruned_pagination[0],
            }
        return {
            "stage": "candidate", "reason": "legal_action_never_retrieved_or_compacted",
            "opportunity": expected_opportunities[-1] if expected_opportunities else None,
        }
    if metrics["bridge_found"] and metrics["tail_found"]:
        return {"stage": "submission", "reason": "complete_visible_evidence_not_validly_submitted"}
    if opportunities:
        return {"stage": "ranking", "reason": "route_diverged_after_retained_progress", "opportunities": opportunities}
    return {"stage": "candidate", "reason": "no_reference_route_opportunity_observed"}


def _score_live(
    result: Any, case: EvaluationCaseV2, judge: LLMSemanticClaimJudgeV2,
) -> dict[str, Any]:
    state = result.final_state
    evidence = _with_evidence_ids(list(state.collected_evidence))
    actions_valid = _actions_valid_live(list(state.action_trace))
    acquisition = _oracle_evidence_probe(case, evidence, actions_valid=actions_valid, judge=judge)
    submitted = state.submitted
    actual = None
    if isinstance(submitted, dict):
        try:
            compact = compact_submission_from_dict_v24(submitted)
            actual = evaluate_compact_submission_posthoc_v24(
                case=case, submission=compact, trajectory_evidence=evidence,
                trajectory_actions_valid=actions_valid, semantic_judge=judge,
            )
        except ValueError as exc:
            actual = {"end_to_end_success": False, "error": str(exc)}
    answer = str((submitted or {}).get("answer") or "") if isinstance(submitted, dict) else ""
    metrics = {
        "bridge_found": bool(acquisition["critical_bridge_evidence_complete"]),
        "bridges_acquired": acquisition["critical_bridges_acquired"],
        "bridge_count": acquisition["critical_bridge_count"],
        "tail_found": bool(acquisition["tail_claim_result"]["passed"]),
        "structured_submission_present": isinstance(submitted, dict),
        "complete_evidence_submitted": bool(actual and actual.get("end_to_end_success")),
        "final_answer": answer,
        "final_answer_correct": normalized(answer) in {
            normalized(value) for value in case.accepted_final_answer_aliases
        },
        "actions_valid": actions_valid, "expansions": result.expansions,
        "stop_reason": result.stop_reason,
        "acquisition_evaluation": acquisition,
        "actual_submission_evaluation": actual,
    }
    metrics["failure"] = _failure_stage_live(result, case, metrics)
    return metrics


def _score_external(
    result: dict[str, Any], case: EvaluationCaseV2,
    judge: LLMSemanticClaimJudgeV2,
) -> dict[str, Any]:
    evidence = _with_evidence_ids(list(result.get("evidence_pages") or []))
    actions_valid = _actions_valid_external(list(result.get("trajectory") or []))
    acquisition = _oracle_evidence_probe(case, evidence, actions_valid=actions_valid, judge=judge)
    answer = str(result.get("final_answer") or "")
    correct = normalized(answer) in {normalized(value) for value in case.accepted_final_answer_aliases}
    bridge = bool(acquisition["critical_bridge_evidence_complete"])
    tail = bool(acquisition["tail_claim_result"]["passed"])
    errors = [row for row in result.get("trajectory", []) if str(row.get("result") or "").startswith("Error:")]
    failure: dict[str, Any]
    if bridge and tail and not correct:
        failure = {"stage": "submission", "reason": "complete_evidence_but_answer_missing_or_wrong"}
    elif errors:
        failure = {"stage": "candidate", "reason": "invalid_tool_action", "errors": errors}
    elif not bridge or not tail:
        failure = {"stage": "ranking", "reason": "free_agent_action_selection_missed_required_evidence"}
    else:
        failure = {"stage": "none", "reason": "answer_correct_but_legacy_A_has_no_evidence_ID_submission"}
    return {
        "bridge_found": bridge,
        "bridges_acquired": acquisition["critical_bridges_acquired"],
        "bridge_count": acquisition["critical_bridge_count"],
        "tail_found": tail,
        "structured_submission_present": False,
        "complete_evidence_submitted": False,
        "complete_evidence_collected": bridge and tail,
        "submission_contract_note": "legacy external-tool A submits answer text only",
        "final_answer": answer, "final_answer_correct": correct,
        "actions_valid": actions_valid,
        "expansions": len(result.get("trajectory") or []),
        "stop_reason": result.get("stop_reason"),
        "acquisition_evaluation": acquisition, "failure": failure,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hydrate_live_result(value: dict[str, Any]) -> SimpleNamespace:
    """Hydrate only the public attributes required by post-hoc scoring.

    Raw search artifacts stay plain JSON so GPU search and API judging can run
    in separate environments without pickles or private-case inputs.
    """
    manifests = []
    for original in value.get("environment_manifests", []):
        row = dict(original)
        manifests.append(SimpleNamespace(
            manifest_id=row["manifest_id"],
            to_dict=lambda payload=row: payload,
        ))
    return SimpleNamespace(
        final_state=SimpleNamespace(**value["final_state"]),
        audit_steps=tuple(value.get("audit_steps", [])),
        environment_manifests=tuple(manifests),
        expansions=int(value.get("expansions", 0)),
        stop_reason=str(value.get("stop_reason") or ""),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--method-freeze", required=True)
    parser.add_argument("--shortcut-summary", required=True)
    parser.add_argument("--semantic-judge-model", default="openai/gpt-4.1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-expansions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--logprob-batch-size", type=int, default=4)
    parser.add_argument(
        "--arm", action="append", choices=tuple(ARM_WIDTHS), default=[],
        help="Run only selected arms; repeat for multiple arms.",
    )
    parser.add_argument(
        "--phase", choices=("all", "search", "score"), default="all",
        help="Run GPU search, local post-hoc scoring, or both.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path.cwd()
    source_hashes = verify_frozen_method(repo_root)
    method_freeze = _load_manifest(Path(args.method_freeze))
    shortcut_summary = _load_manifest(Path(args.shortcut_summary))
    if method_freeze.get("manifest_sha256") != "4eb17884239778c748d74515be39a80565f57a9806dbda57a1f454ad45b02d6e":
        parser.error("unexpected method freeze")
    if shortcut_summary.get("engineering_abcd_authorized") is not True:
        parser.error("shortcut audit has not authorized engineering A/B/C/D")
    cases = {case.case_id: case for case in load_cases_v2(args.cases)}
    selected_ids = tuple(args.case_id or DEFAULT_CASE_IDS)
    if len(selected_ids) != 5 or len(set(selected_ids)) != 5:
        parser.error("engineering pilot requires exactly five unique cases")
    missing = sorted(set(selected_ids) - set(cases))
    if missing:
        parser.error(f"unknown selected cases: {missing}")
    selected = [cases[case_id] for case_id in selected_ids]
    run_arms = tuple(args.arm or ARM_WIDTHS)
    selection_path = output_dir / "selection_freeze.json"
    if not selection_path.exists():
        _atomic(selection_path, {
            "schema_version": SCHEMA_VERSION, "frozen_at": _now(),
            "case_ids": list(selected_ids), "model": args.model,
            "arms": ARM_WIDTHS, "max_expansions": args.max_expansions,
            "max_actions_per_state_by_arm": ARM_LOCAL_EXPANSIONS,
            "dense_action_limit": 30,
            "seed": args.seed, "source_hashes": source_hashes,
            "logprob_batch_size": args.logprob_batch_size,
            "logprob_batching_note": "execution-only microbatch; score formula and action order unchanged",
            "inputs": {
                "cases": {"path": str(Path(args.cases).resolve()), "sha256": _sha(Path(args.cases))},
                "method_freeze": {"path": str(Path(args.method_freeze).resolve()), "sha256": _sha(Path(args.method_freeze))},
                "shortcut_summary": {"path": str(Path(args.shortcut_summary).resolve()), "sha256": _sha(Path(args.shortcut_summary))},
            },
            "formal_conclusion_allowed": False,
        })
    backend = None
    logits = None
    controller = None
    semantic_judge = None
    if args.phase in {"all", "search"}:
        backend = WikipediaPageBackend(cache_path=args.cache_path, min_request_interval=0.1)
        logits = HuggingFaceCausalLMBackendV24(
            args.model, device=args.device, dtype=args.dtype,
        )
        scored_backend = MicrobatchedLogprobBackendV25(
            logits, args.logprob_batch_size,
        )
        controller = HierarchicalOpenWeightLiveControllerV24(
            OpenWeightConditionalActionScorerV24(scored_backend),
            compact_payload_proposer=EvidenceConditionedAnswerGeneratorV24(scored_backend),
            payload_proposer_name="open_weight_evidence_conditioned_answer_generator_v2.4",
        )
    if args.phase in {"all", "score"}:
        semantic_judge = LLMSemanticClaimJudgeV2(
            args.semantic_judge_model, version="abcd-engineering-v25",
            cache_path=output_dir / "semantic_judge.db",
        )
    summary_rows = []
    try:
        for case in selected:
            public = case.public_view()
            for arm in run_arms:
                width = ARM_WIDTHS[arm]
                local_expansion_width = ARM_LOCAL_EXPANSIONS[arm]
                artifact = output_dir / f"{case.case_id}.{arm}.json"
                raw_artifact = output_dir / f"{case.case_id}.{arm}.raw.json"
                if artifact.exists():
                    saved = _load_manifest(artifact)
                    summary_rows.append(saved["summary"])
                    continue
                if not raw_artifact.exists() and args.phase in {"all", "search"}:
                    assert backend is not None and logits is not None and controller is not None
                    if arm == "A":
                        tool_caller = OpenWeightToolCallerV25(logits)
                        result = run_temporal_browsing(
                            model=args.model, backend=backend,
                            start_title=public.start_page, question=public.question,
                            allowed_as_of=[public.cutoff_date, public.target_date],
                            max_steps=args.max_expansions,
                            target_title="__HIDDEN_TARGET__",
                            target_as_of=public.target_date,
                            snapshot_date_range=(public.cutoff_date, public.target_date),
                            reveal_target_title=False, cutoff_reference=public.cutoff_date,
                            semantic_route_contract={"open_world": True}, temperature=0.0,
                            call_model_fn=tool_caller, verbose=False,
                        )
                        raw = {"external_tool_result": result, "local_tool_calls": tool_caller.calls}
                    else:
                        assert width is not None
                        assert local_expansion_width is not None
                        live = run_live_temporal_search_v24(
                            public_case=public, backend=backend,
                            environment=TemporalWikipediaEnvironmentV2(backend),
                            controller=controller,
                            config=LiveSearchConfigV23(
                                beam_width=width, max_expansions=args.max_expansions,
                                max_actions_per_state=local_expansion_width,
                                dense_action_limit=30,
                                seed=args.seed,
                            ),
                        )
                        raw = {"live_search": live.to_dict()}
                    _atomic(raw_artifact, {
                        "schema_version": SCHEMA_VERSION, "created_at": _now(),
                        "case_id": case.case_id, "arm": arm, "beam_width": width,
                        "local_expansion_width": local_expansion_width,
                        **raw,
                    })
                    print(json.dumps({
                        "case_id": case.case_id, "arm": arm,
                        "raw_search_complete": True,
                    }, sort_keys=True), flush=True)
                if args.phase == "search":
                    continue
                if not raw_artifact.exists():
                    raise FileNotFoundError(f"missing raw search artifact: {raw_artifact}")
                assert semantic_judge is not None
                raw = _load_manifest(raw_artifact)
                if arm == "A":
                    metrics = _score_external(
                        cast(dict[str, Any], raw["external_tool_result"]),
                        case, semantic_judge,
                    )
                else:
                    metrics = _score_live(
                        _hydrate_live_result(cast(dict[str, Any], raw["live_search"])),
                        case, semantic_judge,
                    )
                summary = {
                    "case_id": case.case_id, "arm": arm, "beam_width": width,
                    "local_expansion_width": local_expansion_width,
                    **{key: metrics[key] for key in (
                        "bridge_found", "bridges_acquired", "bridge_count",
                        "tail_found", "structured_submission_present",
                        "complete_evidence_submitted", "final_answer",
                        "final_answer_correct", "expansions", "stop_reason", "failure",
                    )},
                }
                _atomic(artifact, {
                    "schema_version": SCHEMA_VERSION, "created_at": _now(),
                    "case": public.to_dict(), "arm": arm, "beam_width": width,
                    "local_expansion_width": local_expansion_width,
                    "metrics": metrics, "summary": summary,
                    "raw_artifact": raw_artifact.name,
                    "formal_conclusion_allowed": False,
                })
                summary_rows.append(summary)
                print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    finally:
        if semantic_judge is not None:
            semantic_judge.close()
        if backend is not None:
            backend.close()
    if args.phase == "search":
        return 0
    aggregate = {
        "schema_version": SCHEMA_VERSION, "created_at": _now(),
        "case_count": len(selected), "run_count": len(summary_rows),
        "by_arm": {
            arm: {
                "runs": sum(row["arm"] == arm for row in summary_rows),
                "bridge_found": sum(row["arm"] == arm and row["bridge_found"] for row in summary_rows),
                "tail_found": sum(row["arm"] == arm and row["tail_found"] for row in summary_rows),
                "complete_evidence_submitted": sum(row["arm"] == arm and row["complete_evidence_submitted"] for row in summary_rows),
                "final_answer_correct": sum(row["arm"] == arm and row["final_answer_correct"] for row in summary_rows),
                "failure_stages": {
                    stage: sum(row["arm"] == arm and row["failure"]["stage"] == stage for row in summary_rows)
                    for stage in ("candidate", "ranking", "beam", "submission", "none")
                },
            } for arm in ARM_WIDTHS
        },
        "rows": summary_rows,
        "formal_conclusion_allowed": False,
        "interpretation": "engineering end-to-end feasibility only",
    }
    _atomic(output_dir / "summary.json", aggregate)
    print(json.dumps(aggregate["by_arm"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
