"""Append-only five-case Evidence-Pheromone Ant Search v2.6 kill test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tkg.experiment.compact_submission_v24 import (
    compact_submission_from_dict_v24, evaluate_compact_submission_posthoc_v24,
)
from tkg.experiment.evidence_pheromone_ant_search_v26 import (
    METHODS_V26, assert_public_only_v26, frozen_method_config_v26,
    run_temporal_evidence_ant_search_v26,
)
from tkg.experiment.open_weight_abcd_engineering_v25 import (
    MicrobatchedLogprobBackendV25, _actions_valid_live, _action_matches,
    _oracle_evidence_probe, _with_evidence_ids,
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
    EvaluationCaseV2, PublicTemporalCaseV2, load_cases_v2, normalized,
)
from tkg.experiment.temporal_semantic_judge_v2 import LLMSemanticClaimJudgeV2
from tkg.wikipedia.backend import WikipediaPageBackend


EXPERIMENT_SCHEMA_V26 = "evidence-pheromone-ant-kill-test-v2.6"
RAW_SCHEMA_V26 = "evidence-pheromone-ant-raw-v2.6"
SCORED_SCHEMA_V26 = "evidence-pheromone-ant-posthoc-score-v2.6"
DEFAULT_SEEDS_V26 = (11, 23, 37, 53, 71)
DEFAULT_CASE_IDS_V26 = (
    "promoted_c4e2822024602c2c4345",
    "promoted_00f4be76d3b2d0377623",
    "promoted_237ae89c2f939b87444c_p19",
    "promoted_4645b1b1444731a0ddc8",
    "promoted_590bebda8b580ddf5dd5",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _atomic_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"append-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def load_public_cases_v26(path: Path) -> list[PublicTemporalCaseV2]:
    value = _load_json(path)
    rows = value.get("cases")
    if not isinstance(rows, list):
        raise ValueError("public manifest cases must be a list")
    assert_public_only_v26(value, path="public_manifest")
    cases = [PublicTemporalCaseV2(**row) for row in rows]
    if tuple(row.case_id for row in cases) != DEFAULT_CASE_IDS_V26:
        raise ValueError("v2.6 public manifest is not the frozen five-case order")
    return cases


def verify_freeze_v26(
    path: Path, repo_root: Path, *, inference_isolation: bool = False,
) -> dict[str, Any]:
    freeze = _load_json(path)
    if freeze.get("schema_version") != "evidence-pheromone-ant-freeze-v2.6":
        raise ValueError("unexpected ant-search freeze schema")
    if freeze.get("status") != "frozen_before_first_model_execution":
        raise ValueError("ant-search manifest was not frozen before execution")
    expected_manifest_sha = freeze.get("manifest_sha256")
    without_hash = {key: value for key, value in freeze.items() if key != "manifest_sha256"}
    if expected_manifest_sha != _canonical_sha(without_hash):
        raise ValueError("ant-search freeze manifest hash mismatch")
    for name, expected in (freeze.get("source_sha256") or {}).items():
        if _sha_path(repo_root / name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    for name, record in (freeze.get("input_files") or {}).items():
        input_path = repo_root / str(record["path"])
        if inference_isolation and name in set(
            freeze.get("inference_excluded_inputs") or []
        ):
            if input_path.exists():
                raise ValueError(
                    f"private/posthoc input exists in inference sandbox: {name}"
                )
            continue
        if _sha_path(input_path) != record["sha256"]:
            raise ValueError(f"frozen input changed: {name}")
    if tuple(freeze.get("seeds") or ()) != DEFAULT_SEEDS_V26:
        raise ValueError("unexpected frozen seeds")
    if tuple(freeze.get("methods") or ()) != METHODS_V26:
        raise ValueError("unexpected frozen methods")
    return freeze


class InstrumentedBackendV26:
    """Count actual model calls and tokenized sequences without changing outputs."""

    def __init__(self, backend: MicrobatchedLogprobBackendV25):
        self.backend = backend
        self.backend_name = backend.backend_name + "_instrumented_v2.6"
        self.conditional_calls = 0
        self.conditional_sequences = 0
        self.conditional_prompt_tokens = 0
        self.conditional_continuation_tokens = 0
        self.generation_calls = 0
        self.generation_prompt_tokens = 0
        self.generated_tokens = 0

    @property
    def tokenizer(self) -> Any:
        return self.backend.backend.tokenizer

    def snapshot(self) -> dict[str, int]:
        return {
            "conditional_calls": self.conditional_calls,
            "conditional_sequences": self.conditional_sequences,
            "conditional_prompt_tokens": self.conditional_prompt_tokens,
            "conditional_continuation_tokens": self.conditional_continuation_tokens,
            "generation_calls": self.generation_calls,
            "generation_prompt_tokens": self.generation_prompt_tokens,
            "generated_tokens": self.generated_tokens,
        }

    @staticmethod
    def delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
        return {key: int(after[key]) - int(before[key]) for key in after}

    def _render_generation(self, prompt: str, system_prompt: str | None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return str(self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                ))
            except TypeError:
                return str(self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                ))
        return "\n".join(
            f"{row['role'].upper()}: {row['content']}" for row in messages
        ) + "\nASSISTANT:"

    def generate_text(
        self, prompt: str, *, max_new_tokens: int = 192,
        system_prompt: str | None = None,
    ) -> str:
        rendered = self._render_generation(prompt, system_prompt)
        prompt_count = len(self.tokenizer(
            rendered, add_special_tokens=True,
        )["input_ids"])
        result = self.backend.generate_text(
            prompt, max_new_tokens=max_new_tokens, system_prompt=system_prompt,
        )
        generated_count = len(self.tokenizer(
            result, add_special_tokens=False,
        )["input_ids"])
        self.generation_calls += 1
        self.generation_prompt_tokens += prompt_count
        self.generated_tokens += generated_count
        return result

    def conditional_token_logprobs(
        self, prompt: str, continuation: str,
    ) -> list[float]:
        values = self.backend.conditional_token_logprobs(prompt, continuation)
        prompt_count = len(self.tokenizer(
            prompt, add_special_tokens=True,
        )["input_ids"])
        self.conditional_calls += 1
        self.conditional_sequences += 1
        self.conditional_prompt_tokens += prompt_count
        self.conditional_continuation_tokens += len(values)
        return values

    def conditional_token_logprobs_batch(
        self, prompt: str, continuations: list[str],
    ) -> list[list[float]]:
        values = self.backend.conditional_token_logprobs_batch(prompt, continuations)
        prompt_count = len(self.tokenizer(
            prompt, add_special_tokens=True,
        )["input_ids"])
        self.conditional_calls += 1
        self.conditional_sequences += len(continuations)
        self.conditional_prompt_tokens += prompt_count * len(continuations)
        self.conditional_continuation_tokens += sum(len(row) for row in values)
        return values


def _search_one(
    *, case: PublicTemporalCaseV2, method: str, seed: int,
    max_expansions: int, backend: WikipediaPageBackend,
    instrumented: InstrumentedBackendV26, controller: Any,
    freeze: dict[str, Any], freeze_path: Path,
) -> dict[str, Any]:
    config = frozen_method_config_v26(method, seed)
    if max_expansions != config.max_expansions:
        config = replace(config, max_expansions=max_expansions)
    environment = TemporalWikipediaEnvironmentV2(backend)
    before = instrumented.snapshot()
    started_at = _now()
    started = time.perf_counter()
    result = run_temporal_evidence_ant_search_v26(
        public_case=case, backend=backend, environment=environment,
        controller=controller, config=config,
    )
    wall = time.perf_counter() - started
    counters = InstrumentedBackendV26.delta(before, instrumented.snapshot())
    return {
        "schema_version": RAW_SCHEMA_V26,
        "created_at": _now(), "started_at": started_at,
        "case_id": case.case_id, "method": method, "seed": seed,
        "freeze_manifest": {
            "path": str(freeze_path), "sha256": _sha_path(freeze_path),
            "manifest_sha256": freeze["manifest_sha256"],
        },
        "wall_time_seconds": wall,
        "model_accounting": counters,
        "search": result.to_dict(),
        "inference_private_inputs_used": False,
        "formal_conclusion_allowed": False,
    }


def _trace_matches_expected(trace: Sequence[Mapping[str, Any]], expected: Any) -> list[int]:
    result = []
    for index, row in enumerate(trace):
        action = row.get("action") or {}
        if _action_matches(action, expected):
            result.append(index)
    return result


def _score_ant(
    *, ant: Mapping[str, Any], case: EvaluationCaseV2,
    judge: LLMSemanticClaimJudgeV2,
) -> dict[str, Any]:
    state = ant["state"]
    evidence = _with_evidence_ids(list(state.get("collected_evidence") or []))
    trace = list(state.get("action_trace") or [])
    actions_valid = _actions_valid_live(trace)
    acquisition = _oracle_evidence_probe(
        case, evidence, actions_valid=actions_valid, judge=judge,
    )
    submitted = state.get("submitted")
    submission_evaluation = None
    if isinstance(submitted, dict):
        try:
            compact = compact_submission_from_dict_v24(submitted)
            submission_evaluation = evaluate_compact_submission_posthoc_v24(
                case=case, submission=compact, trajectory_evidence=evidence,
                trajectory_actions_valid=actions_valid, semantic_judge=judge,
            )
        except ValueError as exc:
            submission_evaluation = {"end_to_end_success": False, "error": str(exc)}
    answer = str((submitted or {}).get("answer") or "") if isinstance(submitted, dict) else ""
    return {
        "ant_id": ant["ant_id"],
        "bridge_found": bool(acquisition["critical_bridge_evidence_complete"]),
        "bridges_acquired": acquisition["critical_bridges_acquired"],
        "bridge_count": acquisition["critical_bridge_count"],
        "tail_found": bool(acquisition["tail_claim_result"]["passed"]),
        "structured_submission_present": isinstance(submitted, dict),
        "complete_evidence_submitted": bool(
            submission_evaluation and submission_evaluation.get("end_to_end_success")
        ),
        "final_answer": answer,
        "final_answer_correct": normalized(answer) in {
            normalized(value) for value in case.accepted_final_answer_aliases
        },
        "actions_valid": actions_valid,
        "acquisition_evaluation": acquisition,
        "submission_evaluation": submission_evaluation,
    }


def _posthoc_diagnose(
    raw: Mapping[str, Any], case: EvaluationCaseV2,
    judge: LLMSemanticClaimJudgeV2,
) -> dict[str, Any]:
    search = raw["search"]
    ant_scores = [_score_ant(ant=ant, case=case, judge=judge) for ant in search["ants"]]
    expected = case.reference_routes[0].actions[0] if case.reference_routes else None
    executed_by_ants = []
    continued_by_ants = []
    if expected is not None:
        for ant in search["ants"]:
            trace = list(ant["state"].get("action_trace") or [])
            matches = _trace_matches_expected(trace, expected)
            if matches:
                executed_by_ants.append(int(ant["ant_id"]))
                if any(index + 1 < len(trace) for index in matches):
                    continued_by_ants.append(int(ant["ant_id"]))
    opportunity_count = 0
    selected_count = 0
    for step in search["audit_steps"]:
        for candidate in step.get("candidate_actions") or []:
            if expected is not None and _action_matches(candidate, expected):
                opportunity_count += 1
                selected_count += int(candidate.get("selected") is True)
    all_nodes = {
        tuple(node)
        for ant in search["ants"]
        for node in ant["state"].get("visited_nodes") or []
    }
    submitted = [row for row in ant_scores if row["structured_submission_present"]]
    metrics = {
        "correct_progress_action": expected.to_dict() if expected is not None else None,
        "correct_progress_candidate_opportunities": opportunity_count,
        "correct_progress_selected_count": selected_count,
        "correct_progress_action_executed": bool(executed_by_ants),
        "correct_progress_executed_by_ants": executed_by_ants,
        "correct_child_continued": bool(continued_by_ants),
        "correct_child_continued_by_ants": continued_by_ants,
        "correct_child_reexploration_count": len(executed_by_ants),
        "bridge_found": any(row["bridge_found"] for row in ant_scores),
        "tail_found": any(row["tail_found"] for row in ant_scores),
        "complete_evidence_submitted": any(
            row["complete_evidence_submitted"] for row in ant_scores
        ),
        "correct_answer": any(row["final_answer_correct"] for row in ant_scores),
        "submitted_answer_count": len(submitted),
        "unique_nodes": len(all_nodes),
        "repeated_or_cyclic_transitions": search["repeated_or_cyclic_transitions"],
        "expansions": search["expansions"],
        "executed_transitions": search["executed_transitions"],
        "parent_states_scored": search["parent_states_scored"],
        "scored_actions": search["scored_actions"],
        "wall_time_seconds": raw["wall_time_seconds"],
        "model_accounting": raw["model_accounting"],
        "ant_scores": ant_scores,
    }
    if metrics["complete_evidence_submitted"]:
        failure = {"stage": "none", "reason": "complete_evidence_submission"}
    elif metrics["bridge_found"] and metrics["tail_found"]:
        failure = {"stage": "submission", "reason": "complete_evidence_not_submitted"}
    elif metrics["bridge_found"]:
        failure = {"stage": "tail", "reason": "bridge_found_tail_missing"}
    elif metrics["correct_child_continued"]:
        failure = {"stage": "downstream", "reason": "correct_child_continued_without_bridge"}
    elif metrics["correct_progress_action_executed"]:
        failure = {"stage": "continuation", "reason": "correct_child_not_continued"}
    elif opportunity_count:
        failure = {"stage": "sampling", "reason": "correct_candidate_not_sampled"}
    else:
        failure = {"stage": "candidate", "reason": "correct_candidate_never_exposed"}
    metrics["failure"] = failure
    return metrics


def _aggregate(scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in METHODS_V26:
        rows = [row for row in scored if row["method"] == method]
        result[method] = {
            "runs": len(rows),
            "correct_progress_action_executed": sum(
                bool(row["metrics"]["correct_progress_action_executed"]) for row in rows
            ),
            "correct_child_continued": sum(
                bool(row["metrics"]["correct_child_continued"]) for row in rows
            ),
            "bridge_found": sum(bool(row["metrics"]["bridge_found"]) for row in rows),
            "tail_found": sum(bool(row["metrics"]["tail_found"]) for row in rows),
            "complete_evidence_submitted": sum(
                bool(row["metrics"]["complete_evidence_submitted"]) for row in rows
            ),
            "correct_answer": sum(bool(row["metrics"]["correct_answer"]) for row in rows),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("search", "score"), required=True)
    parser.add_argument("--public-cases", required=True)
    parser.add_argument("--private-cases")
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--logprob-batch-size", type=int, default=4)
    parser.add_argument("--max-expansions", type=int, default=40)
    parser.add_argument("--method", action="append", choices=METHODS_V26, default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--semantic-judge-model", default="openai/gpt-4.1")
    args = parser.parse_args()

    repo_root = Path.cwd()
    freeze_path = Path(args.freeze_manifest)
    freeze = verify_freeze_v26(
        freeze_path, repo_root, inference_isolation=args.phase == "search",
    )
    public_cases = load_public_cases_v26(Path(args.public_cases))
    selected_ids = tuple(args.case_id or DEFAULT_CASE_IDS_V26)
    if not set(selected_ids) <= set(DEFAULT_CASE_IDS_V26):
        parser.error("only the frozen five development case IDs are allowed")
    public_by_id = {case.case_id: case for case in public_cases}
    selected_public = [public_by_id[case_id] for case_id in selected_ids]
    methods = tuple(args.method or METHODS_V26)
    seeds = tuple(args.seed or DEFAULT_SEEDS_V26)
    if not set(seeds) <= set(DEFAULT_SEEDS_V26):
        parser.error("seed is outside the frozen seed set")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "search":
        backend = WikipediaPageBackend(
            cache_path=args.cache_path, min_request_interval=0.1,
        )
        logits = HuggingFaceCausalLMBackendV24(
            args.model, device=args.device, dtype=args.dtype,
        )
        microbatched = MicrobatchedLogprobBackendV25(
            logits, args.logprob_batch_size,
        )
        instrumented = InstrumentedBackendV26(microbatched)
        controller = HierarchicalOpenWeightLiveControllerV24(
            OpenWeightConditionalActionScorerV24(instrumented),
            compact_payload_proposer=EvidenceConditionedAnswerGeneratorV24(instrumented),
            payload_proposer_name=(
                "open_weight_evidence_conditioned_answer_generator_v2.4"
            ),
        )
        for method in methods:
            for seed in seeds:
                for case in selected_public:
                    path = output_dir / f"{case.case_id}.{method}.seed{seed}.raw.json"
                    if path.exists():
                        existing = _load_json(path)
                        if (
                            existing.get("schema_version") == RAW_SCHEMA_V26
                            and existing.get("case_id") == case.case_id
                            and existing.get("method") == method
                            and existing.get("seed") == seed
                        ):
                            print(json.dumps({"status": "resume_skip_complete", "path": str(path)}))
                            continue
                        raise FileExistsError(f"invalid existing raw artifact: {path}")
                    try:
                        artifact = _search_one(
                            case=case, method=method, seed=seed,
                            max_expansions=args.max_expansions, backend=backend,
                            instrumented=instrumented, controller=controller,
                            freeze=freeze, freeze_path=freeze_path,
                        )
                        _atomic_new(path, artifact)
                        print(json.dumps({
                            "status": "complete", "case_id": case.case_id,
                            "method": method, "seed": seed,
                            "expansions": artifact["search"]["expansions"],
                            "path": str(path),
                        }, sort_keys=True))
                    except Exception as exc:
                        error_path = output_dir / (
                            f"{case.case_id}.{method}.seed{seed}.attempt1.error.json"
                        )
                        if not error_path.exists():
                            _atomic_new(error_path, {
                                "schema_version": "evidence-pheromone-ant-error-v2.6",
                                "created_at": _now(), "case_id": case.case_id,
                                "method": method, "seed": seed,
                                "error_type": type(exc).__name__, "error": str(exc),
                                "retry_policy": "at_most_one_infrastructure_only_resume",
                                "counts_as_reject": False,
                            })
                        raise
        return 0

    if not args.private_cases:
        parser.error("--private-cases is required for score phase")
    private = {case.case_id: case for case in load_cases_v2(args.private_cases)}
    if set(selected_ids) - set(private):
        parser.error("private score file lacks a selected case")
    judge = LLMSemanticClaimJudgeV2(
        args.semantic_judge_model, version="evidence-pheromone-ant-v26-posthoc",
        cache_path=output_dir / "semantic_judge.db",
    )
    scored_rows = []
    try:
        for method in methods:
            for seed in seeds:
                for case_id in selected_ids:
                    raw_path = output_dir / f"{case_id}.{method}.seed{seed}.raw.json"
                    if not raw_path.exists():
                        continue
                    scored_path = output_dir / f"{case_id}.{method}.seed{seed}.scored.json"
                    if scored_path.exists():
                        scored_rows.append(_load_json(scored_path))
                        continue
                    raw = _load_json(raw_path)
                    metrics = _posthoc_diagnose(raw, private[case_id], judge)
                    artifact = {
                        "schema_version": SCORED_SCHEMA_V26, "created_at": _now(),
                        "case_id": case_id, "method": method, "seed": seed,
                        "raw_path": str(raw_path), "raw_sha256": _sha_path(raw_path),
                        "metrics": metrics, "private_evaluation_used_posthoc_only": True,
                        "formal_conclusion_allowed": False,
                    }
                    _atomic_new(scored_path, artifact)
                    scored_rows.append(artifact)
                    print(json.dumps({
                        "case_id": case_id, "method": method, "seed": seed,
                        "bridge_found": metrics["bridge_found"],
                        "correct_child_continued": metrics["correct_child_continued"],
                    }, sort_keys=True))
    finally:
        judge.close()
    summary_path = output_dir / "summary.json"
    summary = {
        "schema_version": EXPERIMENT_SCHEMA_V26, "created_at": _now(),
        "run_count": len(scored_rows), "by_method": _aggregate(scored_rows),
        "rows": [{
            "case_id": row["case_id"], "method": row["method"],
            "seed": row["seed"], **{
                key: row["metrics"][key] for key in (
                    "correct_progress_action_executed", "correct_child_continued",
                    "bridge_found", "tail_found", "complete_evidence_submitted",
                    "correct_answer", "unique_nodes", "expansions",
                    "scored_actions", "wall_time_seconds", "failure",
                )
            },
        } for row in scored_rows],
        "decision_thresholds": freeze["decision_thresholds"],
        "engineering_decision": "pending_deterministic_rule_application",
        "formal_conclusion_allowed": False,
    }
    if summary_path.exists():
        raise FileExistsError("append-only summary already exists")
    _atomic_new(summary_path, summary)
    print(json.dumps(summary["by_method"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
