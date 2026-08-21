# Open-world Temporal Wikipedia Graph evaluation v2

Date: 2026-08-16

## Result

The evaluation layer now treats Wikipedia navigation as open-world and
multi-path. A private reference route proves feasibility and supports conditional
diagnostics; it is not a unique gold trajectory and cannot determine primary
success. Primary success is based on a legal trajectory plus a structured,
semantically and temporally validated evidence submission.

The implementation passes the requested synthetic counterexample:

```text
Route A = private reference route
Route A is retrieved but removed before ranking
Route B remains, reaches an alternative witness for the same bridge
structured bridge + tail + event time validate
end_to_end_success = true
reference_route_recalled = false
alternative_valid_route_found = true
```

This is an engineering contract result, not benchmark accuracy. No fresh
post-freeze cases or formal model runs exist.

## Phase 0 audit

The repository audit is in `OPEN_WORLD_EVAL_V2_PHASE0_AUDIT.md`. The main v1
problem was that one function mixed complete hyperlink legality with `max_links`
solver truncation, while revision sampling exposed eight probes as if they were
the action universe. Forced-state and end-to-end also used different builders.
The legacy scorer then compared a trajectory with one exact private chain and
emitted uniqueness-suggesting labels.

V1 raw artifacts and labels remain untouched. V2 is a parallel, append-only
layer with new schemas, paths, commands, and freeze manifest.

## Schema v2

The machine-readable descriptor is `OPEN_WORLD_EVAL_V2_SCHEMA.json`.

### Public inference case

Only these fields may enter the solver:

```text
case_id, model_id, question, start_page, cutoff_date, target_date
```

The public projection contains no aliases, claims, witnesses, reference actions,
reference revisions, QIDs, or reference distance. Recursive no-gold-leak tests
also check the structured-submission prompt.

### Private evaluation case

`EvaluationCaseV2` adds:

- `accepted_final_answer_aliases`;
- one or more `critical_claims`;
- `tail_relation`;
- `validated_evidence_requirements`;
- `event_time_constraints`;
- zero or more diagnostic-only `reference_routes`;
- source schema and content hash.

Each claim has `subject`, `relation`, `object`, `event_time`, and a multi-valued
`witnesses` list. Each witness binds page title, revision ID, semantic excerpt,
excerpt SHA-256, validation status, and semantic-validation record. The legacy
v6 loader creates one witness where only one existed, marks incomplete legacy
judge provenance, and therefore does not make legacy cases formal-v2 eligible.

### Structured submission

`SUBMIT_ANSWER` now carries:

```json
{
  "answer": "...",
  "critical_claims": [{
    "subject": "...",
    "relation": "...",
    "object": "...",
    "event_time": "...",
    "supporting_evidence_ids": ["..."]
  }],
  "tail_claim": {
    "subject": "...",
    "relation": "...",
    "object": "...",
    "supporting_evidence_ids": ["..."]
  }
}
```

The validator checks alias correctness, evidence ownership, literal answer
presence, semantic relations, event times, tail support, and bridge-to-tail
composition separately. Literal support remains explicitly named
`literal_support_gate_passed` and is never reported as semantic support.

Known witness revisions use deterministic title/revision/excerpt-hash matching.
Other legal evidence routes go to `LLMSemanticClaimJudgeV2`, a post-hoc-only
judge that stores complete input, raw and parsed output, model/version,
confidence, cache key, deterministic guards, and
`machine_pass_human_review_required`. Its output never enters search or ranking.

## Environment and solver data flow

```text
Wikipedia environment
  complete rendered-revision links
  complete revision metadata through rvcontinue
  exact revision fetch and legality checks
             |
             | LIST_LINKS / LIST_REVISIONS pages
             v
Solver retrieval policy
  explicit retrieved pages and continuation actions
             |
             v
Gold-free compaction (at most 30)
             |
             v
Dense ranker exact-ID contract
             |
             v
Expanded actions and beam history
             |
             v
Structured post-hoc submission validation
```

Every v2 funnel separates:

```text
environment_legal_actions
solver_retrieved_actions
compacted_ranker_actions
ranker_scores
expanded_actions
```

Large complete legal sets may be externalized only with a complete count,
canonical SHA-256, and immutable artifact reference. Links use explicit offset
pagination; revisions use MediaWiki `rvcontinue` inside the cutoff-target time
window. The same `execute_environment_query_v2` dispatcher is used by forced and
end-to-end v2 callers. The environment never applies the 30-action dense limit.

## Failure taxonomy

The v2 labels are:

- `REFERENCE_LINK_NOT_RECALLED`;
- `REFERENCE_WITNESS_REVISION_NOT_RECALLED`;
- `REFERENCE_ROUTE_NOT_COMPLETED`;
- `REFERENCE_ACTION_NOT_RANKED`;
- `RANKER_CONTRACT_FAILURE`;
- `SEARCH_BUDGET_EXHAUSTED`;
- `NO_VALIDATED_EVIDENCE_CHAIN_FOUND`.

All `REFERENCE_*` labels describe only a known route. A reference action outside
the compacted set has rank `not_evaluable`; it receives neither `-100` nor a
display rank. `NO_VALIDATED_EVIDENCE_CHAIN_FOUND` describes the saved trajectory,
not the whole Wikipedia graph.

## Primary and diagnostic metrics

Primary:

- end-to-end validated answer accuracy;
- critical-bridge acquisition rate;
- semantically supported submission rate;
- temporally valid submission rate.

Reference-link/revision retrieval, route completion, conditional action rank,
conditional beam survival, and infrastructure failures are diagnostics only.
They cannot turn an otherwise valid alternative route into failure.

## Synthetic multi-route result

Artifact: `examples/temporal_eval_v2/synthetic_multiroute_result_v2_1.json`

At the start state, the environment and solver contain 32 actions. The frozen
gold-free compactor retains 30. Route A is environment-legal and retrieved but is
not compacted, so its rank is `not_evaluable`. Route B is compacted, completely
scored under the dense-ID contract, expanded, and reaches the second semantic
witness revision.

Result:

| Field | Value |
| --- | --- |
| critical bridge | 1/1 |
| tail relation | passed |
| event time | passed |
| composition | passed |
| end-to-end success | true |
| reference route recalled | false |
| alternative valid route found | true |

The result remains `machine_pass_human_review_required`; it is a deterministic
synthetic engineering fixture, not a human-reviewed benchmark item.

## Four-coach development replay

Artifacts:

- `examples/temporal_eval_v2/coach_development_replay_v2_1.jsonl`;
- `examples/temporal_eval_v2/coach_development_replay_v2_1.md`.

Sixteen saved trajectories were replayed without changing the source JSONL or
case manifests. None contains a v2 structured submission, none contains both
critical witnesses, and none can count as a validated v2 end-to-end success.

Derived labels:

| Label | Count |
| --- | ---: |
| `NO_VALIDATED_EVIDENCE_CHAIN_FOUND` | 16 |
| `REFERENCE_LINK_NOT_RECALLED` | 16 |
| `REFERENCE_WITNESS_REVISION_NOT_RECALLED` | 16 |
| `REFERENCE_ROUTE_NOT_COMPLETED` | 16 |
| `REFERENCE_ACTION_NOT_RANKED` | 3 |
| `SEARCH_BUDGET_EXHAUSTED` | 8 |

The old trajectories did not record the complete environment revision action
set, so that layer is explicitly `not_reconstructable_from_legacy_trajectory`.
The replay does not turn the absence of the known route into proof that no
alternative Wikipedia route exists.

## Tests

Final local verification:

```text
191 passed
mypy: success on 47 source files
pyflakes: success
git diff --check: success
```

The 18 v2-specific tests cover multi-route acceptance, alternative witness
revisions, reference-route independence, alias-without-bridge rejection,
trajectory evidence ownership, prompt/data leakage, auditable semantic judging,
environment/solver separation, link position 120 pagination, complete revision
pagination, exact same-page switches, conditional ranker metrics, structured
submissions, legacy read-only loading, and legacy freeze hashes. Existing dense
ranker tests continue to cover omitted, duplicate, and unexpected IDs with
fail-closed handling.

## Modified and added files

Core implementation:

- `src/tkg/experiment/temporal_eval_schema_v2.py`;
- `src/tkg/experiment/temporal_environment_v2.py`;
- `src/tkg/experiment/temporal_evaluation_v2.py`;
- `src/tkg/experiment/temporal_submission_v2.py`;
- `src/tkg/experiment/temporal_semantic_judge_v2.py`;
- `src/tkg/experiment/temporal_eval_v2_replay.py`;
- `src/tkg/experiment/temporal_eval_v2_synthetic.py`;
- `src/tkg/wikipedia/backend.py` (new exact-revision and unsampled paginated
  revision APIs only);
- `pyproject.toml` and `uv.lock` (version 0.18.0 and two v2 commands).

Tests and documentation:

- `tests/test_temporal_evaluation_v2.py`;
- `tests/test_wikipedia_pipeline.py`;
- `docs/OPEN_WORLD_EVAL_V2_PHASE0_AUDIT.md`;
- `docs/OPEN_WORLD_EVAL_V2_SCHEMA.json`;
- this report;
- append-only v2 synthetic and coach-replay artifacts.

No v1 beam implementation or frozen v1 raw artifact was modified. The original
v1 manifest hashes still verify.

## Freeze and remaining limits

The authoritative manifest is
`OPEN_WORLD_EVAL_V2_1_FREEZE_2026-08-16.json`. An earlier pre-freeze manifest was
preserved and explicitly superseded rather than overwritten after the
`REFERENCE_ACTION_NOT_RANKED` regression was added.

Still incomplete or deliberately blocked:

1. No fresh post-freeze cases have been generated or run.
2. No formal benchmark accuracy may be reported from the synthetic fixture or
   the four coach development cases.
3. Machine semantic support still requires human review; a waiver is not an
   approval.
4. The v2 environment, funnel, submission proposer, and evaluator are implemented,
   but the frozen v1 beam runner was not rewritten. A future live v2 runner must
   compose these modules without exposing the private case.
5. The API utility ranker remains an external fallback. Open-weight conditional
   log-probability decoding integration is still not implemented.

Acceptance status:

```text
MULTI_PATH_EVALUATION_IMPLEMENTED
REFERENCE_ROUTE_IS_DIAGNOSTIC_ONLY
ENVIRONMENT_SOLVER_SEPARATION_PASSED
STRUCTURED_EVIDENCE_SUBMISSION_PASSED
NO_GOLD_LEAK_TESTS_PASSED
ALTERNATIVE_ROUTE_SYNTHETIC_TEST_PASSED
LEGACY_ARTIFACTS_UNCHANGED
V2_NOT_YET_FORMAL_BENCHMARK_ACCURACY
```
