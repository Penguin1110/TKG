# Joint rank-and-submit live runner v2.3 engineering report

Date: 2026-08-16

## Outcome

V2.3 replaces the v2.2 `structured proposer → dense ranker` split with one
`JointRankAndSubmitControllerV23` call per unfinished state. The same response
updates the state summary, scores every compacted graph action and the fixed
submit slot, and either abstains or supplies a complete structured submission.

The deterministic engineering suite passes. The post-freeze API smoke did not
complete: GPT-5.4-mini assigned equal utility to the compacted hyperlinks,
followed a distractor, correctly abstained, and eventually reached a state whose
only candidate was a non-executable abstained submit slot. The frozen v2.3 runner
then raised `v2.3 live search produced no retained state`; the API harness saved
this as a failed smoke instead of reporting success.

This is not benchmark accuracy. Joint API controller reduces split-brain
inconsistency, but remains an external controller over program-enumerated graph
actions.

## Phase 0 audit

The pre-implementation audit is
`docs/JOINT_RANK_SUBMIT_V2_3_PHASE0_AUDIT.md`. It records both v2.2 prompts,
response/cache contracts, the old parameterized-submit lifecycle, compaction
ordering, failure behavior, and the minimal append-only change.

## Joint prompt and response

The controller receives only the public question/dates, current
page/revision/snapshot, visible trajectory evidence, state summary/entities,
public candidate actions, budget, and seed. A recursive guard rejects private,
reference, witness, evaluator, correctness, expected-page/revision, gold, QID,
and alias-shaped keys plus Q/P identifiers.

The response contains:

```text
reasoning_summary
extracted_entities
evidence_notes
action_utilities: [{action_id, utility}, ...]
submission: StructuredSubmissionV2 | null
abstain_reason: string | null
```

The action-utility list form permits duplicate-ID detection. Exactly the
compacted IDs must be returned, including `submit_slot:v1`.

## Submit-slot lifecycle

```text
retrieved transitions + pagination
→ reserve pagination and submit_slot:v1
→ document-order compaction to <= 30
→ one joint controller response
→ validate dense ranking independently
→ validate optional submission independently
```

| Joint submission | Slot behavior | Graph scores |
| --- | --- | --- |
| `null` | `JOINT_SUBMISSION_ABSTAINED`; not expanded | retained |
| malformed after retry | `JOINT_SUBMISSION_PAYLOAD_INVALID`; not expanded | retained when ranking is valid |
| nonexistent evidence/literal/claim/date failure | rejected with localized status; not expanded | retained |
| public-gate valid | instantiate payload-hashed `SUBMIT_ANSWER` | competes globally using slot score |

An abstained or invalid slot does not consume `max_actions_per_state`. A valid
submit trace records the slot ID, instantiated action ID, structured payload,
canonical payload hash, public validation, action score, and cumulative score.

## Ranking and submission retry states

| Attempt 1 | Attempt 2 | Result |
| --- | --- | --- |
| ranking valid, submission valid | not called | rank graph and permit submit |
| ranking valid, submission `null` | not called | rank graph and abstain |
| ranking valid, public-gate invalid | not called | rank graph and reject submit |
| ranking valid, submission schema malformed | ranking valid, schema valid/null | use second response |
| ranking valid, submission schema malformed | ranking valid, schema malformed | retain second graph ranking, reject submit |
| ranking invalid/unparseable | ranking valid | use second response |
| ranking invalid/unparseable | ranking invalid/unparseable | branch fails closed |

There is no omitted-action floor and no correctness signal in either contract.

## Deterministic synthetic results

The new fixture is independent of the old v2.2 API failure:

- public start: `Joint Hub@101`;
- private witness route: `Reference Route`;
- solver route: `Alternative Route@111 → Scientist Z@120`;
- final answer: `Harbor City`.

Corrected append-only artifact:
`examples/temporal_eval_v23/deterministic_joint_multiroute_v23_1.json`.

Result:

```text
end_to_end_success = true
reference_route_recalled = false
alternative_valid_route_found = true
standalone_submission_proposer_used = false
```

The first pre-freeze artifact,
`examples/temporal_eval_v23/deterministic_joint_multiroute_v23.json`, is retained
unchanged. It failed because the synthetic required claim was encoded in the
reverse subject/object direction. The fixture was corrected in a new artifact;
no raw output was overwritten.

Separate tests cover complete evidence, incomplete evidence, nonexistent
evidence ID, multi-path acceptance, and finished/unfinished beam competition.

## Tests

- New v2.3 tests: 20 passed.
- Full repository: 220 passed.
- mypy: 54 source files, no issues at freeze time.
- pyflakes: no findings at freeze time.
- Missing, duplicate, unexpected, and non-finite action utilities each retry
  once and then fail closed.
- Same input/seed is reproducible; expansion stops exactly at the budget; trace
  scores recompute cumulative score.
- Legacy v2.2 freeze and API-failure hashes remain unchanged.

## Fresh 40-state submission gate

Not executed. There are zero independent post-freeze labeled states available,
not the required 20 positive and 20 negative states. Neither the old v2.2
failure nor the v2.3 synthetic fixture was duplicated to fill the gate.

Machine-readable status:
`examples/temporal_eval_v23/fresh_submission_gate_status_v23.json`.

Frozen thresholds remain:

```text
positive valid-submission recall >= 0.90
negative false-submit rate <= 0.05
dense ranking contract pass rate = 1.00
evidence ownership pass rate = 1.00
gold leakage = 0
```

## Post-freeze API smoke

Artifact: `examples/temporal_eval_v23/joint_api_gpt54mini_smoke_v23.json`.
Controller cache: `examples/temporal_eval_v23/joint_api_gpt54mini_cache_v23.db`.

Five complete raw joint responses are preserved. The model abstained correctly
while evidence was incomplete. At the 30-action hyperlink state it gave all
listed hyperlinks equal utility, so deterministic tie-breaking selected a
distractor. After that page had no graph progress, the final response again
abstained with only `submit_slot:v1` available.

```text
RUNNER COMPLETED = false
STRUCTURED SUBMISSION = false
END-TO-END SYNTHETIC SUCCESS = false
FAILURE = NO_RETAINED_STATE_RUNTIME_ERROR
```

Because the frozen runner raised before returning its in-memory trajectory, the
five cached responses are preserved but a complete beam trajectory is not
available. This is a known v2.3 terminal-state handling defect: an abstained slot
with no executable graph action should become an auditable exhausted/no-answer
terminal state, not a `RuntimeError`. The frozen runner/prompt was not modified
after observing the smoke. A correction requires a new append-only patch
contract; it cannot be silently folded into v2.3.

## Freeze and files

The authoritative contract is
`docs/OPEN_WORLD_LIVE_RUNNER_V2_3_FREEZE_2026-08-16.json`. It records source,
prompt, schema, compaction, public-gate, evaluator, fixture, and legacy hashes;
model configuration; action/beam budgets; retry behavior; and the preregistered
fresh-state thresholds.

New implementation files:

- `src/tkg/experiment/joint_controller_v23.py`;
- `src/tkg/experiment/temporal_live_runner_v23.py`;
- `src/tkg/experiment/temporal_live_v23_synthetic.py`;
- `src/tkg/experiment/temporal_live_v23_api_smoke.py`;
- `tests/test_joint_controller_v23.py`;
- `tests/test_temporal_live_runner_v23.py`.

## Remaining research work

1. Add an append-only terminal-state correction; do not alter the frozen v2.3
   files or reinterpret the failed API smoke.
2. Generate 40 genuinely fresh submission-decision states and run the frozen
   gate once.
3. Generate fresh cases, run PK admission, and verify live candidate recall.
4. Only then run API A/B/C/D on shared cases and budgets.
5. Implement open-weight conditional action log-probabilities separately.

V2.3 is not graph-integrated attention, model-internal graph traversal,
decoding-logit constrained beam, benchmark accuracy, or evidence of beam-method
superiority.
