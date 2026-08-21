# Joint rank-and-submit v2.3：Phase 0 repository audit

Date: 2026-08-16

## v2.2 data flow

The live v2.2 state loop performs two independent model calls:

```text
visible state/evidence
→ StructuredSubmissionProposerV2.propose
→ public submission gate, possibly creating parameterized SUBMIT_ANSWER
→ retrieval-order compaction to at most 30 actions
→ ApiLiveActionRankerV22.rank
→ expansion
```

This ordering explains the saved post-freeze failure: the ranker explicitly
recognized Person X and Answer City, while the separately cached proposer
returned an empty submission twice.  Consequently no submit action entered the
ranker candidate set and the state ended as `exhausted_no_legal_progress`.

## Prompt, response, and cache contracts

### Structured proposer

- Implementation: `src/tkg/experiment/temporal_submission_v2.py`.
- Input: public question/cutoff/target, seed, and visible evidence with evidence
  IDs. It does not receive graph actions.
- Response: one `StructuredSubmissionV2`; incomplete evidence is represented by
  an empty answer/claims rather than JSON `null`.
- Cache key: canonical SHA-256 of model, complete prompt, and
  `structured-submission-proposer-v2` contract string.
- Parse/schema failure currently escapes into the v2.2 runner's generic
  `runner_or_environment_error`; an empty or public-gate-rejected submission is
  simply omitted from the candidate set.

### Dense ranker

- Implementation: `src/tkg/experiment/temporal_live_ranker_v22.py`.
- Input: public question/cutoff/target, current page/revision, state summary,
  extracted entities, visible evidence, seed, and already-compacted candidate
  actions.
- Response: summary/entities/notes plus a list of distinct
  `{action_id, utility}` rows.
- Cache key: canonical SHA-256 of model, complete prompt, attempt index, and
  `open-world-live-dense-action-ranker-v2.2` contract string.
- It requires exact IDs, finite utilities in `[-100, 100]`, retries once, and
  then fails the branch closed as `ranker_contract_failure`.

## Submit lifecycle and runner assumptions

`SUBMIT_ANSWER` is created in v2.2 `_solver_retrieved_actions`, only after the
standalone proposer returned a non-empty payload and the public ownership/literal
gate passed. Compaction therefore occurs after proposal and reserves the already
parameterized submit action alongside pagination controls.

The v2.2 executor assumes submit is already parameterized: it parses
`action.params` as `StructuredSubmissionV2`, repeats the public gate, marks the
child finished, and records the payload-derived action ID in the trace. There is
no fixed answer-free submit slot.

## Minimal append-only v2.3 change

V2.2 modules, manifest, caches, JSON artifacts, and results remain untouched.
V2.3 needs only new versioned components:

1. a `JointRankAndSubmitControllerV23` with one prompt/cache call per unfinished
   state, exact dense ranking validation, and separately classified submission
   validation;
2. a fixed `submit_slot:v1` inserted before compaction and always retained;
3. a v2.3 runner that expands executable graph actions even when the slot
   abstains or contains an invalid payload, and instantiates a parameterized
   submit action only after the public gate;
4. recursive inference-payload no-gold assertions, new synthetic fixtures/tests,
   append-only artifacts, and a new freeze manifest.

The v2 evaluator and Wikipedia environment remain reusable, read-only
dependencies. Post-hoc aliases, witnesses, semantic decisions, and reference
routes stay outside the controller and beam.
