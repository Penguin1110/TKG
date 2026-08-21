# Temporal Graph-Constrained Beam Search prototype

## Scope

This is an inference-only prototype. It does not add attention interventions,
GNNs, graph tokens, training, or oracle-route scoring. The existing external-tool
agent remains arm A and is not replaced.

The immutable graph semantics are:

```text
node = (canonical Wikipedia page title, revision ID)
FOLLOW_LINK = target present in the current rendered revision
SWITCH_SNAPSHOT = another listed revision of the same canonical page
```

The solver receives only a `TemporalSearchRequest`: opaque case ID, public
question, public start page, registered cutoff date, and target date. It cannot
accept a full experiment case. Gold/reference chains, Wikidata IDs, answer
aliases, hidden entities, target page titles, and intermediate dates exist only
in an optional post-hoc scorer.

## State and actions

`TemporalBeamState` serializes the current page/revision, concise reasoning
summary, monotonically accumulated visible entity names, immutable temporal
constraints, exact visible evidence pages, a sampled revision catalog, visited
`(page, revision)` nodes, full action trace, cumulative score, and terminal
answer/error state.

The candidate actions are `FOLLOW_LINK`, `LIST_REVISIONS`,
`SWITCH_SNAPSHOT(revision_id)`, and `SUBMIT_ANSWER`. Revision discovery costs one
expansion. To avoid an unbounded MediaWiki history scan, it resolves a fixed
number of evenly spaced calendar probes in the cutoff-target interval into real
revision IDs. The sampling policy is written into the action record. A switch is
executed only if the resolved revision ID and canonical page still match.

Every parent expansion writes candidate actions, raw ranker utilities, normalized
action scores, selected and retained IDs, pruning reasons, visible evidence, and
resulting serialized states. The global search has strict beam-width and total
expansion caps. Information states are deduplicated by page, revision, acquired
revision catalog, and terminal answer state; the higher-scoring duplicate wins.

## Ranking

`CallableConditionalLogProbRanker` is the integration point for an open-weight
backend that can return length-normalized conditional log probabilities for each
complete candidate action.

`ApiUtilityRanker` is explicitly a fallback, not decoding integration. It asks an
API model for a utility for every already-legal action, converts utilities to
log-softmax action scores, and freezes responses in a content-addressed SQLite
cache. Before ranking, the gold-free compactor limits the dense request to at
most 30 actions. Returned IDs must equal the legal candidate ID set exactly;
missing, unexpected, duplicate, non-numeric, or non-finite scores invalidate the
call. The fallback retries once and then fails closed instead of assigning an
omitted action a synthetic floor. Gold routes, graph distance, and answer
correctness are never ranker features.

## Experimental arms

The runner exposes the preregisterable arms under a shared tested model and
action/expansion cap:

```text
A external-tool agent baseline
B constrained greedy, width 1
C constrained beam, width 3
D constrained beam, width 5
```

Admission-controlled mode requires a statically valid v6 case plus a matching frozen PK-gate
record. Admission passes only when the critical-bridge known rate is within the
frozen threshold and its unjudgeable count is zero. Engineering-smoke mode does
not run or waive PK: it records `formal_eligible=false` and forbids formal
conclusions. This prototype still records `formal_conclusion_allowed=false` in
all modes because independent final-answer judging and hash-bound human-review
admission have not yet been wired into this runner.

## Engineering smoke

The public two-case manifest is derived from the corrected v7 seeds. It contains
no hidden entities or answers. The older `non_sports_v6/cases.json` is not used as
a valid case source because its README invalidates both its event-order claim and
PK run.

```bash
uv run tkg-run-temporal-beam \
  --public-cases examples/temporal_beam_smoke/public_cases.json \
  --model openai/gpt-4.1-mini \
  --arms A,B,C,D \
  --engineering-smoke \
  --max-expansions 10 \
  --max-actions-per-state 2 \
  --max-links 12 \
  --revision-limit 5 \
  --seed 17 \
  --output temporal_beam_results.jsonl
```

These two cases can establish only engineering behavior: legal transitions,
budget enforcement, reproducibility with frozen caches, serialization, and error
handling. They cannot establish benchmark accuracy, model ranking, or method
superiority. A formal pilot still needs completed v6 case generation, independent
whole-chain judgment, hash-bound human review, valid factorized PK admission, and
predeclared post-hoc scoring.
