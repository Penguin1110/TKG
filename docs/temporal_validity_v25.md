# Temporal Wikipedia validity v2.5

Version 2.5 separates question validity from search-space measurement.

## Admission

A case may reach `machine_pass_human_review_required` only after the existing
v6 route, exact-revision evidence, temporal contrast, event-order, composition,
and hidden-entity wording checks pass, followed by the independent whole-chain
judge. Global shortest-path completion is not an admission requirement.

The resumable runner writes immutable decisions to `validity_v25.jsonl`. Each
row contains the explicit validity contract, the non-blocking search diagnostic,
the source packet, and a deterministic event ID. Resume does not duplicate an
existing event.

Use the non-blocking policy explicitly:

```bash
python -m tkg.experiment.resumable_machine_validation_v24 \
  ... \
  --shortest-policy diagnostic
```

The default remains `required`, preserving the legacy behavior for old runs.

## Search-space diagnostic

The fast diagnostic inspects only exact revisions already fetched for the
reference route. It checks whether the start revision contains a final-answer
alias and whether a route revision hyperlinks a non-adjacent later entity. It
reports `SHORTCUT_FOUND` or `NO_SHORTCUT_FOUND_WITHIN_BOUND`; neither statement
is a claim about the complete Wikipedia graph.

`shortest_path_status` is one of `exact`, `bounded_lower_bound`, `incomplete`,
or `not_computed`. An incomplete status does not change admission.

## Background BFS

`checkpointed_shortest_diagnostic_v25` is an offline, non-admitting reverse BFS.
Its atomic checkpoint stores the FIFO frontier, `(page, revision)` visited
states, depths, parent/actions, revision and backlink cursors, completed and
pending expansions, canonical ordering, errors, and the SQLite cache version
and fingerprint. Re-running the same command resumes the saved frontier.

```bash
python -m tkg.experiment.checkpointed_shortest_diagnostic_v25 \
  --cases admitted_cases.json --case-id CASE \
  --cache-path wikipedia.db --checkpoint bfs/CASE.json \
  --max-expansions 10
```

The BFS artifact is diagnostic only. Generation, PK admission, inference,
compaction, and action scoring must not read it.
