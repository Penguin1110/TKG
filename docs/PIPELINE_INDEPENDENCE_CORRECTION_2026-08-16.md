# Correction: data admission and solver-method gates are independent

Date: 2026-08-16

This note supersedes only the pipeline-lock interpretation in
`TERMINAL_PATCH_AND_FRESH_GATE_V2_3_1_REPORT.md`. It does not modify the frozen
gate, thresholds, raw responses, scores, or NO-GO result.

The failed joint-controller gate locks:

```text
live solver A/B/C/D using the frozen API joint controller
```

It does **not** lock the independent data pipeline:

```text
fresh candidate generation
→ Wikipedia/event-time machine validation
→ factorized prior-knowledge admission
→ admitted-case pool
```

The two workstreams may proceed concurrently:

```text
Data line:   fresh cases → machine validation → PK admission ─┐
                                                               ├→ formal A/B/C/D
Method line: v2.4 submission / open-weight scoring → gates ────┘
```

They meet only before live candidate-recall experiments and A/B/C/D. A case may
be generated and PK-admitted while the solver method remains NO-GO; admission
does not authorize running the failed controller on it.

Corrected current status:

```text
fresh case generation = UNLOCKED
machine validation = UNLOCKED
PK admission = UNLOCKED
live candidate recall with failed controller = LOCKED
API A/B/C/D = LOCKED
```
