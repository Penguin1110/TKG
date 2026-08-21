# Forced-state diagnostics and candidate funnel

Date: 2026-08-15. These are engineering diagnostics only; they do not count
toward benchmark accuracy or model comparison.

## Search change

`SUBMIT_ANSWER` is no longer an empty graph action. At every unfinished state
the controller now enumerates graph actions, independently generates one short
answer candidate, requires a literal match in its cited visible evidence, and
only then ranks the parameterized submit action with graph actions.

The support gate receives no gold answer, aliases, QIDs, private route,
distance-to-gold, or correctness signal. Hyperlink truncation also now preserves
rendered-document order; alphabetical truncation had removed the German and
Canadian critical links before the model saw them.

## GPT-4.1-mini forced-state result

Raw-audit artifact: `examples/temporal_beam_new_questions_v1/forced_diagnostics_gpt41mini_v2.jsonl`.
A cached rerun was identical after timestamps and cache-hit flags were removed.
The corrected fail-closed replay is
`examples/temporal_beam_new_questions_v1/forced_diagnostics_strict_v3.jsonl`.

| Case | Revision | Next page | Bridge extraction | Final composition | Submit |
|---|---:|---:|---:|---:|---:|
| Germany | correct | Andreas, correct | Andreas, correct | Hildesheim, literal-support gate passed | submit omitted by ranker |
| Canada | correct | Pierre candidate present | Pierre, correct | Anaida, literal-support gate passed | Pierre omitted by ranker |

In the original full-candidate calls, both cases are
`RANKER_OUTPUT_COMPLETENESS_FAILURE`. Germany's literal-support-gated submit
and Canada's Pierre action were absent from the raw utility map. Their displayed
ranks were artifacts of the old -100 missing-action floor and cannot be read as
model preferences. The forced states establish candidate recall in the
pre-compaction legal action set and successful extraction/composition; action
ranking is unjudgeable and the API ranker contract failed. The literal-support
gate checks only that the answer string occurs in cited evidence; it does not yet
validate that the passage semantically states the requested relation.

## Post-compaction v4 audit

Artifact:
`examples/temporal_beam_new_questions_v1/forced_diagnostics_compaction_v5.jsonl`.
The controller reconstructed visible document order, recorded the complete legal
set, reserved slots for non-link actions, and sent at most 30 actions to the dense
ranker. Private expected actions were matched only after each call completed.

| Case / forced state | Full legal | Compacted | Correct survives @30 | Dense coverage | Rank among 30 |
|---|---:|---:|---:|---:|---:|
| Germany / revision | 281 | 30 | yes | complete | 2 |
| Germany / next page (Andreas) | 281 | 30 | yes | complete | 1 |
| Germany / submit Hildesheim | 134 | 30 | yes | complete | 3 |
| Canada / revision | 373 | 30 | yes | complete | 3 |
| Canada / next page (Pierre) | 428 | 30 | yes | complete | 2 |
| Canada / submit Anaida | 501 | 30 | yes | complete | 1 |

Thus both critical navigation actions and both submit actions survive this
specific gold-free compaction policy. The v4 ranker returned a complete score for
every compacted action. These are forced-state engineering observations, not beam
recall, formal-case success, or evidence that the cases pass PK admission.

The artifact reports `literal_support_gate_passed` separately from
`semantic_relation_support`, which remains `not_evaluated`. A verbatim answer
match is not proof that the passage states the requested birthplace, spouse, or
other semantic relation.

The missing-action floor has been removed. Dense API scoring now requires exact
ID equality, unique JSON keys, finite in-range utilities, and no unexpected IDs.
It retries once, then records `ranker_infrastructure_error`. Until a separately
audited gold-free hierarchical selector exists, API dense scoring fails closed
above 30 actions; normal search uses document-order compaction before this gate.

The first answer prompt returned verbose multi-step explanations. A stricter,
gold-free noun-phrase contract fixed that formatting failure. The original
artifact remains `forced_diagnostics_gpt41mini.jsonl`.

## Admission order

```text
generation + evidence validation
  -> factorized PK-only admission
  -> review/explicit waiver for PK-admitted cases only
  -> A/B/C/D
```

`tkg-run --pk-only` no longer requires or records human review. Formal beam runs
check frozen PK before resolving review and before opening the graph backend.

## 2025-2026 candidate attempt

No new case reached machine-pass, so PK was correctly not run.

- 24 provisional questions came from 12 mixed/non-sports event spines. Initial
  promotion was dominated by HTTP 429. A throttled retry promoted two records,
  both the same Greater Manchester/Bev Craig spine; V6 rejected its canonical
  page cycle.
- Profiling admitted P1037, P1416, P127, and P749. P102 was quarantined; global
  P39/P108 queries returned WDQS 504. Current candidate topology could use only
  person-target P1037, which had no complete two-change spine.
- P488 chairperson was quarantined at 1/8 semantic passes. P286 head coach passed
  9/9 and supplied a capped supplementary sports pool.
- Four coach records promoted across three independent spines. Cheap V6 yielded
  one deterministic reject and two shortest-arena HTTP 429 infrastructure
  errors, hence zero machine-pass cases.

The candidate bottlenecks are organization-valued topology coverage, adaptive
P39/P108 queries, exact English-Wikipedia coverage, and explicit API-call
budgets/progress for shortest-arena validation.

## Research boundary

`ApiUtilityRanker` is still an LLM ranking controller-provided graph actions,
not graph-integrated decoding. A later open-weight condition must score the same
actions with length-normalized conditional log probability. These artifacts are
not benchmark accuracy, a model ranking, or a temporal-beam research NO-GO.
