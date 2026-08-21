# Evidence-Pheromone Ant Search v2.6 — Phase 0 audit

## Scope

This is a read-only reconstruction from the frozen five-case local-expansion rerun. No search implementation was changed before this report was written. The remaining twelve cases were not copied, executed, or inspected.

Audited raw artifacts:

- Malaysia D raw SHA-256: `259883f4468ccd8d6e84709f3cc9910500a40411b7fcb7e87e90114ceb1066e8`
- AC Ajaccio D raw SHA-256: `ec1ad4291ab5db083c649553070af454584a5694f28fcd51c03171aeca19b367`

Frozen runner/scorer/controller hashes were respectively `3785a6f3...0817`, `d20e21f3...1471`, and `21a62af7...b978`.

## Rank-4 child reconstruction

The recorded action score is the length-normalized conditional log-probability used by the frozen controller. Child cumulative score is exactly `parent cumulative score + local action score`.

| Quantity | Malaysia | AC Ajaccio |
| --- | ---: | ---: |
| Correct action | `FOLLOW_LINK(Kim Pan-gon)` | `FOLLOW_LINK(Olivier Pantaloni)` |
| Local graph rank | 4 | 4 |
| Local action score | -3.734529 | -4.124831 |
| Parent cumulative score | -1.555480 | -1.907135 |
| Resulting child score | -5.290009 | -6.031966 |
| Expanded children in global pool | 10 | 10 |
| Child global rank | 9 | 9 |
| Beam-5 cutoff score | -4.845348 | -5.629763 |
| Pruning margin below cutoff | 0.444661 | 0.402203 |

Both correct actions were executed. Both child states were valid and materialized, but neither survived the global beam-5 truncation.

## What displaced the correct children

All five retained Malaysia children came from the competing revision-list branch whose parent score was -1.009229:

| Retained action | Resulting revision | Child score |
| --- | ---: | ---: |
| switch snapshot | 1227174758 | -4.819908 |
| switch snapshot | 1226865462 | -4.820551 |
| switch snapshot | 1227175747 | -4.833151 |
| switch snapshot | 1226864688 | -4.842346 |
| continue revision pagination | 1226802931 | -4.845348 |

All five retained AC Ajaccio children likewise came from its revision-list branch whose parent score was -1.442201:

| Retained action | Resulting revision | Child score |
| --- | ---: | ---: |
| continue revision pagination | 1222743401 | -5.504609 |
| switch snapshot | 1234033567 | -5.527575 |
| switch snapshot | 1242596027 | -5.551108 |
| switch snapshot | 1228225750 | -5.560882 |
| switch snapshot | 1234033175 | -5.629763 |

The correct local action itself was not exceptionally poor. In Malaysia it was 0.101590 better than the retained cutoff action, but its parent already trailed the competing parent by 0.546252, leaving the child 0.444661 below cutoff. In AC it recovered 0.062731 locally against the cutoff action but inherited a 0.464934 parent deficit, leaving a 0.402203 pruning margin.

This directly supports the proposed kill-test mechanism: cumulative history, rather than the immediate correct action score alone, removed a path with possible delayed value.

## Path-length effect

Cumulative score is an unnormalized sum of negative per-action average log-probabilities. It therefore decreases monotonically as path length grows.

Mean parent-state cumulative scores by action-trace depth were:

| Depth | Malaysia | AC Ajaccio |
| ---: | ---: | ---: |
| 0 | 0.000000 | 0.000000 |
| 1 | -1.282354 | -1.674668 |
| 2 | -4.832261 | -5.554787 |
| 3 | -5.986756 | -7.119846 |

This is not proof that every longer path is semantically better, but it confirms a structural length penalty capable of eliminating delayed-reward paths.

## What `max_expansions=40` counts

The frozen runner increments `expansions` once for every executed candidate action after transition execution. This includes graph actions, pagination/control actions, snapshot switches, and instantiated submit actions. It does **not** count:

- parent states sent to the controller;
- candidate actions merely scored but not executed;
- conditional-logprob sequences evaluated by the model;
- answer-generation calls;
- tokens.

For each audited D trajectory:

| Counter | Value |
| --- | ---: |
| Executed transitions / reported expansions | 40 |
| Parent states scored | 13 |
| Public action candidates ranked, including submit slots | 255 |
| Graph-action conditional continuations | 242 |
| Mode-label conditional continuations | 52 |
| Total conditional continuations | 294 |
| Evidence-conditioned answer-generation calls | 13 |

Exact prompt tokens, continuation tokens, generated tokens, and per-case wall time are not recoverable from these raw files. The five-case D Slurm job took 25:23 in aggregate, but no per-case timer was recorded.

## Budget contract for the kill test

All three ant-search methods will therefore be capped at exactly 40 executed transitions per case and seed. Because stochastic trajectories can expose different candidate counts, the runner must additionally record—without using them for admission or reward—parent states scored, compacted actions scored, conditional continuation tokens, generated tokens, model calls, and wall time. These quantities will be reported separately rather than falsely claimed to be exactly matched.

## Phase 0 conclusion

The narrow failure mechanism is real in both audited cases:

```text
rank-4 correct action executed
→ valid child created
→ child global rank 9/10
→ beam-5 removes it because of inherited cumulative-score deficit
```

This justifies proceeding to an append-only stochastic/pheromone engineering kill test. It does not establish that ACO will recover bridge evidence or outperform beam search.
