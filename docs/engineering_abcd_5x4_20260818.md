# Temporal Graph A/B/C/D Engineering Pilot (5 x 4)

Date: 2026-08-18  
Status: engineering diagnostic only; not a benchmark result

## Contract

- Model: `Qwen/Qwen3.5-4B` for all four arms.
- A: legacy external-tool agent.
- B: temporal constrained greedy search, beam width 1.
- C: temporal constrained beam search, beam width 3.
- D: temporal constrained beam search, beam width 5.
- Search budget: at most 40 global expansions per case-arm.
- Candidate contract: document-order compaction at 30 actions; at most 3
  graph actions expanded per state.
- Scoring: frozen length-normalized conditional action log probability.
- Execution-only log-probability microbatch: 4 actions. This changes memory
  scheduling, not prompts, action strings, scores, ordering, or beam policy.
- Private routes and accepted answers were used only after search for evidence
  scoring and failure diagnosis.

The five cases were selected before running and cover education, birthplace,
citizenship, and award tails plus multiple shortcut-audit categories.

| Start page | Accepted answer |
| --- | --- |
| Malaysia national football team | Yeungnam University |
| AC Ajaccio | Villers-Semeuse |
| Raja CA | Sidi Kacem |
| Lille OSC | Portugal |
| Milton Keynes Dons F.C. | OBE |

## End-to-end results

| Arm | Bridge | Tail | Complete evidence submission | Correct answer |
| --- | ---: | ---: | ---: | ---: |
| A: external-tool | 1/5 | 0/5 | 0/5 | 0/5 |
| B: beam 1 | 0/5 | 0/5 | 0/5 | 0/5 |
| C: beam 3 | 0/5 | 0/5 | 0/5 | 0/5 |
| D: beam 5 | 0/5 | 0/5 | 0/5 | 0/5 |

All B/C/D searches stopped at the 40-expansion limit without a structured
submission. The evidence-conditioned answer generator abstained while bridge
or tail evidence was incomplete. A uses the legacy answer-only submission
contract, so its evidence-submission cell is structurally unavailable; it also
never collected both bridge and tail in any case.

## Failure localization

| Case | A | B: beam 1 | C: beam 3 | D: beam 5 |
| --- | --- | --- | --- | --- |
| Malaysia | action selection | link-pagination state pruned | correct link ranked 4, below top 3 | correct link ranked 4, below top 3 |
| AC Ajaccio | bridge found, tail missed | link-pagination state pruned | correct link ranked 4, below top 3 | correct link ranked 4, below top 3 |
| Raja CA | action selection | link-pagination state pruned | required link not retrieved | required link not retrieved |
| Lille OSC | partial bridge only | link-pagination state pruned | correct link ranked 17 | correct link ranked 17 |
| Milton Keynes Dons | invalid hyperlink action | link-pagination state pruned | correct link ranked 10 | correct link ranked 10 |

The first post-hoc classifier incorrectly stopped at the initial state, before
`LIST_LINKS`, and labeled all constrained runs as candidate failures. The raw
audit showed that this was wrong. Recomputed classification preserves the raw
trajectories and changes no model output:

- B: 5 beam-pruning failures. `LIST_LINKS` was expanded but lost when only one
  state could be retained.
- C: 1 candidate-retrieval failure and 4 ranking failures.
- D: 1 candidate-retrieval failure and 4 ranking failures.
- No constrained run reached a tail or submission failure.

## Is there a beam signal?

There is a limited mechanism-level signal, but no outcome-level success:

- Beam width 1 discarded the hyperlink-pagination branch in all five cases.
- Beam widths 3 and 5 preserved that branch and placed the required first link
  in the compact scored set in 4/5 cases.
- Width 5 did not improve over width 3. Both are capped at three expanded
  actions per state, so actions ranked 4, 10, or 17 cannot be rescued by a
  larger global beam.

Thus graph-constrained search now runs end to end and wider beams preserve more
useful search states, but this pilot provides no evidence that beam search can
complete the task. The immediate bottleneck is the interaction between
within-state top-3 expansion and action ranking, followed by pagination recall
for the Raja CA case.

## Engineering verification

- H200 raw-search jobs: A `273163`, B `273165`, C `272804`, D `272808`.
- All four finished `COMPLETED` with exit code `0:0` and completion markers.
- 20/20 raw JSON trajectories passed schema and unique case-arm validation.
- Frozen method source hashes were checked in the Slurm logs.
- Local checks: 9 targeted tests passed; mypy and pyflakes passed for the new
  orchestrator.
- Formal interpretation remains disallowed.

Artifacts:

- `examples/temporal_eval_v25/abcd_engineering_5x4_h200_20260818/scored_v2/summary.json`
- `examples/temporal_eval_v25/abcd_engineering_5x4_h200_20260818/scored_v2/*.raw.json`
- `examples/temporal_eval_v25/abcd_engineering_5x4_h200_20260818/slurm_logs/`
