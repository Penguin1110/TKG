# Local-expansion-only engineering rerun

## Scope and frozen baseline

This is a development-only rerun on the same five frozen cases. The remaining twelve cases were neither copied into the remote experiment repository nor run or inspected. No formal accuracy or method-superiority conclusion is allowed.

The original 5 x 4 results remain unchanged at `examples/temporal_eval_v25/abcd_engineering_5x4_h200_20260818/scored_v2`. Its summary SHA-256 is `987cf49622e0a85d23090f9e1ce2d0cb187406650922466831d67073a969163c`; all four arms had 0/5 end-to-end success. Arm A was not rerun.

Only the number of children expanded per state changed:

| Arm | Global beam | Local expansions per state |
| --- | ---: | ---: |
| B | 1 | 1 |
| C | 3 | 3 |
| D | 5 | 5 |

The five-case manifest, Qwen checkpoint, prompt, conditional-logprob scorer, candidate generation, 30-action compaction, cumulative scoring, tie-breaks, evaluator, and 40-expansion budget remained fixed. The frozen hashes are recorded in `comparison.json` and `raw_contract_audit.json`.

## Aggregate result

| Arm | Bridge | Tail | Complete evidence | Correct answer | First-failure stages |
| --- | ---: | ---: | ---: | ---: | --- |
| B | 0/5 | 0/5 | 0/5 | 0/5 | candidate 5 |
| C | 0/5 | 0/5 | 0/5 | 0/5 | candidate 1, ranking 4 |
| D | 0/5 | 0/5 | 0/5 | 0/5 | candidate 1, ranking 2, beam 2 |

No arm reached tail extraction or structured submission. The rerun therefore does not establish end-to-end success.

## Did D execute the known rank-4 actions?

Yes.

| Case | Correct progress action rank | C (top 3) | D (top 5) | Next failure |
| --- | ---: | --- | --- | --- |
| Malaysia | 4 | not expanded | expanded | resulting state globally pruned |
| AC Ajaccio | 4 | not expanded | expanded | resulting state globally pruned |

This rules out a runner bug in local top-5 expansion. Widening crossed the previously observed top-3 boundary and executed both actions. However, neither child survived global beam selection, so neither trajectory reached the correct revision or acquired bridge evidence.

The first new bottleneck is therefore cumulative-score/global-beam pruning, not local expansion. The result is a real mechanism-level signal for widening, but not yet an evidence-acquisition or QA-success signal.

## Other three cases

- Raja CA required a pagination action ranked 12; neither C nor D expanded it. The frozen evaluator conservatively labels this `candidate` because it does not reinterpret unexpanded pagination ranks.
- Lille OSC's progress link ranked 17; neither C nor D expanded it.
- Milton Keynes Dons' progress link ranked 10; neither C nor D expanded it.

These remain action-ranking/local-cutoff failures under the frozen scorer. Increasing local top-k further was not attempted.

Arm B also behaved as expected under strict top-1: a separate raw-trajectory audit shows that in all five cases the first useful `LIST_LINKS` pagination action ranked 2 and was not expanded. The frozen evaluator conservatively reports these as `candidate` failures. This is distinct from the old B run, where the wider local expansion created the pagination child and global beam width 1 then pruned it.

## Engineering and infrastructure audit

- Fifteen new raw trajectories exist: five each for B, C, and D.
- Every raw artifact records the intended `(global beam, local expansion)` pair and `max_expansions=40`.
- C and D completed normally on Nano4 H200 jobs `274919` and `274920`.
- The first AC B shard (`275062`) failed from CUDA allocator fragmentation. Its retry (`275081`) used `expandable_segments:True` only; model, scoring, action order, and search policy were unchanged. The retry completed normally.
- B shards for Raja, Lille, and MK Dons were deliberately cancelled only after their complete target artifact had been atomically written, preventing duplicate work on other cases. Malaysia had already been preserved from the initial B job before that job was similarly stopped.
- Complete scheduler states are saved in `slurm_job_states.txt`.
- Canonical scores are in `scored_frozen_evaluator`. A briefly generated finer-grained pagination-label variant in `scored` is non-canonical and excluded because changing that classifier would violate the fixed-evaluator contract.

## Engineering conclusion

The controlled local-expansion change worked mechanically: D executed the two known rank-4 actions that C could not. Both paths went farther than before, but both were immediately removed by global pruning, and all final evidence/answer metrics remained zero.

Under the preregistered interpretation, this is not enough to call beam search successful or failed. The top-3 obstruction is resolved for those two states; the next isolated issue is global state scoring/retention. The remaining twelve cases stay sealed, and no formal run is authorized by this result.
