# Temporal evaluation v2.4: independent data and method lines

Date: 2026-08-16

## Corrected authorization policy

The failed v2.3.1 joint-controller gate remains a NO-GO for API A/B/C/D. It
does not gate case construction or admission. The executable policy is in
`src/tkg/experiment/pipeline_policy_v24.py`:

```text
Data line:   fresh generation -> machine validation -> PK admission --------┐
                                                                            ├-> A/B/C/D
Method line: controller development -> method gate -------------------------┘
```

Fresh generation, machine validation, and PK admission remain authorized even
when every solver-method gate is false. Each A/B/C/D method requires both an
admitted case pool and its own passed method gate.

The append-only correction in
`docs/PIPELINE_INDEPENDENCE_CORRECTION_2026-08-16.md` supersedes the single
incorrect lock field in the frozen v2.3.1 gate manifest. No frozen response,
score, threshold, prompt, or v2.3.1 source file was changed.

## v2.4 compact submission

The model now submits only:

```json
{
  "schema_version": "compact-temporal-evidence-submission-v2.4",
  "answer": "Harbor City",
  "bridge_evidence_ids": ["ev_bridge"],
  "tail_evidence_ids": ["ev_tail"]
}
```

The public inference-time gate checks shape, a 1--8 word answer, trajectory
ownership of every evidence ID, and literal answer presence in cited tail
evidence. It does not receive private claims or reference routes.

After search, the private evaluator supplies the expected bridge/tail claim
shapes and event times, attaches the model's cited evidence IDs, and invokes the
existing semantic and temporal validator. Thus the model no longer has to
restate subject, relation, object, or event time, while a guessed answer without
bridge and tail evidence still cannot pass.

The v2.4 runner starts at the public case's start page and performs live graph
expansion. Its deterministic multi-route test reaches an alternative legal
route, submits the compact payload, and passes the existing private post-hoc
evaluator. This test is engineering evidence only, not a model result.

## Independent prompt-development run

The new development set contains 12 fictional states that do not overlap the
frozen 60-state v2.3.1 gate:

| State family | Count | Development result |
|---|---:|---:|
| Navigation with two acceptable progress actions | 4 | progress in top 3: 3/4 |
| Complete bridge and tail evidence | 4 | valid compact submission: 4/4 |
| Missing bridge or missing tail | 4 | false valid submission: 0/4 |

All 12 calls satisfied the exact dense action-ID contract on the first attempt.
These are prompt-development observations only. They do not constitute a fresh
gate, do not erase the v2.3.1 API-controller NO-GO, and cannot unlock A/B/C/D.

Artifacts:

- `examples/temporal_eval_v24/compact_prompt_dev_manifest_v24.json`
- `examples/temporal_eval_v24/compact_prompt_dev_result_v24.json`
- `examples/temporal_eval_v24/compact_prompt_dev_cache_v24.db`

## Open-weight action scoring

`src/tkg/experiment/open_weight_action_scorer_v24.py` implements per-action
length-normalized conditional log probability:

```text
score(action) = mean token log P(serialized action | public state prompt)
```

Every legal compacted action is scored separately, so there is no sparse JSON
utility output and no missing-action floor. A lazy Hugging Face causal-LM
backend computes token log probabilities from actual logits. Unit tests use a
fake logit backend to verify exact averaging and fail-closed handling of empty
or non-finite scores.

This scorer has started the open-weight method line, but it is not yet wired to
the live v2.4 controller's submission decision and has not been run with a real
open-weight checkpoint. Therefore graph-integrated decoding is still not an
achieved result.

## Fresh data-line attempt

A new, append-only candidate batch was run through the active semantic-profile
intersection for the 2024-06-01 to 2026-08-16 window. The only topology-
compatible admitted direct relation was P1037. Its strict three-event temporal
query returned zero rows, so the batch contains zero fresh candidates.

```text
fresh candidates: 0
machine validation reached: 0
PK admission reached: 0
```

This is an empty data result, not a controller lock and not a validity reject.
No old question was relabeled as fresh, and no A/B/C/D run was started.

Artifacts:

- `examples/temporal_eval_v24/data_line_20260816/candidates.json`
- `examples/temporal_eval_v24/data_line_20260816/candidate_packets.jsonl`
- `examples/temporal_eval_v24/data_line_20260816/candidates.md`

## Verification

```text
pytest: 235 passed
mypy: 64 source files, no issues
pyflakes: clean
```

Current status:

```text
v2.3.1 API controller baseline = NO-GO, preserved
data line = UNLOCKED, latest fresh batch empty
v2.4 compact live runner = engineering prototype passed synthetic test
v2.4 prompt development = 12-state diagnostic completed
open-weight logprob scorer = implemented and unit-tested, not live-integrated
API A/B/C/D = LOCKED
open-weight A/B/C/D = LOCKED
formal benchmark experiment = NOT STARTED
```
