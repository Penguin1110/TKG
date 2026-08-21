# V2.3.1 terminal patch and fresh joint-controller gate

Date: 2026-08-16

## Decision

The terminal-only patch passed its exact cached replay. The fresh joint
navigation/submission gate failed. Fresh real-case generation, machine
validation, PK admission, candidate recall, and A/B/C/D therefore remain locked.

No prompt, compaction, utility, score normalization, tie-break, or search policy
was modified. The failed gate was not rerun.

## V2.3.1 terminal-only patch

The patch activates only when the frozen v2.3 controller has returned a valid
dense response but no candidate can create a child—for example, the only
remaining candidate is `submit_slot:v1` and the controller abstained.

Old behavior:

```text
no executable child
→ empty frontier
→ RuntimeError: v2.3 live search produced no retained state
```

V2.3.1 behavior:

```text
no executable child
→ preserve state, candidates, scores, prompt, raw response and validation
→ finished no-answer state
→ terminal_status = exhausted_no_legal_progress
```

The original five GPT-5.4-mini responses were replayed from the existing cache.
The replay function forbids any cache miss from issuing a model call.

Result artifact:
`examples/temporal_eval_v23/joint_api_gpt54mini_replay_v231.json`.

```text
new_model_calls = 0
cached_controller_responses_replayed = 5
success = false
terminal_status = exhausted_no_legal_progress
complete_trajectory_available = true
runtime_error = false
```

Patch freeze:
`docs/OPEN_WORLD_LIVE_RUNNER_V2_3_1_FREEZE_2026-08-16.json`.

## Fresh gate design

The state manifest was written and hashed before the first model request:
`examples/temporal_eval_v23/fresh_joint_gate_manifest_v231.json`.

It contains 60 independent fictional states:

| Gate | States | Contract |
| --- | ---: | --- |
| Navigation | 20 | 30 candidates; two separately valid progress links |
| Positive submission | 20 | complete bridge, event time and birthplace tail |
| Negative submission | 20 | five each: missing tail, missing bridge, missing event time, relation mismatch |

Navigation never assumes a unique correct action. Either named profile or named
biography link counts as progress. Progress IDs, gate type and positive/negative
labels stay in the post-hoc record and are not included in the controller public
payload.

Gate freeze:
`docs/FRESH_JOINT_CONTROLLER_GATE_V2_3_1_FREEZE_2026-08-16.json`.

Frozen thresholds:

```text
navigation any-progress recall@3 >= 0.90
navigation strict progress separation rate >= 0.80
positive valid-submission recall >= 0.90
negative false-submit rate <= 0.05
dense ranking contract pass rate = 1.00
evidence ownership pass rate = 1.00
gold leakage = 0
```

## One-shot gate result

Model: `openai/gpt-5.4-mini`, temperature 0.0. The run produced 71 cached
responses for 60 states: 49 states used one response and 11 used the permitted
corrective retry.

Artifact: `examples/temporal_eval_v23/fresh_joint_gate_result_v231.json`.
Cache: `examples/temporal_eval_v23/fresh_joint_gate_cache_v231.db`.

| Metric | Result | Threshold | Pass |
| --- | ---: | ---: | --- |
| Navigation any-progress recall@3 | 14/20 = 0.70 | >= 0.90 | No |
| Navigation strict separation | 11/20 = 0.55 | >= 0.80 | No |
| Positive valid submission recall | 0/20 = 0.00 | >= 0.90 | No |
| Negative false-submit rate | 0/20 = 0.00 | <= 0.05 | Yes |
| Dense ranking contract | 60/60 = 1.00 | = 1.00 | Yes |
| Evidence ownership | 1.00 | = 1.00 | Vacuous pass: no submissions survived |
| Gold leakage | 0 | = 0 | Yes |

Positive-state failures:

```text
15 abstained
5 malformed structured submissions after the allowed retry
0 valid submissions
```

All 20 negative states abstained. Six navigation states had no progress action
in the top three; best progress ranks among these were 6, 7, 7, 10, 10 and 25.

The raw positive responses show that the model often recognized the answer in
its reasoning and sometimes assigned high submit-slot utility, but either
abstained or returned string claims instead of the required nested structured
claim objects. This is a response-generation failure under the frozen prompt,
not a dense-ID contract failure. That interpretation is derived from the saved
raw responses; it does not authorize editing this gate.

## Pipeline status

```text
V2.3.1 TERMINAL PATCH = PASS
FRESH JOINT CONTROLLER GATE = FAIL
FRESH REAL CASE GENERATION = LOCKED
MACHINE VALIDATION = NOT STARTED
PK ADMISSION = NOT STARTED
CANDIDATE RECALL = NOT STARTED
A/B/C/D = NOT STARTED
```

Any prompt/schema development must use a new prompt-development set and a new
append-only contract, not these 60 gate states. This remains an external API
controller and is not graph-integrated decoding or benchmark accuracy.
