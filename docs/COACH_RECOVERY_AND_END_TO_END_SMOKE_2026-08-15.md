# Coach recovery and end-to-end engineering smoke

Date: 2026-08-15

## Outcome

The blocked coach-case recovery succeeded: four cases passed machine validation
and the factorized prior-knowledge admission gate. The subsequent start-to-end
A/B/C/D engineering smoke did not acquire a complete critical bridge in any
trajectory. The constrained runs localize the first failure before action
ranking: three cases lose the required revision during revision enumeration and
one loses the required hyperlink during frozen document-order compaction.

This is an engineering result, not benchmark accuracy or evidence that one arm,
model, or beam width is better. The run used an API utility ranker, not
graph-integrated model decoding.

## Frozen search contract

The contract is recorded in `SEARCH_CONTRACT_FREEZE_2026-08-15.json`, including
SHA-256 hashes for the beam implementation and forced-state artifacts. It freezes:

- gold-free document-order compaction;
- at most 30 dense candidate actions;
- exact-ID, finite numeric dense-ranker output with one retry and fail-closed
  handling;
- no synthetic score for an omitted action;
- one-entity or short noun-phrase answer candidates with a literal visible-text
  gate;
- Germany and Canada as development-only forced-state cases.

The frozen manifest also discloses an implementation difference: forced-state
diagnostics reserve non-link action slots within 30, while end-to-end defaults
use the first 20 rendered links, up to eight sampled revisions, and one answer
candidate, subject to the same total dense limit. No beam implementation was
changed after freezing, and Germany/Canada were not rerun.

## Infrastructure recovery

The Wikipedia backend now persists page, revision, and backlink-candidate
results. Wikimedia requests have a configurable hard attempt budget, throttling,
Retry-After-aware exponential backoff, and retry statistics. Exhausted budgets
remain infrastructure errors that can be resumed from cache; they are not
converted into semantic rejection.

The two original P286 head-coach cases were resumed with the existing semantic
contract (`openai/gpt-5.4-mini` generator, `openai/gpt-4.1` judge, branch cap 10,
node cap 150, and 0.7-second request interval):

| Case | Shortest path | Semantic paths | Wikipedia API attempts | Result |
| --- | ---: | ---: | ---: | --- |
| `promoted_6f89248c217ed8e81f26_p106` | 3 | 6 | 211 | machine-pass |
| `promoted_30f747fb37666e85b42d_p106` | 3 | 6 | 168 | machine-pass |

A fixed event-first retry batch then reconsidered eight earlier P286
infrastructure errors without issuing a global P39/P108 query. Three seeds
passed promotion; two passed v6 machine validation (205 and 241 Wikipedia API
attempts), while the third was deterministically rejected because its target
appeared before the required temporal switch without a validated contrast.

## Prior-knowledge admission

All four machine-pass cases passed PK-only admission. Each had 0/10 known
critical-bridge probes and zero unjudgeable probes. Tail knowledge and full-
question correctness did not control admission. Human review was explicitly
waived for this machine-only engineering smoke; that waiver is not an approval.

| Source | Machine-pass | PK-admitted |
| --- | ---: | ---: |
| Rescued 429 coach cases | 2 | 2 |
| Event-first retry batch | 2 | 2 |
| Total | 4 | 4 |

## Start-to-end smoke

The four admitted cases were run from their public start pages with the same
model (`openai/gpt-4.1-mini`), seed 17, 40-expansion cap, four expansions per
state, 20-link cap, and eight-revision limit. Arms were A external-tool agent, B
greedy/beam 1, C beam 3, and D beam 5. No forced intermediate state was supplied.

Across the 12 constrained trajectories (B/C/D):

- 0/12 evidenced both critical post-cutoff bridges;
- 0/12 matched an accepted final-answer alias;
- 9/12 received the raw `LEGAL_CANDIDATE_RECALL_FAILURE` label;
- 3/12 received `COMPACTION_RECALL_FAILURE`;
- every observed correct action that survived compaction was covered by the
  dense ranker (`ranker_coverage = 1.0`).

The nine raw legal-recall labels all come from the same three cases and can be
localized more precisely after the run as
`REVISION_ENUMERATION_RECALL_FAILURE`: the correct first hyperlink was legal,
compacted, scored, and retained, but the exact required next revision was absent
from the sampled revision actions. In the fourth case, the correct first
hyperlink was position 120 in full document order and was removed by the frozen
20-link end-to-end compactor before ranking.

| Start page | B/C/D first bottleneck | Interpretation |
| --- | --- | --- |
| SK Brann | required Eirik Horneland revision absent | revision enumeration recall |
| Reading F.C. | Rubén Sellés link at document position 120 | link compaction recall |
| Sunderland A.F.C. | required Mike Dodds revision absent | revision enumeration recall |
| Lille OSC | required Paulo Fonseca revision absent | revision enumeration recall |

Arm A also evidenced 0/2 critical bridges in all four trajectories and did not
contain an accepted final alias. These Arm-A values are a derived post-hoc audit
over the immutable saved trajectories because the baseline records do not emit
the private-route metrics directly. One Arm-A trajectory recorded a tool/cache
error. No independent semantic final-answer judge was added after the run.

One constrained beam branch produced duplicate ranker IDs on both attempts and
was correctly failed closed; other surviving branches allowed that arm to
finish. Thus there was no terminal dense-ranker omission/floor bug, but the raw
artifact still contains this branch-level contract error and it must not be
silently described as error-free.

Only one or two private-route opportunities were actually reached per
constrained trajectory, although a full route contains eight expected actions.
Unvisited downstream actions therefore are not counted as individual ranking
failures.

## Interpretation and next boundary

The recovery pipeline is now capable of producing machine-pass, PK-admitted
cases despite transient Wikimedia failures. The end-to-end search smoke also
works as a failure-localization instrument. It does not yet establish successful
temporal traversal: increasing beam width from 1 to 3 or 5 cannot restore an
action that revision enumeration or compaction never offers.

The search contract remains frozen; this report does not authorize tuning it on
these four smoke cases. A future, separately declared change should be evaluated
on fresh development cases. Open-weight length-normalized conditional action
log-probability scoring remains unimplemented and should start only after an
end-to-end candidate-recall smoke succeeds.

## Artifacts

- Frozen contract: `docs/SEARCH_CONTRACT_FREEZE_2026-08-15.json`
- Recovered cases: `examples/temporal_beam_new_questions_v1/rare_coach_recovery_cases_v1.json`
- Recovered PK: `examples/temporal_beam_new_questions_v1/rare_coach_recovery_pk_v1.jsonl`
- Event-first v6 cases: `examples/temporal_beam_new_questions_v1/rare_coach_eventfirst_v6_cases_v1.json`
- Event-first PK: `examples/temporal_beam_new_questions_v1/rare_coach_eventfirst_pk_v1.jsonl`
- End-to-end batch 1: `examples/temporal_beam_new_questions_v1/end_to_end_batch1_results_v1.jsonl`
- End-to-end batch 2: `examples/temporal_beam_new_questions_v1/end_to_end_batch2_results_v1.jsonl`

The JSONL trajectories and SQLite caches are raw run artifacts. This Markdown
file is a derived audit and does not alter them.
