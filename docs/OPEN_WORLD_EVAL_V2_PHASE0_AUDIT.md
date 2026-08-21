# Open-world evaluation v2: Phase 0 repository audit

Date: 2026-08-16

This audit precedes the append-only v2 implementation. The frozen v1 source and
artifacts remain the historical baseline.

## Existing locations

| Concern | Existing implementation | Audit finding |
| --- | --- | --- |
| Generated/private case schema | `multihop_generation.py`, `case_validation.py` | v6 stores a single `reasoning_chain`, exact source revisions, final aliases, PK probes, and frozen Wikipedia pages. It has no multi-witness claim schema. |
| Public inference boundary | `temporal_beam_runner.py:public_request_from_record` | Projects only question/start/cutoff/target/model and rejects private-looking fields and QIDs. This boundary is reusable and needs stronger v2 regression tests. |
| Private route reads | `temporal_beam_runner.py:posthoc_private_metrics` and `_posthoc_private_action_funnel` | Route, aliases, and exact revisions enter only after constrained search. The scorer nevertheless treats the one route as the action-recall target and uses uniqueness-suggesting labels. |
| End-to-end candidate generation | `temporal_beam.py:legal_actions_with_compaction` | One function mixes environment legality with solver policy: it reads all rendered links, silently retains the first `max_links`, and exposes either `LIST_REVISIONS` or sampled `SWITCH_SNAPSHOT` actions. |
| Revision sampling | `temporal_beam.py:_revision_options` | Resolves a fixed number of evenly spaced calendar probes. These are solver samples, not the complete legal revision set. |
| Forced-state candidates | `temporal_beam_diagnostics.py:_link_actions` and `_compact_to_dense_limit` | Uses a separate builder and reserves non-link slots inside 30. It is not the same builder as end-to-end search. |
| Ranker | `temporal_beam_ranker.py` | API fallback proposes one literal answer candidate, then densely scores at most 30 IDs. Exact ID equality, numeric bounds, one retry, and fail-closed behavior are already correct and reusable. It is not decoding-logit integration. |
| State and trajectory | `temporal_beam.py`, `temporal_beam_runner.py:_JsonlWriter` | State records `(page, revision)`, evidence, history, scores, and pruning. JSONL records compaction sets, but the pre-compaction set is not a complete revision environment. |
| Bridge scoring | `temporal_runner.py:_critical_bridge_evidence_metrics` | Requires one exact source revision per PK bridge. It cannot accept another semantically equivalent revision or route. |
| Failure labels | `temporal_beam_runner.py:_posthoc_private_action_funnel` | Emits `LEGAL_CANDIDATE_RECALL_FAILURE`, `COMPACTION_RECALL_FAILURE`, and `BEAM_PRUNING_FAILURE` against one private route. These labels must remain immutable in raw v1 output and be corrected only in derived v2 audit. |
| PK admission | `temporal_runner.py`, `temporal_beam_runner.py:validate_formal_admission` | Factorized PK admission is already controlled only by designated critical bridges and can be reused before v2 execution. |

## Leakage and validity risks

1. Passing a full v6 case into any inference component would expose aliases,
   exact revisions, private entities, Wikidata QIDs, and the reference chain.
   V2 must accept a public projection only and assert prompt/context contents.
2. The v1 post-hoc scorer equates failure to reproduce one witness route with
   acquisition failure. Open-world success must instead be decided by a
   structured evidence submission; reference-route metrics remain conditional
   diagnostics.
3. `max_links` and `revision_limit` currently look like graph limits even though
   they are solver policy. V2 needs paginated environment APIs with complete
   counts/hashes and separately recorded retrieved and compacted sets.
4. The current answer gate proves only literal string presence. It does not
   establish relation semantics, event time, or bridge-to-tail composition.
5. Forced-state and end-to-end candidate builders differ. V2 will share one
   environment builder while allowing explicitly versioned solver policies.

## Minimal append-only plan

- Add a v2 case loader that converts legacy v6 cases into claim-level witness
  sets without modifying them.
- Add a paginated environment adapter for full rendered links and revision
  metadata, plus exact action-legality checks.
- Add a solver-funnel record with four distinct layers: environment legal,
  solver retrieved, compacted/ranked, and expanded.
- Add structured submissions and a post-hoc validator. Deterministic witness
  checks are preferred; non-witness semantic support requires an auditable judge
  record and remains `machine_pass_human_review_required`.
- Add open-world primary metrics and reference-only conditional diagnostics.
- Replay the four coach development cases into new derived artifacts. No old
  JSONL, SQLite cache, case file, manifest, or source hash is overwritten.
