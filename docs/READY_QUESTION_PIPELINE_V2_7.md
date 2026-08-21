# Ready-question pipeline v2.7

## Why v2.7 exists

The previous candidate-oriented pipeline repeated the complete temporal review
whenever the same bridge spine was paired with a different tail question.  That
was valid but wasteful: birthplace, spouse, and education variants share the
same event-order certificates and post-cutoff bridge.

V2.7 makes the temporal spine the unit of expensive review.

```text
fresh event discovery
  -> sealed-case opaque exclusion
  -> pair-novelty candidate prefilter
  -> group by first three KG edges
  -> promote one representative per spine
  -> immutable spine certificate
  -> validate each distinct tail against the target revision
  -> v2.5 whole-question validation (PK deferred)
  -> one factorized critical-bridge PK gate per spine
  -> reuse that PK decision for machine-pass tail variants
```

## What is shared and what remains per question

Shared once per exact spine:

- both event-order certificates;
- the three bridge hyperlink/evidence checks;
- cutoff-before/target-after semantic contrast;
- bridge evidence and revision IDs;
- factorized critical-bridge PK admission.

Still checked separately for every question:

- final tail property and target;
- target-revision tail hyperlink and evidence;
- answer composition and whole-question validity;
- final answer aliases and structured submission contract.

The final composition check is intentionally not shared. A passage supporting a
person's birthplace does not automatically support their spouse or education.

## Safety and validity boundaries

- `--require-pair-novelty` is opt-in and only removes KG pairs with an earlier
  qualified tenure. It does not replace Wikipedia temporal contrast.
- A spine certificate is the SHA-256 of the exact first three promoted hops,
  including evidence and revisions. Tail expansion cannot alter those hops.
- V2.5 validation remains authoritative. Tail expansion only produces seeds in
  `promoted_pending_v6_validation` state.
- PK reuse is permitted because the frozen admission contract is controlled only
  by the critical bridge. Tail or composed correctness does not admit a case.
- PK reuse additionally requires an exact hash match over every
  `must_be_unknown` probe (question, aliases, event date, and hop index). Matching
  bridge hops with different probe contracts are split into separate PK calls.
- A missing, failed, or unjudgeable representative PK gate admits no member.
- Resume binds both the seed-file hash and the complete validation-config hash.
  Changing `--skip-pk`, models, thresholds, budgets, or cache paths requires a new
  work directory and fails closed if attempted in an existing one.
- Sealed cases are read only by the exclusion guard. Their identities and content
  are never exported into the fresh pool or model context.

## Executable pieces

- `ready_candidate_curation_v27`: opaque sealed exclusion and unique-spine-first
  candidate ordering.
- `candidate_question_batch --require-pair-novelty`: optional cheap KG prefilter.
- `candidate_seed_promotion`: run on one representative candidate per spine.
- `certified_spine_tail_expansion_v27`: reuse the promoted bridge certificate and
  validate only each distinct tail hyperlink.
- `resumable_machine_validation_v24 --skip-pk`: run v2.5 validity without making
  duplicate PK calls.
- `spine_pk_reuse_v27 plan`: emit one representative machine-pass case per spine.
- `spine_pk_reuse_v27 apply`: propagate a valid critical-bridge PK gate to other
  machine-pass cases bound to the identical spine hash.

## Current append-only engineering artifacts

The first fresh screening batch reviewed 20 unique spines:

- 3 promoted;
- 15 validity rejects;
- 2 infrastructure errors, retained as retryable rather than rejected.

The three promoted spine certificates yielded 19 additional tail candidates. Ten
tail hyperlinks passed without repeating bridge review. One original seed reached
v2.5 machine-pass before the live run was intentionally stopped; its PK call was
interrupted and remains pending. No background generation, validation, or PK
process was left running.

These counts are engineering progress only. They are not a 100-question ready
set, benchmark accuracy, or evidence of method superiority.
