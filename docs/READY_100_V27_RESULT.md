# Ready 100 result (v2.7)

Status: engineering data milestone complete.

The append-only output is `examples/ready_questions_100_v1/READY_100_CASES_V2.json`.
It contains exactly 100 unique case IDs and 100 unique public questions. Every
included case passed the v2.5 whole-question machine validity gate and the
factorized critical-bridge prior-knowledge admission gate. Human evidence review
is not included and was waived for this engineering dataset.

## Funnel

- Previously admitted fresh cases: 7.
- Newly admitted, deduplicated cases: 93.
- Final ready set: 100.
- Temporal spines represented by the new pool: 16.
- Extended linked-tail validation: 53 machine-pass out of 60 sanitized cases.
- Two synthesized tails were removed before validation because they would create
  an entity cycle.
- One earlier 30-case shard failed as infrastructure because that cycle caused
  batch-level seed loading to fail. The raw failure remains append-only; no case
  in that shard was interpreted as a validity reject.

## Current composition

- Domain labels: 96 sports, 4 politics.
- Most common final tail relations: 45 sports-team membership, 23 birthplace,
  13 occupation.
- Other represented tails include citizenship, employer, award, child,
  education, native language, work location, spoken language, and spouse.

This set is ready for engineering use, but it is not domain-balanced. The current
event funnel is dominated by the P286 head-coach relation. It must not be
described as a cross-domain benchmark or a formal test set without a later
domain-balanced fresh-case build and the required review policy.

## Verification

- 36 focused pipeline tests passed.
- Pyflakes passed for the four modified v2.7 pipeline modules.
- Python bytecode compilation passed.
- Final case IDs and public questions are both unique.
- No generation, validation, or promotion process was left running.

