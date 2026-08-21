# Evidence-Pheromone Ant Search v2.6 — infrastructure incident

## Status

`INFRASTRUCTURE_BLOCKED`; no full-search raw trajectory exists and no GO / MECHANISM_ONLY / NO_SIGNAL decision is permitted.

## What succeeded

- Phase 0 was completed before implementation.
- Eight deterministic synthetic tests passed.
- The search/freeze contract was hash-locked before the first model call.
- Nano4 smoke job `277341` completed successfully on one H200 (`0:0`, 27 seconds).
- The smoke produced two executed transitions, two scored parent states, six candidate actions, full token accounting, and a recomputable two-step pheromone history.
- The isolated inference repository did not contain the private five-case file or Phase-0 route diagnostic.

## Full-run failure

The initial 15-task array `277356` copied a frozen `0444` SQLite base cache into per-task working caches. The copied mode remained read-only. All tasks failed before producing a raw trajectory with:

```text
sqlite3.OperationalError: attempt to write a readonly database
```

The frozen retry policy allowed at most one infrastructure-only resubmission. Retry array `277378` changed only the main working database mode to `0644`. However, the first failed attempt had already created `*.db-shm` sidecars with mode `0440`. Reopening those caches still failed with the same SQLite error. The correct complete infrastructure fix would require a fresh writable working-copy filename or writable/removal handling for SQLite sidecars, but submitting that would be a second retry and would violate the frozen policy without an explicit waiver.

## Artifact accounting

- Smoke raw artifacts: 1.
- Full raw artifacts: 0.
- Preserved attempt-1 error artifacts: 15.
- Initial array tasks failed: 15/15.
- Retry array tasks failed: 15/15.
- Error-artifact SHA-256 aggregate: `ab7f8a3f99ade1592f2301ee2160dbd123c32dae1143e31916bc82d52a9f331e`.
- Complete Slurm states and stdout/stderr are under `examples/temporal_eval_v26/ant_kill_test_20260819/infrastructure_failures/`.

## Scientific interpretation

These failures happened during Wikipedia cache writes before completion of the first case. They provide no evidence for or against stochastic exploration, structural pheromone, evidence pheromone, bridge acquisition, tail acquisition, or submission behavior.

The result must not be reported as `NO_SIGNAL`. Continuing requires an explicit, logged retry-policy waiver; all frozen model/search/reward parameters can remain unchanged.
