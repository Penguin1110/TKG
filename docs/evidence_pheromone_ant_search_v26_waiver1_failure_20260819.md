# Evidence-Pheromone Ant Search v2.6 — waiver retry outcome

## Outcome

`INFRASTRUCTURE_FAILURE; STOPPED; NO FURTHER RETRY AUTHORIZED.`

The one-time retry-policy waiver was used by Slurm array job `278379`.  It changed only the SQLite working-cache paths: each launched task received a new writable filename under `/home/lai0017as/TKG_ant_v26_waiver1_cache_20260819`, including the array job/task ID.  No task shared a database, `-shm`, or `-wal` path.  The frozen model, prompt, action scorer, candidates, search/reward policy, seeds, hyperparameters, 40-transition budget, and decision thresholds were unchanged.

The original read-only/sidecar failure did not recur.  The retry instead encountered a new storage failure while atomically writing a raw trajectory.  After the failure, `/work` reported 100% use with 930 MB available, and task 2 left a 71,979,008-byte temporary raw file rather than a completed atomic artifact.  Task 2 ended `FAILED (1:0)`.  In accordance with the waiver, the remaining running tasks were cancelled and no second retry was submitted.

## Preserved evidence

- Prior failure jobs remain preserved: `277356`, `277378`.
- Waiver job: `278379`; launched task jobs: `278380`–`278387`.
- Complete raw trajectories: 11 of 75 expected.
- Incomplete atomic raw: 1.
- Complete raw validation: all 11 passed schema, private-key guard, expansion-budget, and pheromone-replay checks.
- Accounted model calls: at least 1,896 (1,422 conditional-scoring calls plus 474 generation calls).  The exact total is unavailable because seven tasks were cancelled during an in-progress case before accounting was serialized.
- Local append-only artifact root: `examples/temporal_eval_v26/ant_kill_test_20260819/waiver1_failure/`.
- Machine-readable execution record: `examples/temporal_eval_v26/evidence_pheromone_ant_retry_waiver1_execution_278379.json`.

## Engineering decision

The frozen `GO / MECHANISM_ONLY / NO_SIGNAL` rule is **not applied**.  The matrix is incomplete and no `EVIDENCE_ACO` task started, so the partial trajectories cannot distinguish stochastic exploration from pheromone and cannot support a method conclusion.

Accordingly, the five kill-test questions remain unanswered:

1. Whether ACO rescued a beam-pruned route: unjudgeable.
2. Whether any effect came from stochastic exploration or pheromone: unjudgeable.
3. Whether the complete experiment acquired bridge or tail evidence: unjudgeable; post-hoc private scoring was not run.
4. The new verified bottleneck: experiment artifact storage and atomic-write capacity on `/work`.
5. Whether Evidence-Pheromone Search is worth continuing: no scientific decision from this run.

Any further execution requires new explicit authorization.  Existing partial data must not be relabelled as `NO_SIGNAL`.
