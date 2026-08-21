#!/usr/bin/env bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --array=0-14%8
#SBATCH --job-name=tkg-ant-v26-r1
#SBATCH --output=/work/lai0017as/TKG_ant_v26_20260819/logs/retry1-%A_%a.out
#SBATCH --error=/work/lai0017as/TKG_ant_v26_20260819/logs/retry1-%A_%a.err

set -euo pipefail
module purge
module load cuda/12.6

ROOT=/work/lai0017as/TKG_ant_v26_20260819
REPO=$ROOT/repo
ENV=/work/lai0017as/TKG_abcd_20260818/env
export HF_HOME=/work/lai0017as/TKG_abcd_20260818/hf
export PYTHONPATH=$REPO/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
METHODS=(STOCHASTIC_LM STRUCTURAL_ACO EVIDENCE_ACO)
SEEDS=(11 23 37 53 71)
METHOD=${METHODS[$METHOD_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}
CACHE="$ROOT/cache/${METHOD}_seed${SEED}.db"
if [[ ! -f "$CACHE" ]]; then
  cp "$REPO/examples/temporal_eval_v26/cache/base_wikipedia_before_run.db" "$CACHE"
fi
# Infrastructure-only retry fix: the immutable base snapshot is 0444 and cp
# preserves that mode.  The per-task working copy must be writable for cache
# misses; no search/model contract changes here.
chmod 0644 "$CACHE"
cd "$REPO"

echo "timestamp=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "job_id=${SLURM_JOB_ID}"
echo "array_task=${SLURM_ARRAY_TASK_ID}"
echo "retry=1 reason=working_cache_inherited_readonly_mode"
echo "experiment=evidence_pheromone_ant_v26_full"
echo "method=${METHOD} seed=${SEED} max_expansions=40"
echo "output=$ROOT/output/full"
"$ENV/bin/python" --version
nvidia-smi -L
sha256sum \
  examples/temporal_eval_v26/evidence_pheromone_ant_freeze_v26.json \
  examples/temporal_eval_v26/public_development_5_cases.json \
  src/tkg/experiment/evidence_pheromone_ant_search_v26.py \
  src/tkg/experiment/evidence_pheromone_ant_experiment_v26.py

"$ENV/bin/python" -m tkg.experiment.evidence_pheromone_ant_experiment_v26 \
  --phase search \
  --public-cases examples/temporal_eval_v26/public_development_5_cases.json \
  --freeze-manifest examples/temporal_eval_v26/evidence_pheromone_ant_freeze_v26.json \
  --cache-path "$CACHE" \
  --output-dir "$ROOT/output/full" \
  --method "$METHOD" --seed "$SEED" \
  --max-expansions 40 --logprob-batch-size 4

echo "TKG_ANT_V26_RETRY1_${METHOD}_SEED${SEED}_COMPLETE"
