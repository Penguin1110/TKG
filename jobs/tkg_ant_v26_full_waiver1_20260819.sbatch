#!/usr/bin/env bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --array=0-14%8
#SBATCH --job-name=tkg-ant-v26-w1
#SBATCH --output=/work/lai0017as/TKG_ant_v26_20260819/logs/waiver1-%A_%a.out
#SBATCH --error=/work/lai0017as/TKG_ant_v26_20260819/logs/waiver1-%A_%a.err

set -euo pipefail
umask 0022
module purge
module load cuda/12.6

ROOT=/work/lai0017as/TKG_ant_v26_20260819
REPO=$ROOT/repo
ENV=/work/lai0017as/TKG_abcd_20260818/env
CACHE_ROOT=/home/lai0017as/TKG_ant_v26_waiver1_cache_20260819
OUTPUT_ROOT=$ROOT/output/waiver1_full
WAIVER_MANIFEST=$REPO/examples/temporal_eval_v26/evidence_pheromone_ant_retry_waiver1_20260819.json
export HF_HOME=/work/lai0017as/TKG_abcd_20260818/hf
export PYTHONPATH=$REPO/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
METHODS=(STOCHASTIC_LM STRUCTURAL_ACO EVIDENCE_ACO)
SEEDS=(11 23 37 53 71)
METHOD=${METHODS[$METHOD_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

# The waiver permits exactly this infrastructure-only change: every task gets
# a brand-new writable SQLite filename, including job and task identifiers.
# Existing databases and sidecars are neither reused nor modified.
mkdir -p "$CACHE_ROOT" "$OUTPUT_ROOT"
CACHE="$CACHE_ROOT/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${METHOD}_seed${SEED}.db"
for candidate in "$CACHE" "$CACHE-shm" "$CACHE-wal"; do
  if [[ -e "$candidate" ]]; then
    echo "WAIVER_CACHE_COLLISION: $candidate" >&2
    exit 73
  fi
done
install -m 0644 \
  "$REPO/examples/temporal_eval_v26/cache/base_wikipedia_before_run.db" \
  "$CACHE"

cd "$REPO"

echo "timestamp=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "job_id=${SLURM_JOB_ID}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "array_task=${SLURM_ARRAY_TASK_ID}"
echo "waiver=RETRY_POLICY_WAIVER_1_OF_1"
echo "experiment=evidence_pheromone_ant_v26_full"
echo "method=${METHOD} seed=${SEED} max_expansions=40"
echo "cache=${CACHE}"
echo "output=${OUTPUT_ROOT}"
"$ENV/bin/python" --version
nvidia-smi -L
sha256sum \
  "$WAIVER_MANIFEST" \
  examples/temporal_eval_v26/evidence_pheromone_ant_freeze_v26.json \
  examples/temporal_eval_v26/public_development_5_cases.json \
  src/tkg/experiment/evidence_pheromone_ant_search_v26.py \
  src/tkg/experiment/evidence_pheromone_ant_experiment_v26.py

"$ENV/bin/python" -m tkg.experiment.evidence_pheromone_ant_experiment_v26 \
  --phase search \
  --public-cases examples/temporal_eval_v26/public_development_5_cases.json \
  --freeze-manifest examples/temporal_eval_v26/evidence_pheromone_ant_freeze_v26.json \
  --cache-path "$CACHE" \
  --output-dir "$OUTPUT_ROOT" \
  --method "$METHOD" --seed "$SEED" \
  --max-expansions 40 --logprob-batch-size 4

echo "TKG_ANT_V26_WAIVER1_${METHOD}_SEED${SEED}_COMPLETE"
