#!/usr/bin/env bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --job-name=tkg-ant-v26-smoke
#SBATCH --output=/work/lai0017as/TKG_ant_v26_20260819/logs/smoke-%j.out
#SBATCH --error=/work/lai0017as/TKG_ant_v26_20260819/logs/smoke-%j.err

set -euo pipefail
module purge
module load cuda/12.6

ROOT=/work/lai0017as/TKG_ant_v26_20260819
REPO=$ROOT/repo
ENV=/work/lai0017as/TKG_abcd_20260818/env
export HF_HOME=/work/lai0017as/TKG_abcd_20260818/hf
export PYTHONPATH=$REPO/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"

echo "timestamp=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "job_id=${SLURM_JOB_ID}"
echo "experiment=evidence_pheromone_ant_v26_smoke"
echo "method=EVIDENCE_ACO seed=11 case=Malaysia max_expansions=2"
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
  --cache-path "$ROOT/cache/smoke_wikipedia.db" \
  --output-dir "$ROOT/output/smoke" \
  --method EVIDENCE_ACO --seed 11 \
  --case-id promoted_c4e2822024602c2c4345 \
  --max-expansions 2 --logprob-batch-size 4

echo "TKG_ANT_V26_SMOKE_COMPLETE"
