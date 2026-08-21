#!/usr/bin/env bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --job-name=tkg-abcd-mb4
#SBATCH --output=/work/lai0017as/TKG_abcd_20260818/logs/microbatch-%j.out
#SBATCH --error=/work/lai0017as/TKG_abcd_20260818/logs/microbatch-%j.err

set -euo pipefail
module purge
module load cuda/12.6
ROOT=/work/lai0017as/TKG_abcd_20260818
REPO=$ROOT/repo
ENV=$ROOT/env
export HF_HOME=$ROOT/hf
export PYTHONPATH=$REPO/src
cd "$REPO"

echo "timestamp=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "job_id=${SLURM_JOB_ID}"
"$ENV/bin/python" --version
nvidia-smi -L
sha256sum src/tkg/experiment/open_weight_abcd_engineering_v25.py

"$ENV/bin/python" -m tkg.experiment.open_weight_abcd_engineering_v25 \
  --phase search \
  --cases examples/temporal_eval_v25/shortcut_semantic_freeze_20260817/frozen_pk_admitted_cases.json \
  --cache-path examples/temporal_eval_v24/data_line_eventfirst_v3/wikipedia.db \
  --method-freeze examples/temporal_eval_v24/open_weight_independent_gate_freeze_v24.json \
  --shortcut-summary examples/temporal_eval_v25/shortcut_semantic_freeze_20260817/summary_llm_v3.json \
  --output-dir "$ROOT/output/smoke_micro4_max2" \
  --logprob-batch-size 4 \
  --max-expansions 2

echo "TKG_ABCD_MICROBATCH_SMOKE_COMPLETE"
