#!/usr/bin/env bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --job-name=tkg-abcd-arm
#SBATCH --output=/work/lai0017as/TKG_abcd_20260818/logs/arm-%j.out
#SBATCH --error=/work/lai0017as/TKG_abcd_20260818/logs/arm-%j.err

set -euo pipefail
module purge
module load cuda/12.6
: "${PILOT_ARM:?PILOT_ARM must be A, B, C, or D}"
case "$PILOT_ARM" in A|B|C|D) ;; *) exit 64 ;; esac

ROOT=/work/lai0017as/TKG_abcd_20260818
REPO=$ROOT/repo
ENV=$ROOT/env
SOURCE_CACHE=$REPO/examples/temporal_eval_v24/data_line_eventfirst_v3/wikipedia.db
ARM_CACHE=$ROOT/cache_${PILOT_ARM}.db
export HF_HOME=$ROOT/hf
export PYTHONPATH=$REPO/src
if [[ ! -f "$ARM_CACHE" ]]; then
  cp "$SOURCE_CACHE" "$ARM_CACHE"
fi
cd "$REPO"

echo "timestamp=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "job_id=${SLURM_JOB_ID}"
echo "pilot_arm=${PILOT_ARM}"
"$ENV/bin/python" --version
nvidia-smi -L
sha256sum src/tkg/experiment/open_weight_abcd_engineering_v25.py

"$ENV/bin/python" -m tkg.experiment.open_weight_abcd_engineering_v25 \
  --phase search \
  --arm "$PILOT_ARM" \
  --cases examples/temporal_eval_v25/shortcut_semantic_freeze_20260817/frozen_pk_admitted_cases.json \
  --cache-path "$ARM_CACHE" \
  --method-freeze examples/temporal_eval_v24/open_weight_independent_gate_freeze_v24.json \
  --shortcut-summary examples/temporal_eval_v25/shortcut_semantic_freeze_20260817/summary_llm_v3.json \
  --output-dir "$ROOT/output/full_max40_micro4" \
  --logprob-batch-size 4 \
  --max-expansions 40

echo "TKG_ABCD_ARM_${PILOT_ARM}_SEARCH_COMPLETE"
