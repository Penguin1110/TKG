#!/usr/bin/env bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --job-name=tkg-localexp-arm
#SBATCH --output=/work/lai0017as/TKG_abcd_localexp_20260818/logs/arm-%j.out
#SBATCH --error=/work/lai0017as/TKG_abcd_localexp_20260818/logs/arm-%j.err

set -euo pipefail
module purge
module load cuda/12.6
: "${PILOT_ARM:?PILOT_ARM must be B, C, or D}"
case "$PILOT_ARM" in B|C|D) ;; *) exit 64 ;; esac

ROOT=/work/lai0017as/TKG_abcd_localexp_20260818
BASE=/work/lai0017as/TKG_abcd_20260818
REPO=$ROOT/repo
ENV=$BASE/env
SOURCE_CACHE=$BASE/repo/examples/temporal_eval_v24/data_line_eventfirst_v3/wikipedia.db
ARM_CACHE=$ROOT/wikipedia_${PILOT_ARM}.db
export HF_HOME=$BASE/hf
export PYTHONPATH=$REPO/src
if [[ ! -f "$ARM_CACHE" ]]; then cp "$SOURCE_CACHE" "$ARM_CACHE"; fi
cd "$REPO"

case "$PILOT_ARM" in
  B) LOCAL_WIDTH=1 ;;
  C) LOCAL_WIDTH=3 ;;
  D) LOCAL_WIDTH=5 ;;
esac
echo "timestamp=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "job_id=${SLURM_JOB_ID}"
echo "experiment=development_local_expansion_only"
echo "arm=${PILOT_ARM} beam_width=${LOCAL_WIDTH} local_expansion_width=${LOCAL_WIDTH} max_expansions=40"
"$ENV/bin/python" --version
nvidia-smi -L
sha256sum \
  examples/temporal_eval_v25/development_5_freeze_20260818/cases.json \
  src/tkg/experiment/open_weight_action_scorer_v24.py \
  src/tkg/experiment/open_weight_live_controller_v24.py \
  src/tkg/experiment/temporal_live_runner_v24.py \
  src/tkg/experiment/open_weight_abcd_engineering_v25.py

"$ENV/bin/python" -m tkg.experiment.open_weight_abcd_engineering_v25 \
  --phase search --arm "$PILOT_ARM" \
  --cases examples/temporal_eval_v25/development_5_freeze_20260818/cases.json \
  --cache-path "$ARM_CACHE" \
  --method-freeze examples/temporal_eval_v24/open_weight_independent_gate_freeze_v24.json \
  --shortcut-summary examples/temporal_eval_v25/shortcut_semantic_freeze_20260817/summary_llm_v3.json \
  --output-dir "$ROOT/output/full_B1_C3_D5" \
  --logprob-batch-size 4 --max-expansions 40

echo "TKG_LOCAL_EXPANSION_ARM_${PILOT_ARM}_COMPLETE"
