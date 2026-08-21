#!/usr/bin/env bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --job-name=tkg-localexp-smoke
#SBATCH --output=/work/lai0017as/TKG_abcd_localexp_20260818/logs/smoke-%j.out
#SBATCH --error=/work/lai0017as/TKG_abcd_localexp_20260818/logs/smoke-%j.err

set -euo pipefail
module purge
module load cuda/12.6

ROOT=/work/lai0017as/TKG_abcd_localexp_20260818
BASE=/work/lai0017as/TKG_abcd_20260818
REPO=$ROOT/repo
ENV=$BASE/env
export HF_HOME=$BASE/hf
export PYTHONPATH=$REPO/src
if [[ ! -f "$ROOT/wikipedia_smoke.db" ]]; then
  cp "$BASE/repo/examples/temporal_eval_v24/data_line_eventfirst_v3/wikipedia.db" "$ROOT/wikipedia_smoke.db"
fi
cd "$REPO"

echo "timestamp=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "job_id=${SLURM_JOB_ID}"
echo "experiment=development_local_expansion_only"
echo "arm=D beam_width=5 local_expansion_width=5 max_expansions=2"
"$ENV/bin/python" --version
nvidia-smi -L
sha256sum \
  examples/temporal_eval_v25/development_5_freeze_20260818/cases.json \
  src/tkg/experiment/open_weight_action_scorer_v24.py \
  src/tkg/experiment/open_weight_live_controller_v24.py \
  src/tkg/experiment/temporal_live_runner_v24.py \
  src/tkg/experiment/open_weight_abcd_engineering_v25.py

"$ENV/bin/python" -m tkg.experiment.open_weight_abcd_engineering_v25 \
  --phase search --arm D \
  --cases examples/temporal_eval_v25/development_5_freeze_20260818/cases.json \
  --cache-path "$ROOT/wikipedia_smoke.db" \
  --method-freeze examples/temporal_eval_v24/open_weight_independent_gate_freeze_v24.json \
  --shortcut-summary examples/temporal_eval_v25/shortcut_semantic_freeze_20260817/summary_llm_v3.json \
  --output-dir "$ROOT/output/smoke_D_max2" \
  --logprob-batch-size 4 --max-expansions 2

echo "TKG_LOCAL_EXPANSION_SMOKE_COMPLETE"
