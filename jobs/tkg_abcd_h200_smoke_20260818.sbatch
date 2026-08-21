#!/usr/bin/env bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --job-name=tkg-abcd-smoke
#SBATCH --output=/work/lai0017as/TKG_abcd_20260818/logs/smoke-%j.out
#SBATCH --error=/work/lai0017as/TKG_abcd_20260818/logs/smoke-%j.err

set -euo pipefail
module purge
module load cuda/12.6

ROOT=/work/lai0017as/TKG_abcd_20260818
REPO=$ROOT/repo
ENV=$ROOT/env
export HF_HOME=$ROOT/hf
export PYTHONPATH=$REPO/src

if [[ ! -x "$ENV/bin/python" ]]; then
  /work/lai0017as/classroom_loop_env/bin/python -m venv --system-site-packages "$ENV"
fi
if ! "$ENV/bin/python" -c 'import transformers; assert hasattr(transformers, "Qwen3_5ForConditionalGeneration")' >/dev/null 2>&1; then
  "$ENV/bin/pip" install --upgrade \
    'git+https://github.com/huggingface/transformers.git@7ea2320c76117e6742364808a666ef6f2fb40a67' \
    'python-dotenv>=1.0.0'
fi

cd "$REPO"
echo "timestamp=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "job_id=${SLURM_JOB_ID}"
echo "python=$($ENV/bin/python -c 'import sys; print(sys.executable)')"
"$ENV/bin/python" --version
nvidia-smi -L
sha256sum \
  src/tkg/experiment/open_weight_action_scorer_v24.py \
  src/tkg/experiment/open_weight_live_controller_v24.py \
  src/tkg/experiment/temporal_live_runner_v24.py \
  src/tkg/experiment/open_weight_abcd_engineering_v25.py

"$ENV/bin/python" -m tkg.experiment.open_weight_abcd_engineering_v25 \
  --phase search \
  --cases examples/temporal_eval_v25/shortcut_semantic_freeze_20260817/frozen_pk_admitted_cases.json \
  --cache-path examples/temporal_eval_v24/data_line_eventfirst_v3/wikipedia.db \
  --method-freeze examples/temporal_eval_v24/open_weight_independent_gate_freeze_v24.json \
  --shortcut-summary examples/temporal_eval_v25/shortcut_semantic_freeze_20260817/summary_llm_v3.json \
  --output-dir "$ROOT/output/smoke_max1" \
  --max-expansions 1

echo "TKG_ABCD_SMOKE_COMPLETE"
