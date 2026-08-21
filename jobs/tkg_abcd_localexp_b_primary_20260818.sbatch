#!/usr/bin/env bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --job-name=tkg-localexp-b
#SBATCH --output=/work/lai0017as/TKG_abcd_localexp_20260818/logs/b-primary-%j.out
#SBATCH --error=/work/lai0017as/TKG_abcd_localexp_20260818/logs/b-primary-%j.err

set -euo pipefail
module purge
module load cuda/12.6
: "${PRIMARY_CASE:?PRIMARY_CASE is required}"

ROOT=/work/lai0017as/TKG_abcd_localexp_20260818
BASE=/work/lai0017as/TKG_abcd_20260818
REPO=$ROOT/repo
ENV=$BASE/env
export HF_HOME=$BASE/hf
export PYTHONPATH=$REPO/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ALL_CASES=(
  promoted_c4e2822024602c2c4345
  promoted_00f4be76d3b2d0377623
  promoted_237ae89c2f939b87444c_p19
  promoted_4645b1b1444731a0ddc8
  promoted_590bebda8b580ddf5dd5
)
found=false
CASE_ARGS=(--case-id "$PRIMARY_CASE")
for case_id in "${ALL_CASES[@]}"; do
  if [[ "$case_id" == "$PRIMARY_CASE" ]]; then found=true; continue; fi
  CASE_ARGS+=(--case-id "$case_id")
done
[[ "$found" == true ]] || exit 64

SAFE_CASE=${PRIMARY_CASE//[^A-Za-z0-9_]/_}
CACHE=$ROOT/wikipedia_B_${SAFE_CASE}.db
if [[ ! -f "$CACHE" ]]; then
  cp "$BASE/repo/examples/temporal_eval_v24/data_line_eventfirst_v3/wikipedia.db" "$CACHE"
fi
cd "$REPO"
echo "timestamp=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "job_id=${SLURM_JOB_ID}"
echo "experiment=development_local_expansion_only"
echo "arm=B beam_width=1 local_expansion_width=1 primary_case=${PRIMARY_CASE} max_expansions=40"
"$ENV/bin/python" --version
nvidia-smi -L
sha256sum examples/temporal_eval_v25/development_5_freeze_20260818/cases.json \
  src/tkg/experiment/open_weight_action_scorer_v24.py \
  src/tkg/experiment/open_weight_live_controller_v24.py \
  src/tkg/experiment/temporal_live_runner_v24.py \
  src/tkg/experiment/open_weight_abcd_engineering_v25.py

"$ENV/bin/python" -m tkg.experiment.open_weight_abcd_engineering_v25 \
  --phase search --arm B "${CASE_ARGS[@]}" \
  --cases examples/temporal_eval_v25/development_5_freeze_20260818/cases.json \
  --cache-path "$CACHE" \
  --method-freeze examples/temporal_eval_v24/open_weight_independent_gate_freeze_v24.json \
  --shortcut-summary examples/temporal_eval_v25/shortcut_semantic_freeze_20260817/summary_llm_v3.json \
  --output-dir "$ROOT/output/full_B1_C3_D5" \
  --logprob-batch-size 4 --max-expansions 40

echo "TKG_LOCAL_EXPANSION_B_PRIMARY_${PRIMARY_CASE}_COMPLETE"
