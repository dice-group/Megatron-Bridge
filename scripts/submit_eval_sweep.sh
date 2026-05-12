#!/bin/bash
#
# Submit lm-evaluation-harness jobs for every training run under
# sweep_ckpts_combined_24h. One eval job per run, targeting the latest
# checkpoint (find_run_config + load_checkpoint pick the highest iter_*).
# Per-task metrics push to W&B under the original training run name.
#
# Usage:
#   bash scripts/submit_eval_sweep.sh                   # A100 (default)
#   GPU_TYPE=h100 bash scripts/submit_eval_sweep.sh     # H100
#
#   # smoke test on one run, one task
#   ONLY="small_topk_24h" LIMIT=50 TASKS=hellaswag bash scripts/submit_eval_sweep.sh

set -euo pipefail

CKPT_BASE="${CKPT_BASE:-/scratch/hpc-prf-merlin/luke/Megatron-Bridge/sweep_ckpts_combined_24h}"
GPU_TYPE="${GPU_TYPE:-a100}"
WANDB_PROJECT="${WANDB_PROJECT:-variable-moe-routing}"
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,piqa,winogrande,boolq,openbookqa}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
LIMIT="${LIMIT:-}"
# Optional whitelist: ONLY="name1 name2" filters to those run names.
ONLY="${ONLY:-}"

if [ ! -d "$CKPT_BASE" ]; then
    echo "ERROR: CKPT_BASE not found: $CKPT_BASE" >&2
    exit 1
fi

mkdir -p logs

n_submitted=0
n_skipped=0

for run_dir in "$CKPT_BASE"/*/; do
    run_dir="${run_dir%/}"
    name=$(basename "$run_dir")

    if [ -n "$ONLY" ] && ! [[ " $ONLY " == *" $name "* ]]; then
        continue
    fi

    # Need at least one iter_* with run_config.yaml inside to be evaluable.
    latest_iter=$(ls -d "$run_dir"/iter_* 2>/dev/null | sort -V | tail -1 || true)
    if [ -z "$latest_iter" ] || [ ! -f "$latest_iter/run_config.yaml" ]; then
        echo "[skip] $name — no iter_*/run_config.yaml under $run_dir"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    iter_num=$(basename "$latest_iter" | sed 's/iter_0*//')
    echo "Submitting eval: $name  (latest iter=$iter_num)"

    # sbatch --export parses commas as variable separators, which mangles
    # TASKS=hellaswag,arc_easy,... — forward via the process env + --export=ALL.
    CHECKPOINT="$run_dir" \
    TASKS="$TASKS" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_FEWSHOT="$NUM_FEWSHOT" \
    LIMIT="$LIMIT" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_RUN_NAME="$name" \
    sbatch --gres=gpu:${GPU_TYPE}:1 \
        --job-name="eval-${name}" \
        --export=ALL \
        scripts/slurm_eval_lm_harness.sh
    n_submitted=$((n_submitted + 1))
done

echo
echo "Submitted $n_submitted eval job(s); skipped $n_skipped."
echo "Track:    squeue -u \$USER"
echo "W&B:      $WANDB_PROJECT"
echo "Results:  logs/eval_<jobid>.json + per-task metrics in W&B"
