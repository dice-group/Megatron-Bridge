#!/bin/bash
#
# Submit lm-evaluation-harness jobs for the two 1B-scale routing runs.
# Each job auto-loads the model architecture from <ckpt>/iter_*/run_config.yaml
# (or run_config.yaml at the top level) and pushes per-task metrics to W&B.
#
# Usage:
#   bash scripts/submit_eval_sweep.sh                # H100 (default)
#   GPU_TYPE=a100 bash scripts/submit_eval_sweep.sh  # A100
#
#   TASKS="hellaswag,arc_easy" LIMIT=100 bash scripts/submit_eval_sweep.sh  # smoke test

set -euo pipefail

GPU_TYPE="${GPU_TYPE:-h100}"
WANDB_PROJECT="${WANDB_PROJECT:-variable-moe-routing}"
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,piqa,winogrande,boolq,openbookqa}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
LIMIT="${LIMIT:-}"

REPO_ROOT="$PWD"

# (run_name  checkpoint_dir) pairs
RUNS=(
    "topk_1b_1gpu       $REPO_ROOT/checkpoints_topk_1b_1gpu"
    "topany_1b_1gpu_fb  $REPO_ROOT/checkpoints_topany_1b_1gpu_fb"
)

mkdir -p logs

for entry in "${RUNS[@]}"; do
    read -r name ckpt <<< "$entry"
    if [ ! -d "$ckpt" ]; then
        echo "[skip] $name — missing checkpoint dir $ckpt"
        continue
    fi
    echo "Submitting eval: $name  ($ckpt)"
    # sbatch --export uses commas as variable separators, so values containing
    # commas (TASKS=hellaswag,arc_easy,...) get split into bogus empty vars.
    # Forward via the process environment + --export=ALL instead.
    CHECKPOINT="$ckpt" \
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
done

echo
echo "Submitted ${#RUNS[@]} eval jobs. Track with: squeue -u \$USER"
echo "Results will be logged to W&B project: $WANDB_PROJECT"
