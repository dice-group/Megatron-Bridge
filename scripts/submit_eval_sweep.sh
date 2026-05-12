#!/bin/bash
#
# Submit lm-evaluation-harness jobs for every run from the variable-moe-routing
# sweeps (advanced routing + kanneal + topk baseline). One sbatch job per run.
# Each job auto-loads the model architecture from <ckpt>/run_config.yaml and
# pushes per-task metrics to W&B under the original training run name.
#
# Usage:
#   bash scripts/submit_eval_sweep.sh                # A100 (default)
#   GPU_TYPE=h100 bash scripts/submit_eval_sweep.sh  # H100
#
#   TASKS="hellaswag,arc_easy" LIMIT=100 bash scripts/submit_eval_sweep.sh  # smoke test

set -euo pipefail

GPU_TYPE="${GPU_TYPE:-a100}"
WANDB_PROJECT="${WANDB_PROJECT:-variable-moe-routing}"
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,piqa,winogrande,boolq,openbookqa}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
LIMIT="${LIMIT:-}"

# Checkpoint roots from the three sweep scripts.
ADV_BASE=/scratch/hpc-prf-merlin/luke/Megatron-Bridge/sweep_ckpts_advanced_24h
KAN_BASE=/scratch/hpc-prf-merlin/luke/Megatron-Bridge/sweep_ckpts_kanneal_24h
TOPK_BASE=/scratch/hpc-prf-merlin/luke/Megatron-Bridge/sweep_ckpts_24h

# (run_name  checkpoint_dir) pairs — order matches the W&B legend in eval_runs.png
RUNS=(
    "advrouting_remoe_k2_24h        $ADV_BASE/advrouting_remoe_k2_24h"
    "advrouting_remoe_k1p5_24h      $ADV_BASE/advrouting_remoe_k1p5_24h"
    "advrouting_adamoe_m16_k3_24h   $ADV_BASE/advrouting_adamoe_m16_k3_24h"
    "advrouting_dtopp_k2_24h        $ADV_BASE/advrouting_dtopp_k2_24h"
    "advrouting_dtopp_k1p5_24h      $ADV_BASE/advrouting_dtopp_k1p5_24h"
    "advrouting_adamoe_m8_k2_24h    $ADV_BASE/advrouting_adamoe_m8_k2_24h"
    "kanneal_k1p5_slow_24h          $KAN_BASE/kanneal_k1p5_slow_24h"
    "kanneal_k1p25_24h              $KAN_BASE/kanneal_k1p25_24h"
    "kanneal_aux_24h                $KAN_BASE/kanneal_aux_24h"
    "kanneal_sign_24h               $KAN_BASE/kanneal_sign_24h"
    "kanneal_slow_24h               $KAN_BASE/kanneal_slow_24h"
    "small_topk_24h                 $TOPK_BASE/small_topk_24h"
)

mkdir -p logs

for entry in "${RUNS[@]}"; do
    read -r name ckpt <<< "$entry"
    if [ ! -d "$ckpt" ]; then
        echo "[skip] $name — missing checkpoint dir $ckpt"
        continue
    fi
    echo "Submitting eval: $name"
    sbatch --gres=gpu:${GPU_TYPE}:1 \
        --job-name="eval-${name}" \
        --export=ALL,CHECKPOINT=$ckpt,TASKS=$TASKS,BATCH_SIZE=$BATCH_SIZE,NUM_FEWSHOT=$NUM_FEWSHOT,LIMIT=$LIMIT,WANDB_PROJECT=$WANDB_PROJECT,WANDB_RUN_NAME=$name \
        scripts/slurm_eval_lm_harness.sh
done

echo
echo "Submitted ${#RUNS[@]} eval jobs. Track with: squeue -u \$USER"
echo "Results will be logged to W&B project: $WANDB_PROJECT (one row per run)"
