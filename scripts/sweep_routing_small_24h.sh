#!/bin/bash
#
# Subset of sweep_routing_small.sh — three configs, 24h walltime instead of
# the script default. Picks the topk reference plus the two best-performing
# variable-K variants from the original sweep.
#
# Configs:
#   1. small_topk_24h           — topk reference baseline
#   2. small_topany_fb_24h      — cosine+threshold, sweep 2 winner (kt0p3)
#   3. small_sigmoid_kt_fb_24h  — sigmoid router + K-target 0.3 / K=2.5
#
# Usage:
#   bash scripts/sweep_routing_small_24h.sh

set -euo pipefail

CKPT_BASE=/scratch/hpc-prf-merlin/luke/Megatron-Bridge/sweep_ckpts_24h
mkdir -p logs

# Token budget chosen to roughly fill 22h on H100.
# WARNING: MBS=64 → FP32 logit buffer ~32 GiB (vocab=131072), 64 GiB for
# fwd+bwd. Likely OOMs on 96 GiB H100. GBS jumps 32 → 256 (8× larger).
TRAIN_TOKENS=3000000000
MICRO_BATCH_SIZE=64
GRAD_ACCUM_STEPS=4

# Format per row: NAME ROUTING TUMODE TURATE AUX_COEFF KTGT_COEFF KTGT_VALUE FORCE_TOP1
sweep=(
    "small_topk_24h           topk     magnitude 0     0     0     2.0   1"
    "small_topany_fb_24h      topany   sign      0     0.01  0.3   2.5   1"
    "small_sigmoid_kt_fb_24h  sigmoid  magnitude 0     0.01  0.3   2.5   1"
)

for entry in "${sweep[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry
    NAME=$1; ROUT=$2; TUMODE=$3; TURATE=$4; AUX=$5; KTGT=$6; KTGT_VAL=$7; FORCE_TOP1=$8

    echo "Submitting: $NAME (routing=$ROUT aux=$AUX ktgt_coeff=$KTGT ktgt_value=$KTGT_VAL force_top1=$FORCE_TOP1)"

    sbatch --gres=gpu:h100:1 \
           --exclusive \
           --time=24:00:00 \
           --export=ALL,\
RUN_NAME=$NAME,\
CHECKPOINT_DIR=$CKPT_BASE/$NAME,\
ROUTING_TYPE=$ROUT,\
THRESHOLD_UPDATE_MODE=$TUMODE,\
THRESHOLD_UPDATE_RATE=$TURATE,\
AUX_LOSS_COEFF=$AUX,\
TOPANY_K_TARGET_COEFF=$KTGT,\
TOPANY_K_TARGET=$KTGT_VAL,\
TOPANY_FORCE_TOP1=$FORCE_TOP1,\
TRAIN_TOKENS=$TRAIN_TOKENS,\
MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE,\
GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS \
        scripts/slurm_train_super_small_1gpu.sh
done

echo
echo "All ${#sweep[@]} jobs submitted (24h walltime). Track with: squeue -u \$USER"
echo "W&B project: variable-moe-routing"
