#!/bin/bash
#
# 24h routing sweep — baseline + new dynamic-routing variants from the
# variable-K research review. Goal: find a router that beats topk on val loss.
#
# Configs (active):
#   1. small_topk_24h               — topk reference baseline
#   4. small_et_24h                 — Expert Threshold (kth-largest EMA)
#   5. small_topp_24h               — Top-P confidence routing (p=0.5)
#   6. small_kanneal_24h            — sigmoid lossfree with K_target 4 → 2 cosine anneal
#
# Disabled (already run previously, results in W&B):
#   2. small_topany_fb_24h          — cosine+threshold, prior sweep winner (kt0p3)
#   3. small_sigmoid_kt_fb_24h      — sigmoid router + K-target 0.3 / K=2.5
#
# Usage:
#   bash scripts/sweep_routing_small_24h.sh

set -euo pipefail

CKPT_BASE=/scratch/hpc-prf-merlin/luke/Megatron-Bridge/sweep_ckpts_24h
mkdir -p logs

# GPU type for --gres. Override with: GPU_TYPE=a100 bash scripts/sweep_routing_small_24h.sh
GPU_TYPE="${GPU_TYPE:-h100}"

# Token budget chosen to roughly fill 22h on H100.
# WARNING: MBS=64 → FP32 logit buffer ~32 GiB (vocab=131072), 64 GiB for
# fwd+bwd. Likely OOMs on 96 GiB H100. GBS jumps 32 → 256 (8× larger).
TRAIN_TOKENS=3000000000
MICRO_BATCH_SIZE=64
GRAD_ACCUM_STEPS=4

# Annealing schedule for sigmoid_lossfree_anneal:
#   ~9.2k optimizer steps total at GBS=256, seq=1024, 3B tokens →
#   ~36.6k forward calls (×4 grad accum). Anneal across 0..30k forwards
#   ≈ 0..82% of training; final 18% holds at K=2.
K_ANNEAL_START=4.0
K_ANNEAL_END=2.0
K_ANNEAL_START_STEP=0
K_ANNEAL_END_STEP=30000

submit() {
    local name="$1"
    local routing_type="$2"
    shift 2

    local extra_env=""
    while [ $# -gt 0 ]; do
        extra_env+=",$1"
        shift
    done

    echo "Submitting: $name (routing=$routing_type)"

    sbatch --gres=gpu:${GPU_TYPE}:1 \
           --exclusive \
           --time=24:00:00 \
           --export=ALL,RUN_NAME=$name,CHECKPOINT_DIR=$CKPT_BASE/$name,ROUTING_TYPE=$routing_type,TRAIN_TOKENS=$TRAIN_TOKENS,MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE,GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS$extra_env \
        scripts/slurm_train_super_small_1gpu.sh
}

# ── 1. topk baseline ────────────────────────────────────────────────────────
submit small_topk_24h topk \
    THRESHOLD_UPDATE_MODE=magnitude \
    THRESHOLD_UPDATE_RATE=0 \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET_COEFF=0 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1

# ── 2. topany (cosine + learned thresholds), prior winner ──────────────────
# Already run; results in W&B. Re-enable by uncommenting.
# submit small_topany_fb_24h topany \
#     THRESHOLD_UPDATE_MODE=sign \
#     THRESHOLD_UPDATE_RATE=0 \
#     AUX_LOSS_COEFF=0.01 \
#     TOPANY_K_TARGET_COEFF=0.3 \
#     TOPANY_K_TARGET=2.5 \
#     TOPANY_FORCE_TOP1=1

# ── 3. sigmoid + K-target loss ─────────────────────────────────────────────
# Already run; results in W&B. Re-enable by uncommenting.
# submit small_sigmoid_kt_fb_24h sigmoid \
#     THRESHOLD_UPDATE_MODE=magnitude \
#     THRESHOLD_UPDATE_RATE=0 \
#     AUX_LOSS_COEFF=0.01 \
#     TOPANY_K_TARGET_COEFF=0.3 \
#     TOPANY_K_TARGET=2.5 \
#     TOPANY_FORCE_TOP1=1

# ── 4. Expert Threshold (kth-largest EMA) ──────────────────────────────────
# No aux loss, no K-target loss — balance comes from the EMA threshold itself.
submit small_et_24h et \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET_COEFF=0 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1 \
    ET_EMA_BETA=0.99

# ── 5. Top-P confidence routing ────────────────────────────────────────────
# p=0.5: a confident expert with σ≥0.5 alone covers the budget; ambiguous
# tokens pull in more experts naturally. Light entropy-min nudges toward
# decisive distributions over time.
submit small_topp_24h topp \
    AUX_LOSS_COEFF=0.01 \
    TOPANY_FORCE_TOP1=1 \
    TOPP_THRESHOLD=0.5 \
    TOPP_ENTROPY_COEFF=0.001

# ── 6. K-target annealing on sigmoid lossfree ──────────────────────────────
# Start at K=4, anneal to K=2 over the first 82% of training. Lets early
# layers see broader expert mixtures before tightening to the production K.
submit small_kanneal_24h sigmoid_lossfree_anneal \
    THRESHOLD_UPDATE_MODE=magnitude \
    THRESHOLD_UPDATE_RATE=0.1 \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1 \
    TOPANY_K_ANNEAL_START=$K_ANNEAL_START \
    TOPANY_K_ANNEAL_END=$K_ANNEAL_END \
    TOPANY_K_ANNEAL_START_STEP=$K_ANNEAL_START_STEP \
    TOPANY_K_ANNEAL_END_STEP=$K_ANNEAL_END_STEP

echo
echo "All 4 jobs submitted (24h walltime). Track with: squeue -u \$USER"
echo "W&B project: variable-moe-routing"
