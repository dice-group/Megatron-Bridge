#!/bin/bash
#
# Follow-up sweep focused on K-annealed sigmoid lossfree routing — kanneal was
# the only variant from the prior sweep that tracked topk val-loss with stable
# compute (K=2). Goal: explore the design space around it to find a config that
# beats topk on val loss at equal-or-lower compute.
#
# Compare against the existing small_topk_24h run (already in W&B project
# `variable-moe-routing`); no need to re-run topk here.
#
# Knobs explored:
#   - K_end:       lower target post-anneal (1.5, 1.25) → fewer experts/token
#   - update_rate: smoother bias controller (0.05) → less routing churn
#   - update_mode: sign-mode controller as alternative to magnitude
#   - aux loss:    add load-balance gradient to anchor against logit drift
#   - schedule:    shorter anneal window so K_end is reached earlier
#
# Configs (6):
#   1. kanneal_k1p5_24h          — K_end=1.5, rate=0.1, magnitude (compute win)
#   2. kanneal_k1p25_24h         — K_end=1.25 (aggressive compute reduction)
#   3. kanneal_slow_24h          — K_end=2, rate=0.05 (smoother controller)
#   4. kanneal_k1p5_slow_24h     — K_end=1.5 + rate=0.05 (combine winners)
#   5. kanneal_sign_24h          — sign-mode controller, rate=0.3
#   6. kanneal_aux_24h           — magnitude + AUX_LOSS_COEFF=0.01 (anti-drift)
#
# Usage:
#   bash scripts/sweep_kanneal_24h.sh                # H100 (default)
#   GPU_TYPE=a100 bash scripts/sweep_kanneal_24h.sh  # A100 path

set -euo pipefail

CKPT_BASE=/scratch/hpc-prf-merlin/luke/Megatron-Bridge/sweep_ckpts_kanneal_24h
mkdir -p logs

# GPU type for --gres. Override with: GPU_TYPE=a100 bash scripts/sweep_kanneal_24h.sh
GPU_TYPE="${GPU_TYPE:-h100}"

# Match the routing sweep exactly so val-loss curves are directly comparable
# against existing small_topk_24h baseline (same project, same hyperparameters
# except routing).
TRAIN_TOKENS=3000000000
if [ "$GPU_TYPE" = "a100" ]; then
    MICRO_BATCH_SIZE=16
    GRAD_ACCUM_STEPS=4
else
    MICRO_BATCH_SIZE=64
    GRAD_ACCUM_STEPS=1
fi

# Annealing schedule (forward-call units). Default window 0..30k matches the
# original kanneal run; schedule_short variants finish earlier.
K_ANNEAL_START=4.0
K_ANNEAL_START_STEP=0
K_ANNEAL_END_STEP_DEFAULT=30000
K_ANNEAL_END_STEP_SHORT=10000

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

    # No --exclusive: 1 GPU per job, packs cleanly on shared nodes.
    sbatch --gres=gpu:${GPU_TYPE}:1 \
           --time=24:00:00 \
           --export=ALL,RUN_NAME=$name,CHECKPOINT_DIR=$CKPT_BASE/$name,ROUTING_TYPE=$routing_type,TRAIN_TOKENS=$TRAIN_TOKENS,MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE,GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS$extra_env \
        scripts/slurm_train_super_small_1gpu.sh
}

ROUTING=sigmoid_lossfree_anneal

# ── 1. K_end=1.5 (compute win attempt) ─────────────────────────────────────
submit kanneal_k1p5_24h $ROUTING \
    THRESHOLD_UPDATE_MODE=magnitude \
    THRESHOLD_UPDATE_RATE=0.1 \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1 \
    TOPANY_K_ANNEAL_START=$K_ANNEAL_START \
    TOPANY_K_ANNEAL_END=1.5 \
    TOPANY_K_ANNEAL_START_STEP=$K_ANNEAL_START_STEP \
    TOPANY_K_ANNEAL_END_STEP=$K_ANNEAL_END_STEP_DEFAULT

# ── 2. K_end=1.25 (aggressive compute reduction) ──────────────────────────
submit kanneal_k1p25_24h $ROUTING \
    THRESHOLD_UPDATE_MODE=magnitude \
    THRESHOLD_UPDATE_RATE=0.1 \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1 \
    TOPANY_K_ANNEAL_START=$K_ANNEAL_START \
    TOPANY_K_ANNEAL_END=1.25 \
    TOPANY_K_ANNEAL_START_STEP=$K_ANNEAL_START_STEP \
    TOPANY_K_ANNEAL_END_STEP=$K_ANNEAL_END_STEP_DEFAULT

# ── 3. Slow controller (rate=0.05, K_end=2) ────────────────────────────────
submit kanneal_slow_24h $ROUTING \
    THRESHOLD_UPDATE_MODE=magnitude \
    THRESHOLD_UPDATE_RATE=0.05 \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1 \
    TOPANY_K_ANNEAL_START=$K_ANNEAL_START \
    TOPANY_K_ANNEAL_END=2.0 \
    TOPANY_K_ANNEAL_START_STEP=$K_ANNEAL_START_STEP \
    TOPANY_K_ANNEAL_END_STEP=$K_ANNEAL_END_STEP_DEFAULT

# ── 4. K_end=1.5 + slow controller (combine the two best bets) ─────────────
submit kanneal_k1p5_slow_24h $ROUTING \
    THRESHOLD_UPDATE_MODE=magnitude \
    THRESHOLD_UPDATE_RATE=0.05 \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1 \
    TOPANY_K_ANNEAL_START=$K_ANNEAL_START \
    TOPANY_K_ANNEAL_END=1.5 \
    TOPANY_K_ANNEAL_START_STEP=$K_ANNEAL_START_STEP \
    TOPANY_K_ANNEAL_END_STEP=$K_ANNEAL_END_STEP_DEFAULT

# ── 5. Sign-mode controller (alternative to magnitude) ─────────────────────
submit kanneal_sign_24h $ROUTING \
    THRESHOLD_UPDATE_MODE=sign \
    THRESHOLD_UPDATE_RATE=0.3 \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1 \
    TOPANY_K_ANNEAL_START=$K_ANNEAL_START \
    TOPANY_K_ANNEAL_END=2.0 \
    TOPANY_K_ANNEAL_START_STEP=$K_ANNEAL_START_STEP \
    TOPANY_K_ANNEAL_END_STEP=$K_ANNEAL_END_STEP_DEFAULT

# ── 6. With aux load-balance loss (anti-drift anchor) ──────────────────────
# Adds gradient pressure on imbalanced expert counts so the LM gradient can't
# silently push the linear logits up to compensate for negative bias.
submit kanneal_aux_24h $ROUTING \
    THRESHOLD_UPDATE_MODE=magnitude \
    THRESHOLD_UPDATE_RATE=0.1 \
    AUX_LOSS_COEFF=0.01 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1 \
    TOPANY_K_ANNEAL_START=$K_ANNEAL_START \
    TOPANY_K_ANNEAL_END=2.0 \
    TOPANY_K_ANNEAL_START_STEP=$K_ANNEAL_START_STEP \
    TOPANY_K_ANNEAL_END_STEP=$K_ANNEAL_END_STEP_DEFAULT

echo
echo "All 6 kanneal variants submitted (24h walltime each). Track with: squeue -u \$USER"
echo "W&B project: variable-moe-routing (compare against small_topk_24h)"
