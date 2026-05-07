#!/bin/bash
#
# Survey-derived dynamic-routing sweep. Three new mechanisms from the
# 2024-26 MoE-routing literature, none of which we've tested yet:
#
#   1. ReMoE  (Wang et al. 2024, arXiv:2412.14711)
#       Per-expert ReLU gate (fully differentiable, no STE) + adaptive L1
#       sparsity controller. Paper claims consistent wins over TopK across
#       182M–978M active sizes — the strongest literature claim of beating
#       TopK on equal compute.
#
#   2. AdaMoE (Zeng et al. 2024, arXiv:2406.13233)
#       Standard topk + softmax + renormalize, but expert pool augmented
#       with `m` null experts that always output 0. Effective real-K varies
#       per token. Diagnostic: keeps the topk machinery that's been beating
#       us, just adds variable K. If AdaMoE matches/beats topk → variable-K
#       is the win and we know which mechanism. If AdaMoE also loses → the
#       value isn't variable-K at this scale.
#
#   3. DTopP  (arXiv:2512.13996)
#       Top-p with the threshold p adapted by a PI controller targeting
#       average K. Fixes both failure modes we saw with static p=0.5
#       (initial collapse to K=1, eventual runaway to K=9).
#
# Compare against existing small_topk_24h baseline (W&B project
# `variable-moe-routing`); no need to re-run topk.
#
# Usage:
#   bash scripts/sweep_advanced_routing_24h.sh                # H100
#   GPU_TYPE=a100 bash scripts/sweep_advanced_routing_24h.sh  # A100

set -euo pipefail

CKPT_BASE=/scratch/hpc-prf-merlin/luke/Megatron-Bridge/sweep_ckpts_advanced_24h
mkdir -p logs

GPU_TYPE="${GPU_TYPE:-h100}"

# Same compute budget as the prior sweeps so val-loss curves overlay cleanly.
TRAIN_TOKENS=3000000000
if [ "$GPU_TYPE" = "a100" ]; then
    MICRO_BATCH_SIZE=16
    GRAD_ACCUM_STEPS=4
else
    MICRO_BATCH_SIZE=64
    GRAD_ACCUM_STEPS=1
fi

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
           --time=24:00:00 \
           --export=ALL,RUN_NAME=$name,CHECKPOINT_DIR=$CKPT_BASE/$name,ROUTING_TYPE=$routing_type,TRAIN_TOKENS=$TRAIN_TOKENS,MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE,GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS$extra_env \
        scripts/slurm_train_super_small_1gpu.sh
}

# ── 1. ReMoE — target K=2, default L1 controller ───────────────────────────
# Paper-default-ish settings. λ adapts each step toward target K.
submit advrouting_remoe_k2_24h remoe \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1 \
    REMOE_TARGET_K=2.0 \
    REMOE_LAMBDA_INIT=1e-4 \
    REMOE_LAMBDA_ALPHA=0.01 \
    REMOE_LAMBDA_MIN=1e-8 \
    REMOE_LAMBDA_MAX=1.0

# ── 2. ReMoE — aggressive K=1.5 (compute below topk) ───────────────────────
submit advrouting_remoe_k1p5_24h remoe \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET=1.5 \
    TOPANY_FORCE_TOP1=1 \
    REMOE_TARGET_K=1.5 \
    REMOE_LAMBDA_INIT=1e-4 \
    REMOE_LAMBDA_ALPHA=0.01 \
    REMOE_LAMBDA_MIN=1e-8 \
    REMOE_LAMBDA_MAX=1.0

# ── 3. AdaMoE — m=16 nulls, top-3 → expected real-K = 2.0 ──────────────────
# Closest match to topk's K=2: 32 real + 16 null = 48; top-3 over 48 yields
# expected 2 real + 1 null, average real-K = 2.0.
# Already running / completed; uncomment to re-submit.
# submit advrouting_adamoe_m16_k3_24h adamoe \
#     AUX_LOSS_COEFF=0.01 \
#     TOPANY_FORCE_TOP1=1 \
#     ADAMOE_NUM_NULL=16 \
#     ADAMOE_TOPK=3

# ── 4. AdaMoE — m=8 nulls, top-2 → expected real-K = 1.6 ───────────────────
# Compute-reduced variant: more aggressive null usage, lower average real-K.
# Already running / completed; uncomment to re-submit.
# submit advrouting_adamoe_m8_k2_24h adamoe \
#     AUX_LOSS_COEFF=0.01 \
#     TOPANY_FORCE_TOP1=1 \
#     ADAMOE_NUM_NULL=8 \
#     ADAMOE_TOPK=2

# ── 5. DTopP — PI controller targeting K=2 ─────────────────────────────────
# p_init=1.0 (well above the K=1 collapse regime); KP/KI tuned conservative.
# Already running / completed; uncomment to re-submit.
# submit advrouting_dtopp_k2_24h dtopp \
#     AUX_LOSS_COEFF=0.01 \
#     TOPANY_FORCE_TOP1=1 \
#     DTOPP_TARGET_K=2.0 \
#     DTOPP_KP=0.05 \
#     DTOPP_KI=0.005 \
#     DTOPP_P_INIT=1.0 \
#     TOPP_ENTROPY_COEFF=0

# ── 6. DTopP — PI controller targeting K=1.5 ───────────────────────────────
# Already running / completed; uncomment to re-submit.
# submit advrouting_dtopp_k1p5_24h dtopp \
#     AUX_LOSS_COEFF=0.01 \
#     TOPANY_FORCE_TOP1=1 \
#     DTOPP_TARGET_K=1.5 \
#     DTOPP_KP=0.05 \
#     DTOPP_KI=0.005 \
#     DTOPP_P_INIT=1.0 \
#     TOPP_ENTROPY_COEFF=0

echo
echo "Re-submitting only the 2 ReMoE variants (others completed/running)."
echo "Track with: squeue -u \$USER"
echo "W&B project: variable-moe-routing (compare against small_topk_24h)"
