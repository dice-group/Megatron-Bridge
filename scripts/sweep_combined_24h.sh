#!/bin/bash
#
# Combined 24h routing sweep — supersedes the three earlier sweep scripts
# (sweep_routing_small_24h.sh, sweep_kanneal_24h.sh, sweep_advanced_routing_24h.sh)
# which produced unevaluable checkpoints (save happened only at the final
# iteration boundary and was truncated by walltime → only common.pt and
# modelopt_run_config.yaml landed on disk, no distcp shards).
#
# Save fix lives in slurm_train_super_small_1gpu.sh:
#   - SAVE_INTERVAL now defaults to 2500 (≈4 saves per 24h run)
#   - checkpoint.async_save=False forced on the torchrun command
#
# Configs (16):
#   Small-routing baseline + EMA-threshold variants (4):
#     1. small_topk_24h               — topk reference baseline
#     2. small_et_24h                 — Expert Threshold (kth-largest EMA)
#     3. small_topp_24h               — Top-P confidence routing (p=0.5)
#     4. small_kanneal_24h            — sigmoid lossfree, K_target 4 → 2 anneal
#
#   K-anneal exploration (6):
#     5. kanneal_k1p5_24h             — K_end=1.5, rate=0.1, magnitude
#     6. kanneal_k1p25_24h            — K_end=1.25 (aggressive compute cut)
#     7. kanneal_slow_24h             — K_end=2.0, rate=0.05 (smoother)
#     8. kanneal_k1p5_slow_24h        — combine winners (K_end=1.5, rate=0.05)
#     9. kanneal_sign_24h             — sign-mode controller, rate=0.3
#    10. kanneal_aux_24h              — magnitude + AUX_LOSS_COEFF=0.01
#
#   Survey-derived dynamic-routing mechanisms (6):
#    11. advrouting_remoe_k2_24h      — ReMoE, target K=2
#    12. advrouting_remoe_k1p5_24h    — ReMoE, target K=1.5
#    13. advrouting_adamoe_m16_k3_24h — AdaMoE, m=16 nulls + top-3
#    14. advrouting_adamoe_m8_k2_24h  — AdaMoE, m=8 nulls + top-2
#    15. advrouting_dtopp_k2_24h      — DTopP, PI controller, target K=2
#    16. advrouting_dtopp_k1p5_24h    — DTopP, PI controller, target K=1.5
#
# Usage:
#   bash scripts/sweep_combined_24h.sh                # H100 (default)
#   GPU_TYPE=a100 bash scripts/sweep_combined_24h.sh  # A100
#   ONLY="kanneal_k1p5_24h advrouting_remoe_k2_24h" bash scripts/sweep_combined_24h.sh
#
# W&B project: variable-moe-routing (one run per submission).

set -euo pipefail

CKPT_BASE="${CKPT_BASE:-/scratch/hpc-prf-merlin/luke/Megatron-Bridge/sweep_ckpts_combined_24h}"
mkdir -p logs

GPU_TYPE="${GPU_TYPE:-h100}"

# Same compute budget as the prior three sweeps so val-loss curves overlay cleanly.
TRAIN_TOKENS=3000000000
if [ "$GPU_TYPE" = "a100" ]; then
    MICRO_BATCH_SIZE=16
    GRAD_ACCUM_STEPS=4
else
    MICRO_BATCH_SIZE=64
    GRAD_ACCUM_STEPS=1
fi

# Annealing schedule (forward-call units) for *_kanneal_* variants.
K_ANNEAL_START=4.0
K_ANNEAL_START_STEP=0
K_ANNEAL_END_STEP_DEFAULT=30000

# Optional whitelist: ONLY="name1 name2" filters submissions to those names.
ONLY="${ONLY:-}"

submit() {
    local name="$1"
    local routing_type="$2"
    shift 2

    if [ -n "$ONLY" ] && ! [[ " $ONLY " == *" $name "* ]]; then
        return 0
    fi

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

# ── Small-routing baselines (4) ────────────────────────────────────────────

submit small_topk_24h topk \
    THRESHOLD_UPDATE_MODE=magnitude \
    THRESHOLD_UPDATE_RATE=0 \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET_COEFF=0 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1

submit small_et_24h et \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET_COEFF=0 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1 \
    ET_EMA_BETA=0.99

submit small_topp_24h topp \
    AUX_LOSS_COEFF=0.01 \
    TOPANY_FORCE_TOP1=1 \
    TOPP_THRESHOLD=0.5 \
    TOPP_ENTROPY_COEFF=0.001

submit small_kanneal_24h sigmoid_lossfree_anneal \
    THRESHOLD_UPDATE_MODE=magnitude \
    THRESHOLD_UPDATE_RATE=0.1 \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1 \
    TOPANY_K_ANNEAL_START=$K_ANNEAL_START \
    TOPANY_K_ANNEAL_END=2.0 \
    TOPANY_K_ANNEAL_START_STEP=$K_ANNEAL_START_STEP \
    TOPANY_K_ANNEAL_END_STEP=$K_ANNEAL_END_STEP_DEFAULT

# ── K-anneal exploration (6) ───────────────────────────────────────────────
KAN_ROUTING=sigmoid_lossfree_anneal

submit kanneal_k1p5_24h $KAN_ROUTING \
    THRESHOLD_UPDATE_MODE=magnitude \
    THRESHOLD_UPDATE_RATE=0.1 \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1 \
    TOPANY_K_ANNEAL_START=$K_ANNEAL_START \
    TOPANY_K_ANNEAL_END=1.5 \
    TOPANY_K_ANNEAL_START_STEP=$K_ANNEAL_START_STEP \
    TOPANY_K_ANNEAL_END_STEP=$K_ANNEAL_END_STEP_DEFAULT

submit kanneal_k1p25_24h $KAN_ROUTING \
    THRESHOLD_UPDATE_MODE=magnitude \
    THRESHOLD_UPDATE_RATE=0.1 \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1 \
    TOPANY_K_ANNEAL_START=$K_ANNEAL_START \
    TOPANY_K_ANNEAL_END=1.25 \
    TOPANY_K_ANNEAL_START_STEP=$K_ANNEAL_START_STEP \
    TOPANY_K_ANNEAL_END_STEP=$K_ANNEAL_END_STEP_DEFAULT

submit kanneal_slow_24h $KAN_ROUTING \
    THRESHOLD_UPDATE_MODE=magnitude \
    THRESHOLD_UPDATE_RATE=0.05 \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1 \
    TOPANY_K_ANNEAL_START=$K_ANNEAL_START \
    TOPANY_K_ANNEAL_END=2.0 \
    TOPANY_K_ANNEAL_START_STEP=$K_ANNEAL_START_STEP \
    TOPANY_K_ANNEAL_END_STEP=$K_ANNEAL_END_STEP_DEFAULT

submit kanneal_k1p5_slow_24h $KAN_ROUTING \
    THRESHOLD_UPDATE_MODE=magnitude \
    THRESHOLD_UPDATE_RATE=0.05 \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1 \
    TOPANY_K_ANNEAL_START=$K_ANNEAL_START \
    TOPANY_K_ANNEAL_END=1.5 \
    TOPANY_K_ANNEAL_START_STEP=$K_ANNEAL_START_STEP \
    TOPANY_K_ANNEAL_END_STEP=$K_ANNEAL_END_STEP_DEFAULT

submit kanneal_sign_24h $KAN_ROUTING \
    THRESHOLD_UPDATE_MODE=sign \
    THRESHOLD_UPDATE_RATE=0.3 \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1 \
    TOPANY_K_ANNEAL_START=$K_ANNEAL_START \
    TOPANY_K_ANNEAL_END=2.0 \
    TOPANY_K_ANNEAL_START_STEP=$K_ANNEAL_START_STEP \
    TOPANY_K_ANNEAL_END_STEP=$K_ANNEAL_END_STEP_DEFAULT

submit kanneal_aux_24h $KAN_ROUTING \
    THRESHOLD_UPDATE_MODE=magnitude \
    THRESHOLD_UPDATE_RATE=0.1 \
    AUX_LOSS_COEFF=0.01 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1 \
    TOPANY_K_ANNEAL_START=$K_ANNEAL_START \
    TOPANY_K_ANNEAL_END=2.0 \
    TOPANY_K_ANNEAL_START_STEP=$K_ANNEAL_START_STEP \
    TOPANY_K_ANNEAL_END_STEP=$K_ANNEAL_END_STEP_DEFAULT

# ── Survey-derived dynamic routers: ReMoE / AdaMoE / DTopP (6) ─────────────

submit advrouting_remoe_k2_24h remoe \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET=2.0 \
    TOPANY_FORCE_TOP1=1 \
    REMOE_TARGET_K=2.0 \
    REMOE_LAMBDA_INIT=1e-4 \
    REMOE_LAMBDA_ALPHA=0.01 \
    REMOE_LAMBDA_MIN=1e-8 \
    REMOE_LAMBDA_MAX=1.0

submit advrouting_remoe_k1p5_24h remoe \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET=1.5 \
    TOPANY_FORCE_TOP1=1 \
    REMOE_TARGET_K=1.5 \
    REMOE_LAMBDA_INIT=1e-4 \
    REMOE_LAMBDA_ALPHA=0.01 \
    REMOE_LAMBDA_MIN=1e-8 \
    REMOE_LAMBDA_MAX=1.0

submit advrouting_adamoe_m16_k3_24h adamoe \
    AUX_LOSS_COEFF=0.01 \
    TOPANY_FORCE_TOP1=1 \
    ADAMOE_NUM_NULL=16 \
    ADAMOE_TOPK=3

submit advrouting_adamoe_m8_k2_24h adamoe \
    AUX_LOSS_COEFF=0.01 \
    TOPANY_FORCE_TOP1=1 \
    ADAMOE_NUM_NULL=8 \
    ADAMOE_TOPK=2

submit advrouting_dtopp_k2_24h dtopp \
    AUX_LOSS_COEFF=0.01 \
    TOPANY_FORCE_TOP1=1 \
    DTOPP_TARGET_K=2.0 \
    DTOPP_KP=0.05 \
    DTOPP_KI=0.005 \
    DTOPP_P_INIT=1.0 \
    TOPP_ENTROPY_COEFF=0

submit advrouting_dtopp_k1p5_24h dtopp \
    AUX_LOSS_COEFF=0.01 \
    TOPANY_FORCE_TOP1=1 \
    DTOPP_TARGET_K=1.5 \
    DTOPP_KP=0.05 \
    DTOPP_KI=0.005 \
    DTOPP_P_INIT=1.0 \
    TOPP_ENTROPY_COEFF=0

echo
echo "Sweep submitted. Track with: squeue -u \$USER"
echo "W&B project: variable-moe-routing"
echo "Checkpoints: $CKPT_BASE/<run_name>"
