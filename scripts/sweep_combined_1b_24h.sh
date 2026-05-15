#!/bin/bash
#
# 1B counterpart of sweep_combined_24h.sh — runs the most informative subset
# of the small-model sweep at the 1B Nemotron-3 Super scale.
#
# Configs (5):
#   1. topk_1b_24h               — topk=6 reference baseline (matches small_topk)
#   2. kanneal_aux_1b_24h        — sigmoid_lossfree_anneal, magnitude/0.1,
#                                  AUX_LOSS_COEFF=0.01, K_anneal 12 → 6
#   3. kanneal_sign_1b_24h       — sigmoid_lossfree_anneal, sign/0.3, K_anneal 12 → 6
#   4. adamoe_m64_k9_1b_24h      — AdaMoE, m=64 nulls + top-9 → E[K_real]=6 (baseline parity)
#   5. adamoe_m32_k6_1b_24h      — AdaMoE, m=32 nulls + top-6 → E[K_real]=4.8 (0.8× sparser)
#
# K_target/K_anneal are scaled to the 1B's topk=6 baseline (small used topk=2,
# so an anneal 4 → 2 there maps to 12 → 6 here — same factor of 2× → 1×
# relative to the static baseline). AdaMoE m/k are scaled so E[K_real] matches
# the small sweep's *ratios* against its topk=2 baseline (small m16_k3 →
# E[K_real]=2.0 = parity; small m8_k2 → E[K_real]=1.6 ≈ 0.8×). Keeping literal
# m=16,k=3 at the 1B scale would yield E[K_real]≈2.67 — far sparser than the
# topk=6 baseline, mixing router-quality and sparsity effects.
#
# Two-phase submission per config to avoid wasting a 24h slot on a run that
# can't produce loadable checkpoints (which is what happened with the
# original 24h sweep — async_save+walltime truncated saves to common.pt and
# modelopt_run_config.yaml only):
#
#   1. Sanity job (~45 min walltime, ~60 iters, SAVE_INTERVAL=20).
#      slurm_train_super_1b_1gpu.sh with SANITY_MODE=1 trains a few iters,
#      then asserts the latest iter_* dir has run_config.yaml + ≥1 .distcp
#      shard. Exits non-zero if not.
#   2. Main 24h job, submitted with --dependency=afterok:<sanity_id>.
#      Slurm only releases it if the sanity job exits 0. If sanity fails,
#      the main job is cancelled by --kill-on-invalid-dep=yes.
#
# Usage:
#   bash scripts/sweep_combined_1b_24h.sh                    # H100 (default)
#   GPU_TYPE=a100 bash scripts/sweep_combined_1b_24h.sh      # A100
#   ONLY="topk_1b_24h kanneal_aux_1b_24h" bash scripts/sweep_combined_1b_24h.sh
#   SKIP_SANITY=1 bash scripts/sweep_combined_1b_24h.sh      # skip sanity gate
#
# W&B project: variable-moe-routing.

set -euo pipefail

CKPT_BASE="${CKPT_BASE:-/scratch/hpc-prf-merlin/luke/Megatron-Bridge/sweep_ckpts_combined_1b_24h}"
mkdir -p logs

GPU_TYPE="${GPU_TYPE:-h100}"

TRAIN_TOKENS="${TRAIN_TOKENS:-10000000000}"   # 10B upper bound; 24h walltime
                                              # is the binding constraint

# 1B fits comfortably on a single H100; A100-40 needs the MBS halving the
# training script already applies internally.
if [ "$GPU_TYPE" = "h100" ]; then
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
    GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
else
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
    GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-32}"
fi

# Annealing schedule (forward-call units) for *_kanneal_* variants. Mirror
# the small sweep (start=4, end_step=30000) but scale K to the 1B baseline.
K_ANNEAL_START=12.0
K_ANNEAL_END=6.0
K_ANNEAL_START_STEP=0
K_ANNEAL_END_STEP=30000

# Sanity-mode walltime — short enough that a queue full of sanity jobs
# clears fast, long enough to save twice at SAVE_INTERVAL=20.
SANITY_TIME="${SANITY_TIME:-00:45:00}"
SANITY_TRAIN_TOKENS="${SANITY_TRAIN_TOKENS:-15000000}"
SANITY_SAVE_INTERVAL="${SANITY_SAVE_INTERVAL:-20}"

ONLY="${ONLY:-}"
SKIP_SANITY="${SKIP_SANITY:-0}"

TRAIN_SCRIPT="scripts/slurm_train_super_1b_1gpu.sh"

declare -a SUBMITTED
declare -a SUBMITTED_SANITY
declare -a SUBMITTED_MAIN

submit() {
    local name="$1"
    local routing_type="$2"
    shift 2

    if [ -n "$ONLY" ] && ! [[ " $ONLY " == *" $name "* ]]; then
        return 0
    fi

    # Build comma-separated env overrides for sbatch --export. Each remaining
    # arg is a KEY=VAL pair.
    local extra_env=""
    while [ $# -gt 0 ]; do
        extra_env+=",$1"
        shift
    done

    local ckpt_main="$CKPT_BASE/$name"
    local ckpt_sanity="$CKPT_BASE/.sanity/$name"
    mkdir -p "$ckpt_main" "$ckpt_sanity"

    local main_dep=""
    local sanity_id=""

    if [ "$SKIP_SANITY" != "1" ]; then
        echo "Submitting SANITY: $name (routing=$routing_type)"
        sanity_id=$(sbatch --parsable \
            --gres=gpu:${GPU_TYPE}:1 \
            --time=$SANITY_TIME \
            --job-name="sanity-$name" \
            --output="logs/sanity_${name}_%j.out" \
            --error="logs/sanity_${name}_%j.err" \
            --export=ALL,SANITY_MODE=1,RUN_NAME=sanity_${name},CHECKPOINT_DIR=$ckpt_sanity,ROUTING_TYPE=$routing_type,TRAIN_TOKENS=$SANITY_TRAIN_TOKENS,SAVE_INTERVAL=$SANITY_SAVE_INTERVAL,MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE,GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS$extra_env \
            $TRAIN_SCRIPT)
        SUBMITTED_SANITY+=("$sanity_id:$name")
        main_dep="--dependency=afterok:$sanity_id --kill-on-invalid-dep=yes"
    fi

    echo "Submitting MAIN  : $name (routing=$routing_type)${sanity_id:+  [after sanity $sanity_id]}"
    local main_id
    main_id=$(sbatch --parsable \
        --gres=gpu:${GPU_TYPE}:1 \
        --time=24:00:00 \
        --job-name="$name" \
        --output="logs/${name}_%j.out" \
        --error="logs/${name}_%j.err" \
        $main_dep \
        --export=ALL,SANITY_MODE=0,RUN_NAME=$name,CHECKPOINT_DIR=$ckpt_main,ROUTING_TYPE=$routing_type,TRAIN_TOKENS=$TRAIN_TOKENS,MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE,GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS$extra_env \
        $TRAIN_SCRIPT)
    SUBMITTED_MAIN+=("$main_id:$name")
}

# ── 1. topk reference baseline ────────────────────────────────────────────
submit topk_1b_24h topk \
    THRESHOLD_UPDATE_MODE=magnitude \
    THRESHOLD_UPDATE_RATE=0 \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET_COEFF=0 \
    TOPANY_K_TARGET=6.0 \
    TOPANY_FORCE_TOP1=1

# ── 2. K-anneal + AUX (kanneal_aux scaled) ────────────────────────────────
submit kanneal_aux_1b_24h sigmoid_lossfree_anneal \
    THRESHOLD_UPDATE_MODE=magnitude \
    THRESHOLD_UPDATE_RATE=0.1 \
    AUX_LOSS_COEFF=0.01 \
    TOPANY_K_TARGET=6.0 \
    TOPANY_FORCE_TOP1=1 \
    TOPANY_K_ANNEAL_START=$K_ANNEAL_START \
    TOPANY_K_ANNEAL_END=$K_ANNEAL_END \
    TOPANY_K_ANNEAL_START_STEP=$K_ANNEAL_START_STEP \
    TOPANY_K_ANNEAL_END_STEP=$K_ANNEAL_END_STEP

# ── 3. K-anneal sign-mode controller (kanneal_sign scaled) ────────────────
submit kanneal_sign_1b_24h sigmoid_lossfree_anneal \
    THRESHOLD_UPDATE_MODE=sign \
    THRESHOLD_UPDATE_RATE=0.3 \
    AUX_LOSS_COEFF=0 \
    TOPANY_K_TARGET=6.0 \
    TOPANY_FORCE_TOP1=1 \
    TOPANY_K_ANNEAL_START=$K_ANNEAL_START \
    TOPANY_K_ANNEAL_END=$K_ANNEAL_END \
    TOPANY_K_ANNEAL_START_STEP=$K_ANNEAL_START_STEP \
    TOPANY_K_ANNEAL_END_STEP=$K_ANNEAL_END_STEP

# ── 4. AdaMoE m=64 nulls, top-9 → E[K_real]=6 (parity with topk=6 baseline) ─
submit adamoe_m64_k9_1b_24h adamoe \
    AUX_LOSS_COEFF=0.01 \
    TOPANY_FORCE_TOP1=1 \
    ADAMOE_NUM_NULL=64 \
    ADAMOE_TOPK=9

# ── 5. AdaMoE m=32 nulls, top-6 → E[K_real]=4.8 (0.8× baseline, sparser cut) ─
submit adamoe_m32_k6_1b_24h adamoe \
    AUX_LOSS_COEFF=0.01 \
    TOPANY_FORCE_TOP1=1 \
    ADAMOE_NUM_NULL=32 \
    ADAMOE_TOPK=6

echo
echo "=============================="
echo "Sweep submitted."
if [ "$SKIP_SANITY" != "1" ]; then
    echo "Sanity jobs (verify checkpoints):"
    for entry in "${SUBMITTED_SANITY[@]}"; do
        printf "  %s  %s\n" "${entry%%:*}" "${entry#*:}"
    done
fi
echo "Main 24h jobs (released only after their sanity passes):"
for entry in "${SUBMITTED_MAIN[@]}"; do
    printf "  %s  %s\n" "${entry%%:*}" "${entry#*:}"
done
echo
echo "Track:        squeue -u \$USER"
echo "W&B project:  variable-moe-routing"
echo "Checkpoints:  $CKPT_BASE/<run_name>"
echo "Sanity ckpts: $CKPT_BASE/.sanity/<run_name>  (safe to delete after sanity passes)"
