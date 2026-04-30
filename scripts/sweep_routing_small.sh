#!/bin/bash
#
# Sweep 3 — sigmoid-linear router + fallback ablation.
#
# Sweep 2 verdict: stronger K-target HURT val loss (kt5 worst, kt0p3 best).
# Forcing K=2.0 overrides a useful per-token K signal in the LM gradient.
# Conclusion: tweaking coefficients in the cosine+threshold parameterization
# is exhausted — the gating logic itself needs a cleaner alternative.
#
# Sweep 3 introduces two new router classes:
#   - SigmoidGateRouter:        Linear(d, E) → σ → STE(>0.5).  No threshold;
#                               cutoff fixed at 0.5. Aux loss + optional
#                               K-target via env vars.
#   - LossFreeSigmoidRouter:    Same forward + per-expert bias buffer updated
#                               outside autograd to balance load.
#
# Plus a fallback ablation across both parameterizations:
#   TOPANY_FORCE_TOP1=1 (default): tokens with K=0 fall back to top-1.
#   TOPANY_FORCE_TOP1=0:           K=0 tokens skip MoE (residual passthrough).
#
# 9 jobs:
#   1. topk                         — reference baseline
#   2. topany_fb       (kt0p3)      — sweep 2 winner, unchanged (fallback ON)
#   3. topany_nofb     (kt0p3)      — same config, fallback OFF (isolates fallback effect)
#   4. sigmoid_fb                   — new sigmoid router, no K-target, fallback ON
#   5. sigmoid_nofb                 — same, fallback OFF
#   6. sigmoid_kt_fb                — sigmoid + K-target=0.3 K=2.5 (mirror cosine winner)
#   7. sigmoid_kt_nofb              — same, fallback OFF
#   8. sigmoid_lf_fb                — loss-free sigmoid (bias update), fallback ON
#   9. sigmoid_lf_nofb              — same, fallback OFF
#
# What each comparison answers:
#   2 vs 3                  → fallback effect on cosine
#   4 vs 6                  → does K-target help the sigmoid router?
#   4 vs 8                  → aux loss vs loss-free in sigmoid
#   4 vs 5, 6 vs 7, 8 vs 9  → fallback effect across sigmoid variants
#   2 vs 6                  → cosine vs sigmoid (matched K-target config)
#   topk vs all             → reference
#
# Win: any variable-K variant ≤ topk val loss at step 3051.
#
# Usage:
#   bash scripts/sweep_routing_small.sh

set -euo pipefail

CKPT_BASE=/scratch/hpc-prf-merlin/luke/Megatron-Bridge/sweep_ckpts
mkdir -p logs

# Format per row: NAME ROUTING TUMODE TURATE AUX_COEFF KTGT_COEFF KTGT_VALUE FORCE_TOP1
# TUMODE/TURATE are only used by *lossfree variants (otherwise inert).
# KTGT_COEFF=0 disables the K-target loss for topany/sigmoid.
sweep=(
    "small_topk                topk             magnitude 0     0     0     2.0   1"
    "small_topany_fb           topany           sign      0     0.01  0.3   2.5   1"
    "small_topany_nofb         topany           sign      0     0.01  0.3   2.5   0"
    "small_sigmoid_fb          sigmoid          magnitude 0     0.01  0     2.0   1"
    "small_sigmoid_nofb        sigmoid          magnitude 0     0.01  0     2.0   0"
    "small_sigmoid_kt_fb       sigmoid          magnitude 0     0.01  0.3   2.5   1"
    "small_sigmoid_kt_nofb     sigmoid          magnitude 0     0.01  0.3   2.5   0"
    "small_sigmoid_lf_fb       sigmoid_lossfree sign      0.01  0     0     2.0   1"
    "small_sigmoid_lf_nofb     sigmoid_lossfree sign      0.01  0     0     2.0   0"
)

for entry in "${sweep[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry
    NAME=$1; ROUT=$2; TUMODE=$3; TURATE=$4; AUX=$5; KTGT=$6; KTGT_VAL=$7; FORCE_TOP1=$8

    echo "Submitting: $NAME (routing=$ROUT aux=$AUX ktgt_coeff=$KTGT ktgt_value=$KTGT_VAL force_top1=$FORCE_TOP1)"

    sbatch --gres=gpu:a100:1 \
           --export=ALL,\
RUN_NAME=$NAME,\
CHECKPOINT_DIR=$CKPT_BASE/$NAME,\
ROUTING_TYPE=$ROUT,\
THRESHOLD_UPDATE_MODE=$TUMODE,\
THRESHOLD_UPDATE_RATE=$TURATE,\
AUX_LOSS_COEFF=$AUX,\
TOPANY_K_TARGET_COEFF=$KTGT,\
TOPANY_K_TARGET=$KTGT_VAL,\
TOPANY_FORCE_TOP1=$FORCE_TOP1 \
        scripts/slurm_train_super_small_1gpu.sh
done

echo
echo "All ${#sweep[@]} jobs submitted. Track with: squeue -u \$USER"
echo "W&B project: variable-moe-routing"
echo "After completion: bash scripts/consolidate_sweep_csv.sh"
