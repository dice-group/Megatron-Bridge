#!/bin/bash
#
# Sweep over routing strategies on the small smoke-test config (1 H100, ~2-4h
# per job). Each variant tests a different way of controlling K (experts per
# token) and load balance.
#
# Variants:
#   1. topk                     — fixed-K baseline (reference for everything)
#   2. lossfree-sign-r0.001     — original loss-free idea, sign mode + slow rate
#                                 (deltas bounded ±0.001, no oscillation risk)
#   3. lossfree-mag-r0.03       — magnitude mode, lower rate, with clamps from
#                                 the previous round (pegged at +9 last time;
#                                 want to see if slower lets the controller
#                                 keep up with sim_matrix drift)
#   4. topany-aux-loadbalance   — gradient-trained threshold + standard load-
#                                 balance aux loss only. K is unconstrained
#                                 (free to drift).
#   5. topany-aux-ktarget       — gradient-trained threshold + K-target aux
#                                 loss only.  coeff·(K̄ − 2)². Anchors K
#                                 directly via gradient.
#   6. topany-aux-both          — load-balance + K-target. Belt-and-suspenders.
#
# Usage:
#   bash scripts/sweep_routing_small.sh
#
# Each job lands in its own checkpoint dir + W&B run name.

set -euo pipefail

CKPT_BASE=/scratch/hpc-prf-merlin/luke/Megatron-Bridge/sweep_ckpts
mkdir -p logs

# Format per row: NAME ROUTING TUMODE TURATE AUX_COEFF KTGT_COEFF
# (TUMODE/TURATE only matter for ROUTING=lossfree; AUX/KTGT only for topany.)
sweep=(
    "small_topk                  topk     magnitude 0      0      0"
    "small_lf_sign_r0p001        lossfree sign      0.001  0      0"
    "small_lf_mag_r0p03          lossfree magnitude 0.03   0      0"
    "small_topany_aux_lb_0p01    topany   sign      0      0.01   0"
    "small_topany_aux_ktgt_0p1   topany   sign      0      0      0.1"
    "small_topany_aux_both       topany   sign      0      0.01   0.1"
)

for entry in "${sweep[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry
    NAME=$1; ROUT=$2; TUMODE=$3; TURATE=$4; AUX=$5; KTGT=$6

    echo "Submitting: $NAME (routing=$ROUT mode=$TUMODE rate=$TURATE aux=$AUX ktgt=$KTGT)"

    sbatch --gres=gpu:a100:1 \
           --export=ALL,\
RUN_NAME=$NAME,\
CHECKPOINT_DIR=$CKPT_BASE/$NAME,\
ROUTING_TYPE=$ROUT,\
THRESHOLD_UPDATE_MODE=$TUMODE,\
THRESHOLD_UPDATE_RATE=$TURATE,\
AUX_LOSS_COEFF=$AUX,\
TOPANY_K_TARGET_COEFF=$KTGT,\
TOPANY_K_TARGET=2.0 \
        scripts/slurm_train_super_small_1gpu.sh
done

echo
echo "All ${#sweep[@]} jobs submitted. Track with: squeue -u \$USER"
echo "W&B project: variable-moe-routing"
