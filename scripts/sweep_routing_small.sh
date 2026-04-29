#!/bin/bash
#
# Sweep 2 — focused on the K-drift question from sweep 1.
#
# Sweep 1 verdict: topany family is structurally fine (thresholds bounded, val
# loss within 0.011 of topk), but K drifted to ~1.6 instead of staying at the
# target=2.0. The K-target loss with coeff=0.1 was too weak; the LM gradient
# pulled K downward. This sweep tests whether stronger K-target settings can
# anchor K at 2.0, and whether closing the K gap also closes the val-loss gap.
#
# Variants:
#   1. topk                  — fresh baseline (apples-to-apples for this sweep)
#   2. topany_kt1_lb01       — ktgt=1.0, aux=0.01, target_K=2.0  (10× stronger pull)
#   3. topany_kt5_lb01       — ktgt=5.0, aux=0.01, target_K=2.0  (50× stronger; near-hard constraint)
#   4. topany_kt0p3_K2p5     — ktgt=0.3, aux=0.01, target_K=2.5  (overshoot init to land at 2)
#   5. topany_kt1_lb03       — ktgt=1.0, aux=0.03, target_K=2.0  (stronger balance + ktgt)
#
# Win condition: a variant lands K≈2.0 stably AND its val loss is ≤ topk's.
#
# Usage:
#   bash scripts/sweep_routing_small.sh

set -euo pipefail

CKPT_BASE=/scratch/hpc-prf-merlin/luke/Megatron-Bridge/sweep_ckpts
mkdir -p logs

# Format per row: NAME ROUTING TUMODE TURATE AUX_COEFF KTGT_COEFF KTGT_VALUE
# TUMODE/TURATE only matter for ROUTING=lossfree (none in this sweep).
sweep=(
    "small_topk                topk    magnitude 0  0     0    2.0"
    "small_topany_kt1_lb01     topany  sign      0  0.01  1.0  2.0"
    "small_topany_kt5_lb01     topany  sign      0  0.01  5.0  2.0"
    "small_topany_kt0p3_K2p5   topany  sign      0  0.01  0.3  2.5"
    "small_topany_kt1_lb03     topany  sign      0  0.03  1.0  2.0"
)

for entry in "${sweep[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry
    NAME=$1; ROUT=$2; TUMODE=$3; TURATE=$4; AUX=$5; KTGT=$6; KTGT_VAL=$7

    echo "Submitting: $NAME (routing=$ROUT aux=$AUX ktgt_coeff=$KTGT ktgt_value=$KTGT_VAL)"

    sbatch --gres=gpu:a100:1 \
           --export=ALL,\
RUN_NAME=$NAME,\
CHECKPOINT_DIR=$CKPT_BASE/$NAME,\
ROUTING_TYPE=$ROUT,\
THRESHOLD_UPDATE_MODE=$TUMODE,\
THRESHOLD_UPDATE_RATE=$TURATE,\
AUX_LOSS_COEFF=$AUX,\
TOPANY_K_TARGET_COEFF=$KTGT,\
TOPANY_K_TARGET=$KTGT_VAL \
        scripts/slurm_train_super_small_1gpu.sh
done

echo
echo "All ${#sweep[@]} jobs submitted. Track with: squeue -u \$USER"
echo "W&B project: variable-moe-routing"
echo "After completion: bash scripts/consolidate_sweep_csv.sh"
