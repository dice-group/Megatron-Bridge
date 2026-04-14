#!/bin/bash
#
# MoE routing sweep — Table 1 (36 runs: 12 configs × 3 models)
#
# Submits one 6-hour SLURM job per (model, routing, mode, rate) tuple.
# Each run gets a unique wandb exp name and checkpoint dir.
#
# Usage:
#   bash scripts/sweep_moe_routing.sh               # submit all
#   DRY_RUN=1 bash scripts/sweep_moe_routing.sh     # print sbatch lines only
#   MODELS=super bash scripts/sweep_moe_routing.sh  # subset: super|qwen3|gptoss
#   SWEEPS=1,2   bash scripts/sweep_moe_routing.sh  # subset: 1, 2, 3

set -euo pipefail

WALLTIME="${WALLTIME:-06:00:00}"
CKPT_ROOT="${CKPT_ROOT:-/scratch/hpc-prf-merlin/luke/Megatron-Bridge/sweep_ckpts}"
MODELS="${MODELS:-super,qwen3,gptoss}"
SWEEPS="${SWEEPS:-1,2,3}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$CKPT_ROOT" logs

# ── model → sbatch script ───────────────────────────────────────────────────
declare -A SCRIPT=(
    [super]=scripts/slurm_train_super_lossfree_1b.sh
    [qwen3]=scripts/slurm_train_qwen3_moe_1b.sh
    [gptoss]=scripts/slurm_train_gpt_oss_1b.sh
)

# ── sweep configs: "sweep|routing|mode|rate|coeff|tag" ──────────────────────
# routing: topk | topany | lossfree
# mode:    sign | magnitude (lossfree only; "-" = default)
# rate:    moe_topany_update_rate for lossfree; "-" = script default
# coeff:   moe_aux_loss_coeff  (topany aux-loss weight);  "-" = 0
CONFIGS=(
    # Sweep 1 — at default K per model (4 configs)
    "1|topk|-|-|-|topk"
    "1|topany|-|-|0.01|topany_c0p01"
    "1|lossfree|sign|0.001|-|lossfree_sign_r0p001"
    "1|lossfree|magnitude|0.0001|-|lossfree_mag_r0p0001"

    # Sweep 2 — lossfree rate/mode scan (5 configs)
    "2|lossfree|sign|0.0001|-|lossfree_sign_r0p0001"
    "2|lossfree|sign|0.01|-|lossfree_sign_r0p01"
    "2|lossfree|sign|0.1|-|lossfree_sign_r0p1"
    "2|lossfree|magnitude|0.001|-|lossfree_mag_r0p001"
    "2|lossfree|magnitude|0.01|-|lossfree_mag_r0p01"

    # Sweep 3 — topany aux-loss coeff scan (3 configs)
    "3|topany|-|-|0|topany_c0"
    "3|topany|-|-|0.001|topany_c0p001"
    "3|topany|-|-|0.1|topany_c0p1"
)

# ── helper: is value in comma-separated list ────────────────────────────────
contains() {
    local needle="$1" haystack="$2"
    [[ ",$haystack," == *",$needle,"* ]]
}

submit() {
    local model="$1" sweep="$2" routing="$3" mode="$4" rate="$5" coeff="$6" tag="$7"
    local script="${SCRIPT[$model]}"
    local run_name="${model}_sw${sweep}_${tag}"
    local ckpt_dir="$CKPT_ROOT/$run_name"

    # Build --export list: only override knobs that are not "-" so the
    # per-script defaults apply otherwise.
    local exports="ALL,RUN_NAME=$run_name,CHECKPOINT_DIR=$ckpt_dir,ROUTING_TYPE=$routing"
    [[ "$mode"  != "-" ]] && exports="$exports,THRESHOLD_UPDATE_MODE=$mode"
    [[ "$rate"  != "-" ]] && exports="$exports,THRESHOLD_UPDATE_RATE=$rate"
    [[ "$coeff" != "-" ]] && exports="$exports,AUX_LOSS_COEFF=$coeff"

    local cmd=(
        sbatch
        --time="$WALLTIME"
        --job-name="$run_name"
        --output="logs/${run_name}_%j.out"
        --error="logs/${run_name}_%j.err"
        --export="$exports"
        "$script"
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[dry-run] '
        printf '%q ' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}"
    fi
}

# ── enumerate & submit ──────────────────────────────────────────────────────
count=0
for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r sweep routing mode rate coeff tag <<< "$cfg"
    contains "$sweep" "$SWEEPS" || continue

    for model in super qwen3 gptoss; do
        contains "$model" "$MODELS" || continue
        submit "$model" "$sweep" "$routing" "$mode" "$rate" "$coeff" "$tag"
        count=$((count + 1))
    done
done

echo ""
echo "Submitted/planned $count jobs (walltime=$WALLTIME each)."
