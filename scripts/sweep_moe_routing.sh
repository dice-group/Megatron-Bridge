#!/bin/bash
#
# MoE routing sweep (36 runs: 12 configs × 3 models)
#
# Submits one 6-hour SLURM job per (model, routing, mode, rate) tuple.
# Each run gets a unique wandb exp name and checkpoint dir.
#
# Usage:
#   bash scripts/sweep_moe_routing.sh               # submit all
#   DRY_RUN=1 bash scripts/sweep_moe_routing.sh     # print sbatch lines only
#   MODELS=super bash scripts/sweep_moe_routing.sh  # subset: super|qwen3|gptoss

set -euo pipefail

WALLTIME="${WALLTIME:-06:00:00}"
CKPT_ROOT="${CKPT_ROOT:-/scratch/hpc-prf-merlin/luke/Megatron-Bridge/sweep_ckpts}"
MODELS="${MODELS:-super,qwen3,gptoss}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$CKPT_ROOT" logs

# ── model → sbatch script ───────────────────────────────────────────────────
declare -A SCRIPT=(
    [super]=scripts/slurm_train_super_lossfree_1b.sh
    [qwen3]=scripts/slurm_train_qwen3_moe_1b.sh
    [gptoss]=scripts/slurm_train_gpt_oss_1b.sh
)

# ── sweep configs: "routing|mode|rate|coeff|tag" ────────────────────────────
# routing: topk | topany | lossfree
# mode:    sign | magnitude (lossfree only; "-" = default)
# rate:    moe_topany_update_rate for lossfree; "-" = script default
# coeff:   moe_aux_loss_coeff  (topany aux-loss weight);  "-" = 0
#
# topany runs OOM at the per-script default MBS, so we halve MBS and double
# GAS for them (see submit()) — preserves global batch size.
CONFIGS=(
    "topk|-|-|-|topk"
    "topany|-|-|0.01|topany_c0p01"
    "lossfree|sign|0.001|-|lossfree_sign_r0p001"
    "lossfree|magnitude|0.0001|-|lossfree_mag_r0p0001"
    "lossfree|sign|0.0001|-|lossfree_sign_r0p0001"
    "lossfree|sign|0.01|-|lossfree_sign_r0p01"
    "lossfree|sign|0.1|-|lossfree_sign_r0p1"
    "lossfree|magnitude|0.001|-|lossfree_mag_r0p001"
    "lossfree|magnitude|0.01|-|lossfree_mag_r0p01"
    "topany|-|-|0|topany_c0"
    "topany|-|-|0.001|topany_c0p001"
    "topany|-|-|0.1|topany_c0p1"
)

# Per-model default (MBS, GAS) — must match the slurm script defaults.
# Used to derive halved MBS / doubled GAS for topany runs.
declare -A DEFAULT_MBS=( [super]=8  [qwen3]=16 [gptoss]=8 )
declare -A DEFAULT_GAS=( [super]=4  [qwen3]=2  [gptoss]=4 )

# ── helper: is value in comma-separated list ────────────────────────────────
contains() {
    local needle="$1" haystack="$2"
    [[ ",$haystack," == *",$needle,"* ]]
}

submit() {
    local model="$1" routing="$2" mode="$3" rate="$4" coeff="$5" tag="$6"
    local script="${SCRIPT[$model]}"
    local run_name="${model}_${tag}"
    local ckpt_dir="$CKPT_ROOT/$run_name"

    # Build --export list: only override knobs that are not "-" so the
    # per-script defaults apply otherwise.
    local exports="ALL,RUN_NAME=$run_name,CHECKPOINT_DIR=$ckpt_dir,ROUTING_TYPE=$routing"
    [[ "$mode"  != "-" ]] && exports="$exports,THRESHOLD_UPDATE_MODE=$mode"
    [[ "$rate"  != "-" ]] && exports="$exports,THRESHOLD_UPDATE_RATE=$rate"
    [[ "$coeff" != "-" ]] && exports="$exports,AUX_LOSS_COEFF=$coeff"

    # topany OOMs at default MBS — halve MBS, double GAS (keeps GBS constant).
    if [[ "$routing" == "topany" ]]; then
        local mbs=$(( ${DEFAULT_MBS[$model]} / 2 ))
        local gas=$(( ${DEFAULT_GAS[$model]} * 2 ))
        exports="$exports,MICRO_BATCH_SIZE=$mbs,GRAD_ACCUM_STEPS=$gas"
    fi

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
    IFS='|' read -r routing mode rate coeff tag <<< "$cfg"

    for model in super qwen3 gptoss; do
        contains "$model" "$MODELS" || continue
        submit "$model" "$routing" "$mode" "$rate" "$coeff" "$tag"
        count=$((count + 1))
    done
done

echo ""
echo "Submitted/planned $count jobs (walltime=$WALLTIME each)."
