#!/bin/bash
#
# Nemotron-3 Super 1B — single GPU, unified routing.
#
# Sister of slurm_train_super_small_1gpu.sh (small model, all routing types)
# scaled up to the 1B architecture from slurm_train_super_topk_1b_1gpu.sh /
# slurm_train_super_topany_1b_1gpu.sh.
#
# Supports every routing_type wired through gate.py:
#   topk, et, topp, sigmoid_lossfree_anneal, remoe, adamoe, dtopp, topany, lossfree
# Routing-specific knobs are read from env vars (see gate.py):
#   TOPANY_*, ET_EMA_BETA, TOPP_*, REMOE_*, ADAMOE_*, DTOPP_*
#
# Two modes (controlled by SANITY_MODE):
#   SANITY_MODE=0 (default): full 24h run, SAVE_INTERVAL=2500
#   SANITY_MODE=1:           short run to verify checkpoints save correctly.
#                            After training, asserts that the latest iter_*
#                            dir contains run_config.yaml + at least one
#                            .distcp shard. Exits non-zero if not — so a
#                            chained sbatch --dependency=afterok will skip
#                            the main job. Catches the combined-sweep bug
#                            where async_save+walltime truncated saves to
#                            just common.pt + modelopt_run_config.yaml.
#
# Async save is forced off (the original combined sweep hit a save
# truncation bug; SAVE_INTERVAL=2500 + async_save=False is the fix).
#
# Usage (single run):
#   sbatch --export=ALL,RUN_NAME=...,CHECKPOINT_DIR=...,ROUTING_TYPE=topk \
#       scripts/slurm_train_super_1b_1gpu.sh
#
#   # sanity smoke (20-30 min):
#   sbatch --time=00:45:00 --export=ALL,RUN_NAME=...,CHECKPOINT_DIR=...,ROUTING_TYPE=topk,SANITY_MODE=1 \
#       scripts/slurm_train_super_1b_1gpu.sh

#SBATCH --job-name=moe-1b-1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128GB
#SBATCH --account=hpc-prf-merlin
#SBATCH --output=logs/1b_%j.out
#SBATCH --error=logs/1b_%j.err

set -uo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR=/scratch/hpc-prf-merlin/luke/Megatron-Bridge/data/fineweb
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/hpc-prf-merlin/luke/Megatron-Bridge/checkpoints_super_1b_1gpu}"
RUN_NAME="${RUN_NAME:-super_1b_1gpu}"

if [ -f "$PWD/.env" ]; then
    export $(grep -v '^#' "$PWD/.env" | xargs)
fi

# Routing (defaults match topk baseline)
ROUTING_TYPE="${ROUTING_TYPE:-topk}"
AUX_LOSS_COEFF="${AUX_LOSS_COEFF:-0}"
THRESHOLD_UPDATE_MODE="${THRESHOLD_UPDATE_MODE:-magnitude}"
THRESHOLD_UPDATE_RATE="${THRESHOLD_UPDATE_RATE:-0}"

# Routing-type env knobs (all read by gate.py via os.environ). Re-export
# explicitly so they propagate into the apptainer container regardless of
# how it inherits env from the slurm shell.
export TOPANY_K_TARGET="${TOPANY_K_TARGET:-6.0}"
export TOPANY_K_TARGET_COEFF="${TOPANY_K_TARGET_COEFF:-0}"
export TOPANY_FORCE_TOP1="${TOPANY_FORCE_TOP1:-1}"
export TOPANY_K_ANNEAL_START="${TOPANY_K_ANNEAL_START:-$TOPANY_K_TARGET}"
export TOPANY_K_ANNEAL_END="${TOPANY_K_ANNEAL_END:-$TOPANY_K_TARGET}"
export TOPANY_K_ANNEAL_START_STEP="${TOPANY_K_ANNEAL_START_STEP:-0}"
export TOPANY_K_ANNEAL_END_STEP="${TOPANY_K_ANNEAL_END_STEP:-0}"
export ET_EMA_BETA="${ET_EMA_BETA:-0.99}"
export TOPP_THRESHOLD="${TOPP_THRESHOLD:-0.5}"
export TOPP_ENTROPY_COEFF="${TOPP_ENTROPY_COEFF:-0}"
export REMOE_TARGET_K="${REMOE_TARGET_K:-$TOPANY_K_TARGET}"
export REMOE_LAMBDA_INIT="${REMOE_LAMBDA_INIT:-1e-4}"
export REMOE_LAMBDA_ALPHA="${REMOE_LAMBDA_ALPHA:-0.01}"
export REMOE_LAMBDA_MIN="${REMOE_LAMBDA_MIN:-1e-8}"
export REMOE_LAMBDA_MAX="${REMOE_LAMBDA_MAX:-1.0}"
export ADAMOE_NUM_NULL="${ADAMOE_NUM_NULL:-16}"
export ADAMOE_TOPK="${ADAMOE_TOPK:-3}"
export DTOPP_TARGET_K="${DTOPP_TARGET_K:-$TOPANY_K_TARGET}"
export DTOPP_KP="${DTOPP_KP:-0.05}"
export DTOPP_KI="${DTOPP_KI:-0.005}"
export DTOPP_P_INIT="${DTOPP_P_INIT:-1.0}"

# Sanity vs full run.
SANITY_MODE="${SANITY_MODE:-0}"
if [ "$SANITY_MODE" = "1" ]; then
    # Small enough to finish in well under the 30-45 min sanity walltime,
    # but several saves so verification has something to check even if the
    # job is killed mid-training.
    TRAIN_TOKENS="${TRAIN_TOKENS:-15000000}"   # 15M tokens (≈60 iters)
    SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
else
    TRAIN_TOKENS="${TRAIN_TOKENS:-10000000000}"   # 10B upper bound; 24h walltime
                                                  # is the binding constraint
    SAVE_INTERVAL="${SAVE_INTERVAL:-2500}"
fi

N_GPUS=1
TP=1
EP=1
CP=1
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
SEQ_LENGTH=2048

DP=$(( N_GPUS / (TP * EP * CP) ))
GLOBAL_BATCH_SIZE=$(( DP * MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS ))

# ==============================================================================

CONTAINER="$PWD/../nemo-container"
BLEND_PATH="$DATA_DIR/blend.json"

_ITER_CALC_ERR=$(mktemp)
read TRAIN_ITERS LR_WARMUP_ITERS EPOCH_TOKENS EFFECTIVE_TOKENS <<< $(python3 -c "
import numpy as np, struct, sys

def count_tokens(idx_path):
    with open(idx_path, 'rb') as f:
        f.read(18)
        seq_count = struct.unpack('<Q', f.read(8))[0]
        f.read(8)
        lengths = np.frombuffer(f.read(seq_count * 4), dtype=np.int32)
        return int(np.sum(lengths))

epoch_tokens = count_tokens('${DATA_DIR}/fineweb_train_text_document.idx')
train_tokens = ${TRAIN_TOKENS}
seq_length   = ${SEQ_LENGTH}
gbs          = ${GLOBAL_BATCH_SIZE}

if train_tokens == 0:
    train_tokens = epoch_tokens
elif train_tokens > epoch_tokens:
    train_tokens = epoch_tokens

tokens_per_iter = gbs * seq_length
train_iters     = max(1, train_tokens // tokens_per_iter)
warmup_iters    = max(10, train_iters // 10)

print(train_iters, warmup_iters, epoch_tokens, train_tokens)
" 2>"$_ITER_CALC_ERR")

if [ -z "$TRAIN_ITERS" ] || ! [[ "$TRAIN_ITERS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Failed to compute TRAIN_ITERS (got '$TRAIN_ITERS')."
    cat "$_ITER_CALC_ERR"
    rm -f "$_ITER_CALC_ERR"
    exit 1
fi
rm -f "$_ITER_CALC_ERR"

mkdir -p logs
mkdir -p "$CHECKPOINT_DIR"

GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
if [ "${GPU_CC:-0}" -ge 89 ]; then
    MIXED_PRECISION=bf16_with_fp8_current_scaling_mixed
    module load tools/Apptainer/1.3.5-GCCcore-13.3.0
else
    MIXED_PRECISION=bf16_mixed
    module load tools/Apptainer/1.3.4-GCCcore-13.3.0
    MICRO_BATCH_SIZE=$(( MICRO_BATCH_SIZE / 4 ))
    [ "$MICRO_BATCH_SIZE" -lt 1 ] && MICRO_BATCH_SIZE=1
    GRAD_ACCUM_STEPS=$(( GLOBAL_BATCH_SIZE / (DP * MICRO_BATCH_SIZE) ))
    echo "WARNING: GPU CC ${GPU_CC} < 89 — bf16_mixed, MBS=$MICRO_BATCH_SIZE GAS=$GRAD_ACCUM_STEPS"
fi

echo "=============================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Run name  : $RUN_NAME"
echo "Mode      : $([ "$SANITY_MODE" = "1" ] && echo "SANITY (verify ckpt only)" || echo "FULL 24h")"
echo "Routing   : $ROUTING_TYPE (aux=$AUX_LOSS_COEFF mode=$THRESHOLD_UPDATE_MODE rate=$THRESHOLD_UPDATE_RATE)"
echo "Top-any   : k_target=$TOPANY_K_TARGET coeff=$TOPANY_K_TARGET_COEFF force_top1=$TOPANY_FORCE_TOP1"
echo "K-anneal  : [$TOPANY_K_ANNEAL_START → $TOPANY_K_ANNEAL_END] over steps [$TOPANY_K_ANNEAL_START_STEP, $TOPANY_K_ANNEAL_END_STEP]"
echo "AdaMoE    : num_null=$ADAMOE_NUM_NULL topk_aug=$ADAMOE_TOPK"
echo "Tokens    : $EFFECTIVE_TOKENS / $EPOCH_TOKENS (epoch)"
echo "Iters     : $TRAIN_ITERS (warmup=$LR_WARMUP_ITERS, save_interval=$SAVE_INTERVAL)"
echo "GPUs      : $N_GPUS (DP=$DP)"
echo "Batch     : global=$GLOBAL_BATCH_SIZE micro=$MICRO_BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS"
echo "Model     : Nemotron-3 Super 1B (17L, hidden=1280, 128 experts, topk=6 baseline)"
echo "=============================="

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

MASTER_PORT=$((20000 + SLURM_JOB_ID % 30000))
echo "Master port: $MASTER_PORT"

apptainer exec \
    --nv \
    --no-home \
    --bind "$PWD":/opt/Megatron-Bridge \
    --bind "$DATA_DIR":"$DATA_DIR" \
    --bind "$CHECKPOINT_DIR":"$CHECKPOINT_DIR" \
    --pwd /opt/Megatron-Bridge \
    "$CONTAINER" \
    bash -c "
        export HOME=/tmp
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export TOPANY_K_TARGET=$TOPANY_K_TARGET
        export TOPANY_K_TARGET_COEFF=$TOPANY_K_TARGET_COEFF
        export TOPANY_FORCE_TOP1=$TOPANY_FORCE_TOP1
        export TOPANY_K_ANNEAL_START=$TOPANY_K_ANNEAL_START
        export TOPANY_K_ANNEAL_END=$TOPANY_K_ANNEAL_END
        export TOPANY_K_ANNEAL_START_STEP=$TOPANY_K_ANNEAL_START_STEP
        export TOPANY_K_ANNEAL_END_STEP=$TOPANY_K_ANNEAL_END_STEP
        export ET_EMA_BETA=$ET_EMA_BETA
        export TOPP_THRESHOLD=$TOPP_THRESHOLD
        export TOPP_ENTROPY_COEFF=$TOPP_ENTROPY_COEFF
        export REMOE_TARGET_K=$REMOE_TARGET_K
        export REMOE_LAMBDA_INIT=$REMOE_LAMBDA_INIT
        export REMOE_LAMBDA_ALPHA=$REMOE_LAMBDA_ALPHA
        export REMOE_LAMBDA_MIN=$REMOE_LAMBDA_MIN
        export REMOE_LAMBDA_MAX=$REMOE_LAMBDA_MAX
        export ADAMOE_NUM_NULL=$ADAMOE_NUM_NULL
        export ADAMOE_TOPK=$ADAMOE_TOPK
        export DTOPP_TARGET_K=$DTOPP_TARGET_K
        export DTOPP_KP=$DTOPP_KP
        export DTOPP_KI=$DTOPP_KI
        export DTOPP_P_INIT=$DTOPP_P_INIT

        pip install plotly --quiet

        torchrun --nproc-per-node=$N_GPUS --master-port=$MASTER_PORT \
            examples/models/nemotron_3/pretrain_nemotron_3_super.py \
            --per-split-data-args-path=$BLEND_PATH \
            logger.wandb_project=variable-moe-routing \
            logger.wandb_entity=lukefriedrichs-paderborn-university \
            logger.wandb_exp_name=$RUN_NAME \
            logger.log_interval=10 \
            model.moe_per_layer_logging=True \
            dataset.num_workers=4 \
            dataset.mmap_bin_files=True \
            mixed_precision=$MIXED_PRECISION \
            train.global_batch_size=$GLOBAL_BATCH_SIZE \
            train.micro_batch_size=$MICRO_BATCH_SIZE \
            train.train_iters=$TRAIN_ITERS \
            scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
            model.num_layers=17 \
            model.hybrid_override_pattern=\"MEMEMEM*EMEMEM*EME\" \
            model.hidden_size=1280 \
            model.ffn_hidden_size=832 \
            model.num_moe_experts=128 \
            model.moe_ffn_hidden_size=832 \
            model.moe_shared_expert_intermediate_size=1664 \
            model.moe_latent_size=320 \
            model.moe_router_topk=6 \
            model.num_attention_heads=10 \
            model.num_query_groups=1 \
            model.mamba_num_heads=32 \
            model.mamba_state_dim=32 \
            model.mamba_num_groups=2 \
            model.routing_type=$ROUTING_TYPE \
            model.moe_aux_loss_coeff=$AUX_LOSS_COEFF \
            model.moe_topany_target_k=$TOPANY_K_TARGET \
            model.moe_topany_update_rate=$THRESHOLD_UPDATE_RATE \
            model.moe_topany_threshold_update_mode=$THRESHOLD_UPDATE_MODE \
            model.tensor_model_parallel_size=$TP \
            model.expert_model_parallel_size=$EP \
            model.sequence_parallel=False \
            model.context_parallel_size=$CP \
            model.seq_length=$SEQ_LENGTH \
            dataset.sequence_length=$SEQ_LENGTH \
            checkpoint.save=$CHECKPOINT_DIR \
            checkpoint.save_interval=$SAVE_INTERVAL \
            checkpoint.async_save=False
    "
TRAIN_RC=$?

echo "=============================="
echo "torchrun exited with $TRAIN_RC"

# Sanity-mode post-flight: verify a usable checkpoint actually landed on disk.
# This catches the combined-sweep bug (async_save + walltime → only common.pt
# and modelopt_run_config.yaml saved, no distcp shards). Returning non-zero
# here lets a chained sbatch --dependency=afterok skip the 24h main job.
if [ "$SANITY_MODE" = "1" ]; then
    echo "--- Sanity-mode checkpoint verification ---"
    latest=$(ls -d "$CHECKPOINT_DIR"/iter_* 2>/dev/null | sort -V | tail -1 || true)
    if [ -z "$latest" ]; then
        echo "FAIL: no iter_* dirs under $CHECKPOINT_DIR" >&2
        exit 2
    fi
    echo "Latest iter dir: $latest"
    ls -la "$latest" || true
    if [ ! -f "$latest/run_config.yaml" ]; then
        echo "FAIL: $latest/run_config.yaml missing — eval sweep would skip this run" >&2
        exit 3
    fi
    distcp_count=$(find "$latest" -maxdepth 1 -name "*.distcp" -type f 2>/dev/null | wc -l)
    if [ "$distcp_count" -lt 1 ]; then
        echo "FAIL: $latest has $distcp_count .distcp shards — save was truncated" >&2
        exit 4
    fi
    echo "OK: $latest has run_config.yaml + $distcp_count .distcp shard(s)"
    echo "Sanity check passed for $RUN_NAME."
fi

echo "Job finished"
echo "=============================="
exit $TRAIN_RC
