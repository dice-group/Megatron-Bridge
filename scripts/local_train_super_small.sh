#!/bin/bash
#
# Non-sbatch sibling of slurm_train_super_small_1gpu.sh, used by the
# agent_loop driver. Same apptainer + torchrun launch as the slurm
# version, but without SBATCH headers / module load — caller is
# responsible for resource allocation and for killing the run via
# `timeout` when the iteration budget is up.
#
# Usage (driver-managed):
#   timeout 20m bash scripts/local_train_super_small.sh \
#       2>&1 | tee path/to/train.log
#
# Same env vars as the slurm version (RUN_NAME, CHECKPOINT_DIR,
# ROUTING_TYPE, AUX_LOSS_COEFF, TOPANY_K_TARGET, TOPANY_K_TARGET_COEFF,
# THRESHOLD_UPDATE_MODE, THRESHOLD_UPDATE_RATE, MICRO_BATCH_SIZE,
# GRAD_ACCUM_STEPS, TRAIN_TOKENS, EVAL_INTERVAL, DATA_DIR, CONTAINER).

set -euo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR="${DATA_DIR:-/scratch/hpc-prf-merlin/luke/Megatron-Bridge/data/fineweb}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/hpc-prf-merlin/luke/Megatron-Bridge/agent_loop_ckpts/scratch}"
RUN_NAME="${RUN_NAME:-agent_loop_local}"

# Load WandB key from .env if present
if [ -f "$PWD/.env" ]; then
    export $(grep -v '^#' "$PWD/.env" | xargs)
fi

ROUTING_TYPE="${ROUTING_TYPE:-topany}"

THRESHOLD_UPDATE_MODE="${THRESHOLD_UPDATE_MODE:-sign}"
THRESHOLD_UPDATE_RATE="${THRESHOLD_UPDATE_RATE:-0}"
AUX_LOSS_COEFF="${AUX_LOSS_COEFF:-0.01}"
TOPANY_K_TARGET="${TOPANY_K_TARGET:-2.5}"
TOPANY_K_TARGET_COEFF="${TOPANY_K_TARGET_COEFF:-0.3}"
TOPANY_FORCE_TOP1="${TOPANY_FORCE_TOP1:-1}"

# Token budget intentionally large; the driver bounds wall-clock with `timeout`.
TRAIN_TOKENS="${TRAIN_TOKENS:-2000000000}"

N_GPUS="${N_GPUS:-1}"
TP=1
EP=1
CP=1
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
SEQ_LENGTH=1024

DP=$(( N_GPUS / (TP * EP * CP) ))
GLOBAL_BATCH_SIZE=$(( DP * MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS ))

# Frequent eval so a 20-min run produces ~6-10 val_loss readings.
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
SAVE_INTERVAL=1000000   # effectively no checkpointing

CONTAINER="${CONTAINER:-$PWD/../nemo-container}"
BLEND_PATH="$DATA_DIR/blend.json"

# ==============================================================================

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

if train_tokens == 0 or train_tokens > epoch_tokens:
    train_tokens = epoch_tokens

tokens_per_iter = gbs * seq_length
train_iters     = train_tokens // tokens_per_iter
warmup_iters    = max(50, train_iters // 10)

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
else
    MIXED_PRECISION=bf16_mixed
    MICRO_BATCH_SIZE=$(( MICRO_BATCH_SIZE / 2 ))
    [ "$MICRO_BATCH_SIZE" -lt 1 ] && MICRO_BATCH_SIZE=1
    GRAD_ACCUM_STEPS=$(( GLOBAL_BATCH_SIZE / (DP * MICRO_BATCH_SIZE) ))
    echo "WARNING: GPU CC ${GPU_CC} < 89 — bf16_mixed, MBS=$MICRO_BATCH_SIZE GAS=$GRAD_ACCUM_STEPS"
fi

echo "=============================="
echo "Run name  : $RUN_NAME"
echo "Routing   : $ROUTING_TYPE"
echo "K-target  : value=$TOPANY_K_TARGET coeff=$TOPANY_K_TARGET_COEFF aux=$AUX_LOSS_COEFF"
echo "Tokens    : $EFFECTIVE_TOKENS / $EPOCH_TOKENS (epoch)"
echo "Iters     : $TRAIN_ITERS (warmup=$LR_WARMUP_ITERS, eval every $EVAL_INTERVAL)"
echo "GPUs      : $N_GPUS (DP=$DP)"
echo "Batch     : global=$GLOBAL_BATCH_SIZE micro=$MICRO_BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS"
echo "=============================="

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# Unique torchrun port per process to avoid collisions if two iterations
# overlap (driver should serialize, but cheap to be defensive).
MASTER_PORT=$((20000 + $$ % 30000))
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
        export TOPANY_FORCE_TOP1=$TOPANY_FORCE_TOP1
        export TOPANY_K_TARGET=$TOPANY_K_TARGET
        export TOPANY_K_TARGET_COEFF=$TOPANY_K_TARGET_COEFF
        export RUN_NAME=$RUN_NAME

        pip install plotly --quiet

        torchrun --nproc-per-node=$N_GPUS --master-port=$MASTER_PORT \
            examples/models/nemotron_3/pretrain_nemotron_3_super.py \
            --per-split-data-args-path=$BLEND_PATH \
            logger.wandb_project=variable-moe-routing \
            logger.wandb_entity=lukefriedrichs-paderborn-university \
            logger.wandb_exp_name=$RUN_NAME \
            logger.log_interval=10 \
            model.moe_per_layer_logging=True \
            dataset.num_workers=2 \
            dataset.mmap_bin_files=True \
            mixed_precision=$MIXED_PRECISION \
            train.global_batch_size=$GLOBAL_BATCH_SIZE \
            train.micro_batch_size=$MICRO_BATCH_SIZE \
            train.train_iters=$TRAIN_ITERS \
            train.eval_interval=$EVAL_INTERVAL \
            scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
            model.num_layers=7 \
            model.hybrid_override_pattern=\"MEM*EME\" \
            model.hidden_size=512 \
            model.ffn_hidden_size=384 \
            model.num_moe_experts=32 \
            model.moe_ffn_hidden_size=384 \
            model.moe_shared_expert_intermediate_size=768 \
            model.moe_latent_size=160 \
            model.moe_router_topk=2 \
            model.num_attention_heads=8 \
            model.num_query_groups=1 \
            model.mamba_num_heads=16 \
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
            checkpoint.save_interval=$SAVE_INTERVAL
    "

echo "=============================="
echo "Run finished"
echo "=============================="
