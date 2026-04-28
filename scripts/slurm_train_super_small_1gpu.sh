#!/bin/bash
#
# Small-scale single-GPU smoke test for routing comparison.
#
# Purpose: quickly validate whether lossfree top-any (with chosen rate) reaches
# parity with topk on a tiny model, before burning multi-GPU compute on the 1B
# version. Same routing code path, smaller model & token budget.
#
# Usage:
#   # Top-K baseline run (no routing-specific knobs needed)
#   sbatch --export=ALL,RUN_NAME=small_topk,CHECKPOINT_DIR=/scratch/.../small_topk,ROUTING_TYPE=topk \
#     scripts/slurm_train_super_small_1gpu.sh
#
#   # Lossfree run (uses tuned rate)
#   sbatch --export=ALL,RUN_NAME=small_lossfree_mag_r0p1,CHECKPOINT_DIR=/scratch/.../small_lossfree_mag_r0p1,ROUTING_TYPE=lossfree,THRESHOLD_UPDATE_MODE=magnitude,THRESHOLD_UPDATE_RATE=0.1 \
#     scripts/slurm_train_super_small_1gpu.sh
#
#SBATCH --job-name=moe-small-1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80GB
#SBATCH --account=hpc-prf-merlin
#SBATCH --output=logs/small_train_%j.out
#SBATCH --error=logs/small_train_%j.err

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR=/scratch/hpc-prf-merlin/luke/Megatron-Bridge/data/fineweb
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/hpc-prf-merlin/luke/Megatron-Bridge/sweep_ckpts/small_default}"
RUN_NAME="${RUN_NAME:-small_default}"

# Load WandB key from .env
if [ -f "$PWD/.env" ]; then
    export $(grep -v '^#' "$PWD/.env" | xargs)
fi

# Routing: "topk" (baseline), "topany", or "lossfree"
ROUTING_TYPE="${ROUTING_TYPE:-topk}"

# Lossfree-only knobs (ignored by topk)
THRESHOLD_UPDATE_MODE="${THRESHOLD_UPDATE_MODE:-magnitude}"
THRESHOLD_UPDATE_RATE="${THRESHOLD_UPDATE_RATE:-0.1}"

AUX_LOSS_COEFF="${AUX_LOSS_COEFF:-0}"

# Token budget for the smoke test — small enough to finish in ~2-4h on 1 GPU,
# big enough that routing dynamics differentiate the two variants.
TRAIN_TOKENS="${TRAIN_TOKENS:-100000000}"   # 100M tokens

# Single-GPU parallelism
N_GPUS=1
TP=1
EP=1
CP=1
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
SEQ_LENGTH=1024

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
    print(f'ERROR: TRAIN_TOKENS ({train_tokens:,}) exceeds epoch tokens ({epoch_tokens:,})', file=sys.stderr)
    sys.exit(1)

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

# Denser eval cadence so we get a good val-loss curve in a short run
EVAL_INTERVAL=200
SAVE_INTERVAL=10000   # effectively no checkpointing for a smoke test

mkdir -p logs
mkdir -p "$CHECKPOINT_DIR"

# Auto-detect GPU architecture
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
if [ "${GPU_CC:-0}" -ge 89 ]; then
    MIXED_PRECISION=bf16_with_fp8_current_scaling_mixed
    module load tools/Apptainer/1.3.5-GCCcore-13.3.0
else
    MIXED_PRECISION=bf16_mixed
    module load tools/Apptainer/1.3.4-GCCcore-13.3.0
    # A100 40GB: halve micro batch size, restore GBS via grad accum
    MICRO_BATCH_SIZE=$(( MICRO_BATCH_SIZE / 2 ))
    [ "$MICRO_BATCH_SIZE" -lt 1 ] && MICRO_BATCH_SIZE=1
    GRAD_ACCUM_STEPS=$(( GLOBAL_BATCH_SIZE / (DP * MICRO_BATCH_SIZE) ))
    echo "WARNING: GPU CC ${GPU_CC} < 89 — bf16_mixed, MBS=$MICRO_BATCH_SIZE GAS=$GRAD_ACCUM_STEPS"
fi

echo "=============================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Run name  : $RUN_NAME"
echo "Routing   : $ROUTING_TYPE (mode=$THRESHOLD_UPDATE_MODE rate=$THRESHOLD_UPDATE_RATE)"
echo "Tokens    : $EFFECTIVE_TOKENS / $EPOCH_TOKENS (epoch)"
echo "Iters     : $TRAIN_ITERS (warmup=$LR_WARMUP_ITERS, eval every $EVAL_INTERVAL)"
echo "GPUs      : $N_GPUS (DP=$DP)"
echo "Batch     : global=$GLOBAL_BATCH_SIZE micro=$MICRO_BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS"
echo "Model     : Nemotron-3 Super SMALL (7L, hidden=512, 32 experts, topk=2)"
echo "=============================="

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

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

        pip install plotly --quiet

        torchrun --nproc-per-node=$N_GPUS \
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
            model.moe_topany_target_k=2 \
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
echo "Job finished"
echo "=============================="
