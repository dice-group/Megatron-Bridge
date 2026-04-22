#!/bin/bash
#
# Parallelism presets
# ─────────────────────────────────────────────────────────────────────────────
# 4-GPU DP=4 (recommended — best throughput for 1B, no EP comm overhead):
#   N_GPUS=4  TP=1  EP=1  CP=1  MBS=16  GAS=2   →  GBS=128
#   #SBATCH --gres=gpu:h100:4
#
# 4-GPU EP=4 (original — expert parallelism, higher all-to-all overhead):
#   N_GPUS=4  TP=1  EP=4  CP=1  MBS=8   GAS=16  →  GBS=128
#   #SBATCH --gres=gpu:h100:4
#
# 1-GPU:
#   N_GPUS=1  TP=1  EP=1  CP=1  MBS=8   GAS=16  →  GBS=128
#   #SBATCH --gres=gpu:h100:1  --mem=128GB
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=qwen3-moe-1b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256GB
#SBATCH --account=hpc-prf-merlin
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR=/scratch/hpc-prf-merlin/luke/Megatron-Bridge/data/fineweb
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/hpc-prf-merlin/luke/Megatron-Bridge/checkpoints_qwen3_moe_1b}"
RUN_NAME="${RUN_NAME:-qwen3_moe_1b}"

# Load WandB key from .env file
if [ -f "$PWD/.env" ]; then
    export $(grep -v '^#' "$PWD/.env" | xargs)
fi

# Routing type: "topany", "lossfree", or "topk"
ROUTING_TYPE="${ROUTING_TYPE:-lossfree}"

# Threshold update mode: "sign" or "magnitude" (only applies to lossfree routing)
THRESHOLD_UPDATE_MODE="${THRESHOLD_UPDATE_MODE:-magnitude}"
THRESHOLD_UPDATE_RATE="${THRESHOLD_UPDATE_RATE:-0.0001}"


# Auxiliary load-balance loss weight (primarily used with topany routing)
AUX_LOSS_COEFF="${AUX_LOSS_COEFF:-0}"

# Token budget: 0 = train for one full epoch over the training split (default).
# Set to a positive integer to train on exactly that many tokens (must be ≤ epoch tokens).
TRAIN_TOKENS="${TRAIN_TOKENS:-0}"

# Parallelism — single node, 4 GPUs, DP=4, EP=1
N_GPUS=4
TP=1
EP=1
CP=1
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
SEQ_LENGTH=2048

# Derived
DP=$(( N_GPUS / (TP * EP * CP) ))
GLOBAL_BATCH_SIZE=$(( DP * MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS ))

# ==============================================================================
# Derived paths — do not edit
# ==============================================================================

CONTAINER="$PWD/../nemo-container"
BLEND_PATH="$DATA_DIR/blend.json"

# Compute train_iters from the tokenized .idx file so we do exactly 1 epoch
# (or the user-specified token budget).
_ITER_CALC_ERR=$(mktemp)
read TRAIN_ITERS LR_WARMUP_ITERS EPOCH_TOKENS EFFECTIVE_TOKENS <<< $(python3 -c "
import numpy as np, struct, sys

def count_tokens(idx_path):
    with open(idx_path, 'rb') as f:
        f.read(18)  # magic(9) + version(8) + dtype(1)
        seq_count = struct.unpack('<Q', f.read(8))[0]
        f.read(8)   # doc_count
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
warmup_iters    = train_iters // 10

print(train_iters, warmup_iters, epoch_tokens, train_tokens)
" 2>"$_ITER_CALC_ERR")

if [ -z "$TRAIN_ITERS" ] || ! [[ "$TRAIN_ITERS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Failed to compute TRAIN_ITERS (got '$TRAIN_ITERS')."
    echo "Python stderr:" && cat "$_ITER_CALC_ERR"
    rm -f "$_ITER_CALC_ERR"
    exit 1
fi
rm -f "$_ITER_CALC_ERR"

SAVE_INTERVAL=5000

# ==============================================================================

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
    # A100 40GB: halve micro batch size, double grad accum to preserve GBS
    MICRO_BATCH_SIZE=$(( MICRO_BATCH_SIZE / 2 ))
    [ "$MICRO_BATCH_SIZE" -lt 1 ] && MICRO_BATCH_SIZE=1
    GRAD_ACCUM_STEPS=$(( GLOBAL_BATCH_SIZE / (DP * MICRO_BATCH_SIZE) ))
    echo "WARNING: GPU compute capability ${GPU_CC} < 89 — using bf16_mixed, MBS=$MICRO_BATCH_SIZE"
fi

echo "=============================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Container : $CONTAINER"
echo "Data dir  : $DATA_DIR"
echo "Blend     : $BLEND_PATH"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Routing   : $ROUTING_TYPE"
echo "Tokens    : $EFFECTIVE_TOKENS / $EPOCH_TOKENS (epoch)"
echo "Iters     : $TRAIN_ITERS (warmup=$LR_WARMUP_ITERS)"
echo "GPUs      : $N_GPUS (DP=$DP, TP=$TP, EP=$EP, CP=$CP)"
echo "Batch     : global=$GLOBAL_BATCH_SIZE micro=$MICRO_BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS"
echo "Model     : Qwen3 MoE 1B (12 layers, 128 experts, top-8, scaled from 30B)"
echo "Vocab Size: 151936"
echo "Init Loss : 11.9312"
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
            examples/models/qwen/pretrain_qwen3_moe_30b.py \
            --per-split-data-args-path=$BLEND_PATH \
            logger.wandb_project=variable-moe-routing \
            logger.wandb_entity=lukefriedrichs-paderborn-university \
            logger.wandb_exp_name=$RUN_NAME \
            logger.log_interval=1 \
            model.moe_per_layer_logging=True \
            dataset.num_workers=4 \
            dataset.mmap_bin_files=True \
            mixed_precision=$MIXED_PRECISION \
            train.global_batch_size=$GLOBAL_BATCH_SIZE \
            train.micro_batch_size=$MICRO_BATCH_SIZE \
            train.train_iters=$TRAIN_ITERS \
            scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
            model.num_layers=12 \
            model.hidden_size=768 \
            model.num_attention_heads=12 \
            model.num_query_groups=4 \
            model.ffn_hidden_size=2304 \
            model.moe_ffn_hidden_size=192 \
            model.num_moe_experts=128 \
            model.moe_router_topk=8 \
            model.routing_type=$ROUTING_TYPE \
            model.moe_aux_loss_coeff=$AUX_LOSS_COEFF \
            model.moe_topany_target_k=8 \
            model.moe_topany_update_rate=$THRESHOLD_UPDATE_RATE \
            model.moe_topany_threshold_update_mode=${THRESHOLD_UPDATE_MODE:-sign} \
            model.tensor_model_parallel_size=$TP \
            model.pipeline_model_parallel_size=1 \
            model.expert_model_parallel_size=$EP \
            model.moe_token_dispatcher_type=alltoall \
            model.sequence_parallel=False \
            model.context_parallel_size=$CP \
            model.seq_length=$SEQ_LENGTH \
            dataset.sequence_length=$SEQ_LENGTH \
            model.recompute_granularity=null \
            checkpoint.save=$CHECKPOINT_DIR \
            checkpoint.save_interval=$SAVE_INTERVAL
    "

echo "=============================="
echo "Job finished"
echo "=============================="
