#!/bin/bash

#SBATCH --job-name=moe-routing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=64GB
#SBATCH --account=hpc-prf-merlin
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ==============================================================================
# CONFIGURATION — only these two lines need to be set before submitting
# ==============================================================================

# Directory produced by scripts/prepare_fineweb.sh --output <this dir>
DATA_DIR=/scratch/hpc-prf-merlin/luke/Megatron-Bridge/data/fineweb

# Where to save checkpoints
CHECKPOINT_DIR=/scratch/hpc-prf-merlin/luke/Megatron-Bridge/checkpoints

# Load WandB key from .env file
if [ -f "$PWD/.env" ]; then
    export $(grep -v '^#' "$PWD/.env" | xargs)
fi

# Token budget: 0 = train for one full epoch over the training split (default).
# Set to a positive integer to train on exactly that many tokens (must be ≤ epoch tokens).
TRAIN_TOKENS="${TRAIN_TOKENS:-0}"

# Parallelism — defaults to data parallelism only on 4 GPUs
N_GPUS=4
TP=1
EP=1
CP=1
MICRO_BATCH_SIZE=4
GRAD_ACCUM_STEPS=1
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

module load tools/Apptainer/1.3.5-GCCcore-13.3.0

echo "=============================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Container : $CONTAINER"
echo "Data dir  : $DATA_DIR"
echo "Blend     : $BLEND_PATH"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Tokens    : $EFFECTIVE_TOKENS / $EPOCH_TOKENS (epoch)"
echo "Iters     : $TRAIN_ITERS (warmup=$LR_WARMUP_ITERS)"
echo "GPUs      : $N_GPUS (DP=$DP, TP=$TP, EP=$EP, CP=$CP)"
echo "Batch     : global=$GLOBAL_BATCH_SIZE micro=$MICRO_BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS"
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
            logger.log_interval=1 \
            model.moe_per_layer_logging=True \
            +precision_config=bf16_with_fp8_current_scaling_mixed \
            train.global_batch_size=$GLOBAL_BATCH_SIZE \
            train.micro_batch_size=$MICRO_BATCH_SIZE \
            train.train_iters=$TRAIN_ITERS \
            scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
            model.num_layers=7 \
            model.hybrid_override_pattern=\"MEME*ME\" \
            model.num_moe_experts=8 \
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
