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

# Parallelism — defaults to data parallelism only on 4 GPUs
N_GPUS=4
TP=1
EP=1
CP=1
MICRO_BATCH_SIZE=4
GRAD_ACCUM_STEPS=1

# Derived
DP=$(( N_GPUS / (TP * EP * CP) ))
GLOBAL_BATCH_SIZE=$(( DP * MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS ))

# ==============================================================================
# Derived paths — do not edit
# ==============================================================================

CONTAINER="$PWD/../nemo-container"
BLEND_PATH="$DATA_DIR/blend.json"

# Iterations:
# global_batch_size=4, seq_length=2048 → 8192 tokens/step
# 50 000 steps ≈ 410M tokens ≈ 8-12 h on a single H100 for this model size
TRAIN_ITERS=50000
LR_WARMUP_ITERS=2000
SAVE_INTERVAL=1000

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
echo "Iters     : $TRAIN_ITERS"
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
            model.seq_length=2048 \
            dataset.sequence_length=2048 \
            checkpoint.save=$CHECKPOINT_DIR \
            checkpoint.save_interval=$SAVE_INTERVAL
    "

echo "=============================="
echo "Job finished"
echo "=============================="
