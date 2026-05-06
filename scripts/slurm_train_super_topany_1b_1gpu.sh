#!/bin/bash
#
# Nemotron-3 Super 1B with topany routing — single GPU.
# Mirrors the `small_topany_fb` config from sweep_routing_small.sh, scaled
# up to the 1B architecture used by slurm_train_super_lossfree_1b.sh.
#
# Knobs taken from small_topany_fb:
#   ROUTING_TYPE=topany, THRESHOLD_UPDATE_MODE=sign, THRESHOLD_UPDATE_RATE=0,
#   AUX_LOSS_COEFF=0.01, TOPANY_K_TARGET_COEFF=0.3, TOPANY_K_TARGET=2.5,
#   TOPANY_FORCE_TOP1=1
#
# Default TOPANY_K_TARGET=6 matches the 1B's topk baseline. Override to
# 2.5 if you want the literal small_topany_fb value (sparser regime).
#
# Usage:
#   sbatch scripts/slurm_train_super_topany_1b_1gpu.sh
#   # or override:
#   sbatch --export=ALL,RUN_NAME=...,TOPANY_K_TARGET=6 \
#       scripts/slurm_train_super_topany_1b_1gpu.sh

#SBATCH --job-name=moe-topany-1b-1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128GB
#SBATCH --account=hpc-prf-merlin
#SBATCH --output=logs/topany_1b_%j.out
#SBATCH --error=logs/topany_1b_%j.err

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR=/scratch/hpc-prf-merlin/luke/Megatron-Bridge/data/fineweb
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/hpc-prf-merlin/luke/Megatron-Bridge/checkpoints_topany_1b_1gpu_fb}"
RUN_NAME="${RUN_NAME:-super_topany_1b_1gpu_fb}"

# Load WandB key from .env
if [ -f "$PWD/.env" ]; then
    export $(grep -v '^#' "$PWD/.env" | xargs)
fi

# Routing knobs from small_topany_fb
ROUTING_TYPE="${ROUTING_TYPE:-topany}"
THRESHOLD_UPDATE_MODE="${THRESHOLD_UPDATE_MODE:-sign}"
THRESHOLD_UPDATE_RATE="${THRESHOLD_UPDATE_RATE:-0}"
AUX_LOSS_COEFF="${AUX_LOSS_COEFF:-0.01}"
# These are read by the Python recipe/router via os.environ if wired:
export TOPANY_K_TARGET_COEFF="${TOPANY_K_TARGET_COEFF:-0.3}"
export TOPANY_K_TARGET="${TOPANY_K_TARGET:-6}"
export TOPANY_FORCE_TOP1="${TOPANY_FORCE_TOP1:-1}"

# Token budget: 0 = full epoch
TRAIN_TOKENS="${TRAIN_TOKENS:-0}"

# 1-GPU preset (from slurm_train_super_lossfree_1b.sh header)
N_GPUS=1
TP=1
EP=1
CP=1
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-32}"
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
    print(f'ERROR: TRAIN_TOKENS ({train_tokens:,}) exceeds epoch tokens ({epoch_tokens:,})', file=sys.stderr)
    sys.exit(1)

tokens_per_iter = gbs * seq_length
train_iters     = train_tokens // tokens_per_iter
warmup_iters    = train_iters // 10

print(train_iters, warmup_iters, epoch_tokens, train_tokens)
" 2>"$_ITER_CALC_ERR")

if [ -z "$TRAIN_ITERS" ] || ! [[ "$TRAIN_ITERS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Failed to compute TRAIN_ITERS (got '$TRAIN_ITERS')."
    cat "$_ITER_CALC_ERR"
    rm -f "$_ITER_CALC_ERR"
    exit 1
fi
rm -f "$_ITER_CALC_ERR"

SAVE_INTERVAL=5000

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
echo "Routing   : $ROUTING_TYPE (mode=$THRESHOLD_UPDATE_MODE rate=$THRESHOLD_UPDATE_RATE aux=$AUX_LOSS_COEFF)"
echo "Top-any   : k_target=$TOPANY_K_TARGET k_target_coeff=$TOPANY_K_TARGET_COEFF force_top1=$TOPANY_FORCE_TOP1"
echo "Tokens    : $EFFECTIVE_TOKENS / $EPOCH_TOKENS (epoch)"
echo "Iters     : $TRAIN_ITERS (warmup=$LR_WARMUP_ITERS)"
echo "GPUs      : $N_GPUS (DP=$DP)"
echo "Batch     : global=$GLOBAL_BATCH_SIZE micro=$MICRO_BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS"
echo "Model     : Nemotron-3 Super 1B (topk=6 baseline, topany routing)"
echo "=============================="

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

MASTER_PORT=$((20000 + SLURM_JOB_ID % 30000))

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
        export TOPANY_K_TARGET_COEFF=$TOPANY_K_TARGET_COEFF
        export TOPANY_K_TARGET=$TOPANY_K_TARGET
        export TOPANY_FORCE_TOP1=$TOPANY_FORCE_TOP1

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
            checkpoint.save_interval=$SAVE_INTERVAL
    "

echo "=============================="
echo "Job finished"
echo "=============================="
