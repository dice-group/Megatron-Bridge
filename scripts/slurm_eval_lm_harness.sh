#!/bin/bash
#
# Run lm-evaluation-harness on a Megatron-Bridge checkpoint.
#
# Usage:
#   sbatch --export=ALL,CHECKPOINT=/scratch/.../checkpoints_lossfree,TASKS=hellaswag,arc_easy \
#       scripts/slurm_eval_lm_harness.sh
#
# Or set vars in the environment then sbatch:
#   CHECKPOINT=/path/to/ckpt TASKS="hellaswag,arc_easy,piqa" sbatch scripts/slurm_eval_lm_harness.sh
#
# The script auto-loads the model architecture from <CHECKPOINT>/run_config.yaml,
# so no model.* overrides are needed even if the checkpoint came from
# slurm_train_super_lossfree_1b.sh, slurm_train_qwen3_moe_1b.sh, etc.

#SBATCH --job-name=lm-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128GB
#SBATCH --account=hpc-prf-merlin
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT=/path/to/checkpoint_dir}"
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,piqa,winogrande,boolq,openbookqa}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
LIMIT="${LIMIT:-}"   # leave empty to run full task; set e.g. 50 for smoke test
OUTPUT_PATH="${OUTPUT_PATH:-logs/eval_${SLURM_JOB_ID}.json}"

# ==============================================================================

mkdir -p logs
CONTAINER="$PWD/../nemo-container"

GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
if [ "${GPU_CC:-0}" -ge 89 ]; then
    module load tools/Apptainer/1.3.5-GCCcore-13.3.0
else
    module load tools/Apptainer/1.3.4-GCCcore-13.3.0
fi

# Build optional --limit arg
LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

echo "=============================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Container : $CONTAINER"
echo "Checkpoint: $CHECKPOINT"
echo "Tasks     : $TASKS"
echo "Batch sz  : $BATCH_SIZE  fewshot=$NUM_FEWSHOT  limit=${LIMIT:-full}"
echo "Output    : $OUTPUT_PATH"
echo "=============================="

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

apptainer exec \
    --nv \
    --no-home \
    --bind "$PWD":/opt/Megatron-Bridge \
    --bind "$CHECKPOINT":"$CHECKPOINT" \
    --pwd /opt/Megatron-Bridge \
    "$CONTAINER" \
    bash -c "
        export HOME=/tmp
        export HF_HOME=/tmp/hf_cache

        pip install lm-eval --quiet

        torchrun --nproc-per-node=1 \
            scripts/lm_eval_megatron.py \
            --checkpoint $CHECKPOINT \
            --tasks $TASKS \
            --batch-size $BATCH_SIZE \
            --num-fewshot $NUM_FEWSHOT \
            --output-path $OUTPUT_PATH \
            $LIMIT_ARG
    "

echo "=============================="
echo "Job finished. Results: $OUTPUT_PATH"
echo "=============================="
