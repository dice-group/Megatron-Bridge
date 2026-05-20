#!/bin/bash
#
# Run lm-evaluation-harness on a Megatron-Bridge checkpoint.
#
# Usage:
#   sbatch --export=ALL,CHECKPOINT=/path/to/ckpt,TASKS=hellaswag,arc_easy,WANDB_PROJECT=variable-moe-routing,WANDB_RUN_NAME=my_run \
#       scripts/slurm_eval_lm_harness.sh
#
# Or set vars in the environment then sbatch:
#   CHECKPOINT=/path/to/ckpt TASKS="hellaswag,arc_easy,piqa" sbatch scripts/slurm_eval_lm_harness.sh
#
# The script auto-loads the model architecture from <CHECKPOINT>/run_config.yaml,
# so no model.* overrides are needed even if the checkpoint came from
# slurm_train_super_lossfree_1b.sh, slurm_train_qwen3_moe_1b.sh, etc.
#
# If WANDB_PROJECT is set, per-task metrics are pushed to W&B under
# WANDB_RUN_NAME (which should match the original training run name so the
# eval row lands on the same run page).

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

# W&B logging — leave WANDB_PROJECT empty to disable.
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-lukefriedrichs-paderborn-university}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$(basename "$CHECKPOINT")}"

# Embed run name in the result filename so it's obvious which training run
# produced which eval (job id kept as a uniqueness suffix).
OUTPUT_PATH="${OUTPUT_PATH:-logs/eval_${WANDB_RUN_NAME}_${SLURM_JOB_ID}.json}"

# Load WANDB_API_KEY (and any other secrets) from .env
if [ -f "$PWD/.env" ]; then
    export $(grep -v '^#' "$PWD/.env" | xargs)
fi

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

# Build optional --wandb-* args
WANDB_ARGS=""
if [ -n "$WANDB_PROJECT" ]; then
    WANDB_ARGS="--wandb-project $WANDB_PROJECT --wandb-entity $WANDB_ENTITY --wandb-run-name $WANDB_RUN_NAME"
fi

echo "=============================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Container : $CONTAINER"
echo "Checkpoint: $CHECKPOINT"
echo "Tasks     : $TASKS"
echo "Batch sz  : $BATCH_SIZE  fewshot=$NUM_FEWSHOT  limit=${LIMIT:-full}"
echo "Output    : $OUTPUT_PATH"
echo "W&B       : ${WANDB_PROJECT:-disabled} / run=${WANDB_RUN_NAME}"

# Multiple eval jobs may land on the same node — give each torchrun a unique
# rendezvous port to avoid EADDRINUSE on the default 29500.
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 40000 ))
echo "MasterPort: $MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES (slurm): ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS               : ${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM_STEP_GPUS              : ${SLURM_STEP_GPUS:-<unset>}"
echo "--- nvidia-smi (outside container) ---"
nvidia-smi -L || echo "(nvidia-smi -L failed)"
echo "=============================="

# Fail fast on the no-GPU-allocated case so we don't spend the queue slot on
# a job that will crash inside Megatron's CUDA init.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -z "${SLURM_JOB_GPUS:-}" ] && [ -z "${SLURM_STEP_GPUS:-}" ]; then
    echo "ERROR: no GPU env vars set by SLURM — was --gres=gpu:...:1 honored?" >&2
    exit 1
fi

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# Explicitly forward GPU-visibility env vars to the container. Some apptainer
# configs scrub the env on exec; without these, --nv exposes the device files
# but the runtime sees CUDA_VISIBLE_DEVICES unset → "No CUDA GPUs are available".
apptainer exec \
    --nv \
    --no-home \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
    --env SLURM_JOB_GPUS="${SLURM_JOB_GPUS:-}" \
    --env SLURM_STEP_GPUS="${SLURM_STEP_GPUS:-}" \
    --env ADAMOE_NUM_NULL="${ADAMOE_NUM_NULL:-}" \
    --env ADAMOE_TOPK="${ADAMOE_TOPK:-}" \
    --bind "$PWD":/opt/Megatron-Bridge \
    --bind "$CHECKPOINT":"$CHECKPOINT" \
    --pwd /opt/Megatron-Bridge \
    "$CONTAINER" \
    bash -c "
        export HOME=/tmp
        export HF_HOME=/tmp/hf_cache
        echo 'CUDA_VISIBLE_DEVICES (in container): '\${CUDA_VISIBLE_DEVICES:-<unset>}
        nvidia-smi -L || echo '(nvidia-smi -L failed inside container)'

        pip install lm-eval wandb --quiet

        torchrun --nproc-per-node=1 \
            --master-port=$MASTER_PORT \
            scripts/lm_eval_megatron.py \
            --checkpoint $CHECKPOINT \
            --tasks $TASKS \
            --batch-size $BATCH_SIZE \
            --num-fewshot $NUM_FEWSHOT \
            --output-path $OUTPUT_PATH \
            $LIMIT_ARG \
            $WANDB_ARGS
    "

echo "=============================="
echo "Job finished. Results: $OUTPUT_PATH"
echo "=============================="
