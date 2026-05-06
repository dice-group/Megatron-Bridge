#!/bin/bash
#
# Sbatch wrapper for the agent_loop driver.
#
# Allocates a single H100 for up to 24h, then runs the loop driver inside
# that allocation. Each iteration the driver shells out to
# scripts/local_train_super_small.sh (which uses `apptainer exec` directly,
# NOT a nested sbatch — we already hold the GPU here).
#
# Usage:
#   sbatch scripts/slurm_agent_loop.sh [--max-iters 30] [--train-timeout 1200] \
#                                      [--skip-baseline] [--start-iter N]
#
# Anything after the script name is passed through to loop_driver.py.
#
# Resume: if the job dies, re-submit with --skip-baseline to keep the cached
# baseline.json, and --start-iter K to continue numbering at K.
#
#SBATCH --job-name=agent-moe-loop
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --exclusive
#SBATCH --mem=80GB
#SBATCH --account=hpc-prf-merlin
#SBATCH --output=logs/agent_loop_%j.out
#SBATCH --error=logs/agent_loop_%j.err

set -euo pipefail

mkdir -p logs

# Load .env (AGENT_API_KEY, WANDB_API_KEY, AGENT_MODEL, AGENT_API_URL).
if [ -f "$PWD/.env" ]; then
    export $(grep -v '^#' "$PWD/.env" | xargs)
else
    echo "ERROR: .env missing — copy .env.example to .env and fill it in." >&2
    exit 1
fi

if [ -z "${AGENT_API_KEY:-}" ]; then
    echo "ERROR: AGENT_API_KEY not set in .env." >&2
    exit 1
fi

# The driver shells out to apptainer for each training run.
module load tools/Apptainer/1.3.5-GCCcore-13.3.0

echo "=============================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Model     : ${AGENT_MODEL:-nemotron-3-super-120B-a12b}"
echo "Endpoint  : ${AGENT_API_URL:-default}"
echo "Args      : $*"
echo "=============================="

# `requests` is the only Python dep beyond the stdlib. Driver fails with a
# clear error if it's missing.
python3 -u agent_loop/loop_driver.py "$@"

echo "=============================="
echo "Agent loop finished"
echo "=============================="
