#!/bin/bash
#SBATCH --job-name=pull_nemo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=pull_nemo_%j.out
#SBATCH --error=pull_nemo_%j.err

# Load the necessary Apptainer module
module load tools/Apptainer/1.3.4-GCCcore-13.3.0

# Set up temp and cache directories to avoid filling up /tmp
export APPTAINER_TMPDIR=$PWD/.apptainer_tmp
export APPTAINER_CACHEDIR=$PWD/.apptainer_cache
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

echo "Starting container pull at $(date)"

# Execute the pull command
apptainer pull ../nemo-container docker://nvcr.io/nvidia/nemo:26.02.nemotron_3_super

echo "Finished container pull at $(date)"