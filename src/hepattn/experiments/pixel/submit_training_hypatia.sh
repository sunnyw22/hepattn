#!/bin/bash

#SBATCH --job-name=pixel-train
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/pixel/slurm_logs/slurm-%j.%x.out


# Comet variables
echo "Setting comet experiment key"
timestamp=$( date +%s )
COMET_EXPERIMENT_KEY=$timestamp
echo $COMET_EXPERIMENT_KEY
echo "COMET_WORKSPACE"
echo $COMET_WORKSPACE

# Print host info
echo "Hostname: $(hostname)"
echo "CPU count: $(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Move to workdir
cd /share/rcifdata/maxhart/hepattn/hepattn/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/share/rcifdata/maxhart/tmp/

echo "nvidia-smi:"
nvidia-smi

# Run the training
echo "Running training script..."

# Python command that will be run
PYTORCH_CMD="python src/hepattn/experiments/pixel/main.py fit --config src/hepattn/experiments/pixel/configs/base.yaml"

# Pixi commnand that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
APPTAINER_CMD="apptainer run --nv --bind /share/rcifdata/maxhart /share/rcifdata/maxhart/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
