#!/bin/bash

#SBATCH --job-name=cld-training
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --output=/home/syw24/ftag/hepattn/src/hepattn/experiments/cld/slurm_logs/slurm-%j.%x.out


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
echo "nvidia-smi:"
nvidia-smi

# Move to workdir
cd /home/syw24/ftag/hepattn
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/home/syw24/tmp

# Run the training
echo "Running training script..."

# Python command that will be run
# CONFIG_PATH="/home/syw24/ftag/hepattn/src/hepattn/experiments/cld/configs/base.yaml"
CONFIG_PATH="/home/syw24/ftag/hepattn/logs/CLD_5_96_charged_10MeV_single3d_simflag_F16_20250618-T041315/config.yaml"
CKPT_PATH="/home/syw24/ftag/hepattn/logs/CLD_5_96_charged_10MeV_single3d_simflag_F16_20250618-T041315/ckpts/epoch=000-train_loss=3.94384.ckpt"
# PYTORCH_CMD="python src/hepattn/experiments/cld/main.py fit --config $CONFIG_PATH"
PYTORCH_CMD="python src/hepattn/experiments/cld/main.py test --config $CONFIG_PATH --ckpt_path $CKPT_PATH"

# Pixi commnand that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
APPTAINER_CMD="apptainer run --nv --bind /home/syw24 --bind /share/rcifdata/maxhart /home/syw24/ftag/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
