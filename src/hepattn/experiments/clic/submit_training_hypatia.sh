#!/bin/bash

#SBATCH --job-name=mpflow-training
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --output=/share/gpu1/syw24/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

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
cd /share/gpu1/syw24/hepattn
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/share/gpu1/syw24/tmp

# Run the training
echo "Running training script..."

# Python command that will be run
CONFIG_PATH="/share/gpu1/syw24/hepattn/src/hepattn/experiments/clic/configs/base.yaml"
# PYTORCH_CMD="python src/hepattn/experiments/clic/main.py fit --config $CONFIG_PATH"
PYTORCH_CMD="python src/hepattn/experiments/clic/main.py fit --config $CONFIG_PATH"

# CONFIG_PATH="/share/gpu1/syw24/hepattn/logs/CLIC_Pflow_FullDiceFocFix_BF16_B64_20250625-T144653/config.yaml"
# CKPT_PATH="/share/gpu1/syw24/hepattn/logs/CLIC_Pflow_FullDiceFocFix_BF16_B64_20250625-T144653/ckpts/epoch=046-val_loss=1.07280.ckpt"
# PYTORCH_CMD="python src/hepattn/experiments/clic/main.py test --config $CONFIG_PATH --ckpt_path $CKPT_PATH"

# Setting up for multiple gpus training
export MASTER_ADDR=$(hostname -f)
export MASTER_PORT=$((12000 + RANDOM % 1000))
export PYTORCH_DISTRIBUTED_BACKEND=nccl
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=$(ip route get 8.8.8.8 | awk '{print $5}')
export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export PYTORCH_DIST_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_P2P_DISABLE=1

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"

# Pixi commnand that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
# Add srun in front of apptainer command for multiple gpus training
APPTAINER_CMD="srun apptainer run --nv --home /share/gpu1/syw24 /share/gpu1/syw24/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
