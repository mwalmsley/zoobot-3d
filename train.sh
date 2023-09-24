#!/bin/bash
#SBATCH --job-name=train                    # Job name
#SBATCH --output=%x_%A.log 
#SBATCH --mem=0
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=24                              # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=72:00:00
#SBATCH --constraint=A100    
#SBATCH --exclusive   # only one task per node

pwd; hostname; date

nvidia-smi

PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python

# export NCCL_DEBUG=INFO
# export PYTORCH_KERNEL_CACHE_PATH=/share/nas2/walml/.cache/torch/kernels

SAVE_DIR=/share/nas2/walml/repos/zoobot-3d/results/models

srun $PYTHON /share/nas2/walml/repos/zoobot-3d/train.py $HYDRA_OVERRIDES
