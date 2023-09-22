#!/bin/bash
#SBATCH --job-name=train                    # Job name
#SBATCH --output=%x_%A.log 
#SBATCH --mem=0
#SBATCH --cpus-per-task=24                              # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=72:00:00
#SBATCH --constraint=A100    
#SBATCH --exclusive   # only one task per node

pwd; hostname; date

SAVE_DIR=/share/nas2/walml/repos/zoobot-3d/results/models/train_$RANDOM
srun /share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python /share/nas2/walml/repos/zoobot-3d/train.py --save-dir $SAVE_DIR
# srun /share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python /share/nas2/walml/repos/zoobot-3d/train.py --save-dir $SAVE_DIR --debug
