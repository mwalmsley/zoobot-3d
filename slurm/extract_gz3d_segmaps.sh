#!/bin/bash
#SBATCH --job-name=ex-seg
#SBATCH --output=%x_%A_%a.log                                 # "reserve all the available memory on each node assigned to the job"
#SBATCH --no-requeue                                    # Do not resubmit a failed jobt
#SBATCH --time=72:00:00       

#SBATCH --mem=10GB
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8

#SBATCH --array=0-16

pwd; hostname; date

PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python

srun $PYTHON /share/nas2/walml/repos/zoobot-3d/extract_gz3d_segmaps.py
