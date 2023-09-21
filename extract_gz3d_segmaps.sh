#!/bin/bash
#SBATCH --job-name=ex-seg
#SBATCH --output=%x_%A.log                                 # "reserve all the available memory on each node assigned to the job"
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=72:00:00       

#SBATCH --mem=0 
#SBATCH --exclusive   # only one task per node
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=24

pwd; hostname; date

PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python

srun $PYTHON /share/nas2/walml/repos/zoobot-3d/extract_gz3d_segmaps.py
