#!/bin/bash
#SBATCH --job-name=segswp                # Job name
#SBATCH --array=1-200%1
#SBATCH --output=%x_%A_%a.log 
#SBATCH --mem=0
#SBATCH -c 24                                      # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=24:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --exclusive   # only one task per node

# sbatch should run this to make the agent e.g. sbatch slurm/start_sweep_agent.sh

/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python -m wandb agent --count 1 jbca-ice/zoobot-3d/5sqsjjiv
# agent will then run the shell script to create the python command

