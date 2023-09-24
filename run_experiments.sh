#!/bin/bash

# PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python
# TRAIN_SCRIPT_LOC=/share/nas2/walml/repos/zoobot-3d/train.py

# $PYTHON $TRAIN_SCRIPT_LOC hydra/launcher=submitit_slurm debug=True

# #!/bin/bash

# use one loss or the other (baselines for each task)
# sbatch train.sh debug=True

sbatch train.sh "oversampling_ratio=1 use_vote_loss=True use_seg_loss=False"
sbatch train.sh "oversampling_ratio=1 use_vote_loss=False use_seg_loss=True"

# # use both lossses
# sbatch train.sh oversampling_ratio=1 use_vote_loss=True use_seg_loss=True
# # add oversampling
# sbatch train.sh oversampling_ratio=10 use_vote_loss=True use_seg_loss=True
# # all using gz3d masks plus spiral-predicted desi dr5 galaxies
