#!/bin/bash

# PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python
# TRAIN_SCRIPT_LOC=/share/nas2/walml/repos/zoobot-3d/train.py

# $PYTHON $TRAIN_SCRIPT_LOC hydra/launcher=submitit_slurm debug=True

# #!/bin/bash

# quick local check
# sbatch train.sh debug=True

# use seg loss only
# sbatch train.sh "oversampling_ratio=1 use_vote_loss=False use_seg_loss=True"

# as above, but gz3d only (as even simpler baseline)
sbatch train.sh "oversampling_ratio=1 use_vote_loss=False use_seg_loss=True --gz3d_galaxies_only=True"

# use both lossses
sbatch train.sh oversampling_ratio=1 use_vote_loss=True use_seg_loss=True

# add oversampling
sbatch train.sh oversampling_ratio=10 use_vote_loss=True use_seg_loss=True

# all filter out non-spiral desi galaxies by default