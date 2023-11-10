#!/bin/bash

# PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python
# TRAIN_SCRIPT_LOC=/share/nas2/walml/repos/zoobot-3d/train.py

# $PYTHON $TRAIN_SCRIPT_LOC hydra/launcher=submitit_slurm debug=True

# #!/bin/bash

# quick local check
# sbatch train.sh debug=True

# seg loss only, gz3d only (as even simpler baseline)
sbatch train.sh "oversampling_ratio=1 use_vote_loss=False use_seg_loss=True vote_loss_weighting=0.01 gz3d_galaxies_only=True"
# sbatch train.sh "oversampling_ratio=1 use_vote_loss=True use_seg_loss=True vote_loss_weighting=0.01 gz3d_galaxies_only=True" 
# repeat with L1 loss instead
# sbatch train.sh "oversampling_ratio=1 use_vote_loss=False use_seg_loss=True seg_loss_weighting=100 gz3d_galaxies_only=True seg_loss_metric=l1"
# sbatch train.sh "oversampling_ratio=1 use_vote_loss=True use_seg_loss=True seg_loss_weighting=100 gz3d_galaxies_only=True seg_loss_metric=l1" 
# L1/constnorm results are quite similar to MSE, I think I prefer the L1 results for being appropriately uncertain
# all four still show unpredictable improvement at some point during training, but end up very similar
# set L1 and constnorm as default


# sbatch train.sh "oversampling_ratio=5 use_vote_loss=False use_seg_loss=True vote_loss_weighting=0.01 max_additional_galaxies=10000"
# sbatch train.sh "oversampling_ratio=5 use_vote_loss=True use_seg_loss=True vote_loss_weighting=0.01 max_additional_galaxies=10000"


# all galaxies, checking each loss
# sbatch train.sh "oversampling_ratio=10 use_vote_loss=False use_seg_loss=True seg_loss_weighting=1. vote_loss_weighting=0.01 random_state=42 max_additional_galaxies=50000 spiral_galaxies_only=False"
# sbatch train.sh "oversampling_ratio=10 use_vote_loss=True use_seg_loss=True seg_loss_weighting=1. vote_loss_weighting=0.01 random_state=42 max_additional_galaxies=50000 spiral_galaxies_only=False"

# sbatch train.sh "oversampling_ratio=10 use_vote_loss=False use_seg_loss=True seg_loss_weighting=400 random_state=42"
# sbatch train.sh "oversampling_ratio=10 use_vote_loss=True use_seg_loss=True seg_loss_weighting=400 random_state=42"
# and with a different seed to check consistency
# sbatch train.sh "oversampling_ratio=10 use_vote_loss=False use_seg_loss=True seg_loss_weighting=300 random_state=43"
# sbatch train.sh "oversampling_ratio=10 use_vote_loss=True use_seg_loss=True seg_loss_weighting=300 random_state=43"

# use seg loss only, or use both losses
# sbatch train.sh "oversampling_ratio=1 use_vote_loss=False use_seg_loss=True seg_loss_weighting=100"
# sbatch train.sh "oversampling_ratio=1 use_vote_loss=True use_seg_loss=True seg_loss_weighting=100" 

# add oversampling
# sbatch train.sh "oversampling_ratio=10 use_vote_loss=True use_seg_loss=True"

# as above, extra oversampling, and not filtering out non-feat galaxies
# sbatch train.sh "oversampling_ratio=30 use_vote_loss=True use_seg_loss=True spiral_galaxies_only=False"

# sbatch train.sh "oversampling_ratio=1 use_vote_loss=True use_seg_loss=False loss_to_monitor=validation/epoch_vote_loss:0"

# sbatch train.sh "oversampling_ratio=1 use_vote_loss=True use_seg_loss=False loss_to_monitor=validation/epoch_vote_loss:0 use_dummy_encoder=True"
