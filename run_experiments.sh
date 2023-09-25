#!/bin/bash

# PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python
# TRAIN_SCRIPT_LOC=/share/nas2/walml/repos/zoobot-3d/train.py

# $PYTHON $TRAIN_SCRIPT_LOC hydra/launcher=submitit_slurm debug=True

# #!/bin/bash

# quick local check
# sbatch train.sh debug=True

# seg loss only, gz3d only (as even simpler baseline)
# sbatch train.sh "oversampling_ratio=1 use_vote_loss=False use_seg_loss=True seg_loss_weighting=100 gz3d_galaxies_only=True"
# sbatch train.sh "oversampling_ratio=1 use_vote_loss=True use_seg_loss=True seg_loss_weighting=100 gz3d_galaxies_only=True" 

# checking that early discontinuity with a different seed
sbatch train.sh "oversampling_ratio=1 use_vote_loss=False use_seg_loss=True seg_loss_weighting=100 gz3d_galaxies_only=True random_state=43"
sbatch train.sh "oversampling_ratio=1 use_vote_loss=True use_seg_loss=True seg_loss_weighting=100 gz3d_galaxies_only=True random_state=43"

# use seg loss only, or use both losses
# sbatch train.sh "oversampling_ratio=1 use_vote_loss=False use_seg_loss=True seg_loss_weighting=100"
# sbatch train.sh "oversampling_ratio=1 use_vote_loss=True use_seg_loss=True seg_loss_weighting=100" 

# add oversampling
# sbatch train.sh "oversampling_ratio=10 use_vote_loss=True use_seg_loss=True"

# as above, extra oversampling, and not filtering out non-feat galaxies
# sbatch train.sh "oversampling_ratio=30 use_vote_loss=True use_seg_loss=True spiral_galaxies_only=False"

# sbatch train.sh "oversampling_ratio=1 use_vote_loss=True use_seg_loss=False loss_to_monitor=validation/epoch_vote_loss:0"

# sbatch train.sh "oversampling_ratio=1 use_vote_loss=True use_seg_loss=False loss_to_monitor=validation/epoch_vote_loss:0 use_dummy_encoder=True"
