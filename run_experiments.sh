#!/bin/bash

sbatch train.sh
sbatch train.sh "seg_loss_metric=mse"
