#!/bin/bash

#SBATCH -A zghodsi -q normal -p ai -c 14 --gpus-per-node=1 --time=60

export TORCH_HOME=${SCRATCH}/torch
export HF_HUB_DISABLE_PROGRESS_BARS=1

# modify this to set up directory:
export DATA_DIR="${SCRATCH}/mia/data"

# This script is used to partition the dataset into target dataset and shadow dataset, then train the target model
seed=0 # keep seed = 0
# for repeat training, we do shuffle_seed from 2 to 4 to reshuffle the target-auxiliary dataset partition

# for regular training, we do shuffle_seed from 1
shuffle_seed=1
data_dir="${DATA_DIR}/miae_standard_exp/target"
#data_dir="${DATA_DIR}/repeat_miae_standard_exp/miae_standard_exp_0/target"
mkdir -p "$data_dir"

dataset=$1

mkdir -p "$data_dir/$dataset"
# save the dataset
echo "Saving dataset $dataset"
python3 obtain_pred.py --pretrained "True" --dataset "$dataset" --save_dataset "True" --data_path "$data_dir" --seed "$seed" --data_aug "True" --shuffle_seed "$shuffle_seed"
