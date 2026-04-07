#!/bin/bash

#SBATCH -A zghodsi -q normal --mem=64G -p ai -c 14 --gpus-per-node=1 --time=1440

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

datasets=("cifar10")
#datasets=("purchase100" "texas100")

archs=("densenet121" "resnet50" "alexnet" "vgg19")
#archs=("mlp_for_texas_purchase")

prepare_path="${DATA_DIR}/prepare_sd${seed}"

target_model_path="$data_dir/target_models"

for dataset in "${datasets[@]}"; do
  # if assign different num_epoch for different dataset
  if [ "$dataset" == "cifar10" ]; then
    num_epoch=60
  elif [ "$dataset" == "cifar100" ]; then
    num_epoch=100
  elif [ "$dataset" == "cinic10" ]; then
    num_epoch=60
  elif [ "$dataset" == "purchase100" ]; then
    num_epoch=30
  elif [ "$dataset" == "texas100" ]; then
    num_epoch=30
  fi

  mkdir -p "$data_dir/$dataset"
  # save the dataset
  echo "Saving dataset $dataset"
  python3 obtain_pred.py --pretrained "True" --dataset "$dataset" --save_dataset "True" --data_path "$data_dir" --seed "$seed" --data_aug "True" --shuffle_seed "$shuffle_seed"
    for arch in "${archs[@]}"; do
      # for each arch, train the target model
      mkdir -p "$target_model_path/$dataset/$arch"
      target_model_save_path="$target_model_path/$dataset/$arch"
      echo "Obtaining target_model for $dataset $arch"
      # if the target model is already trained, then skip
      if [ -f "$target_model_save_path/target_model_$arch$dataset.pkl" ]; then
        echo "Target model for $dataset $arch already exists, skip"
        continue
      fi
      python3 obtain_pred.py --train_target_model "True" --dataset "$dataset" --target_model "$arch" \
       --seed "$seed" --delete-files "True" --data_aug "True"  --target_model_path "$target_model_save_path" \
       --attack_epochs "$num_epoch" --target_epochs "$num_epoch" --data_path "$data_dir" --shuffle_seed "$shuffle_seed" \
       --attack_lr "0.001" --pretrained "True"
    done
done