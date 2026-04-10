#!/bin/bash

#SBATCH -A zghodsi -q normal --mem=16G -p ai -c 14 --gpus-per-node=1 --time=120

export TORCH_HOME=${SCRATCH}/torch
export HF_HUB_DISABLE_PROGRESS_BARS=1
export DATA_DIR="${SCRATCH}/mia/data"

dataset=$1
arch=$2
mia=$3
id=$4

seed=$5 # keep seed = 0

data_dir="${DATA_DIR}/miae_standard_exp/target"
target_model_path="$data_dir/target_models"
target_model_save_path="$target_model_path/$dataset/$arch"

preds_dir="${DATA_DIR}/miae_standard_exp/preds_sd${seed}"
lira_shadow_dir="$preds_dir/$dataset/$arch/lira_shadow_ckpts"

prepare_dir="${preds_dir}/prepare_sd${seed}"
result_dir="$preds_dir/$dataset/$arch/${mia}"

if [ "$dataset" == "cifar10_32" ]; then
  num_epoch=60
elif [ "$dataset" == "cifar10_256" ]; then
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

if [[ "$arch" == "densenet121" || "$arch" == "resnet50" ]]; then
  lr=0.001
elif [[ "$arch" == "alexnet" || "$arch" == "vgg19" ]]; then
  lr=0.0001
else
  lr=0.1
fi

python obtain_pred.py --dataset "$dataset" --target_model "$arch" --attack "$mia" \
      --train_shadow_models "True" --shadow_id $id \
      --seed "$seed" --delete-files "True" --data_aug "False" \
      --attack_epochs "$num_epoch" --data_path "$data_dir" --shuffle_seed 1 \
      --result_path "$result_dir"\
      --preparation_path "$prepare_dir" \
      --target_model_path "$target_model_save_path" \
      --data_path "$data_dir" \
      --dataset_file_root="$data_dir" \
      --lira_shadow_path "$lira_shadow_dir" \
      --attack_lr "$lr"