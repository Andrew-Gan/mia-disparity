#!/bin/bash

#SBATCH -A zghodsi -q normal -p ai -c 14 --gpus-per-node=1 --time=1440

export TORCH_HOME=${SCRATCH}/torch
export HF_HUB_DISABLE_PROGRESS_BARS=1
export DATA_DIR="${SCRATCH}/mia/data"

dataset=$1
arch=$2
mia=$3
seed=$4

data_dir="${DATA_DIR}/miae_standard_exp/target"

preds_dir="${DATA_DIR}/miae_standard_exp/preds_sd${seed}"
lira_shadow_dir="$preds_dir/$dataset/$arch/lira_shadow_ckpts"

prepare_dir="${preds_dir}/prepare_sd${seed}"
result_dir="$preds_dir/$dataset/$arch/${mia}"

if [[ "$arch" == "densenet121"* || "$arch" == "resnet50"* || "$arch" == "resnet56"* ]]; then
  attack_lr=0.1
elif [[ "$arch" == "alexnet"* ]]; then
  attack_lr=0.01
elif [[ "$arch" == "vgg19"* ]]; then
  attack_lr=0.05
fi

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

if [[ "$arch" == *"_"* ]]; then
  lira_shadow_dir="$preds_dir/$dataset/${arch%%_*}_dp/lira_shadow_ckpts"
else
  lira_shadow_dir="$preds_dir/$dataset/$arch/lira_shadow_ckpts"
fi

for id in {0..19}; do
  mkdir -p "$lira_shadow_dir/$id"
  if [ ! -f "$lira_shadow_dir/$id/shadow.pth" ]; then
    python obtain_pred.py --dataset "$dataset" --target_model "$arch" \
      --attack "$mia" --train_shadow_models "True" --shadow_id $id \
      --seed "$seed" --delete-files "True" --data_aug "False" \
      --attack_epochs "$num_epoch" --data_path "$data_dir" --shuffle_seed 1 \
      --result_path "$result_dir"\
      --preparation_path "$prepare_dir" \
      --data_path "$data_dir" \
      --dataset_file_root="$data_dir" \
      --lira_shadow_path "$lira_shadow_dir" \
      --pretrained "True" \
      --attack_lr "$attack_lr"
  fi
done