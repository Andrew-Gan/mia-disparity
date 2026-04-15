#!/bin/bash

#SBATCH -A zghodsi -q normal --mem=16G -p ai -c 14 --mem=16G --gpus-per-node=1 --time=120

export TORCH_HOME=${SCRATCH}/torch
export HF_HUB_DISABLE_PROGRESS_BARS=1

# modify this to set up directory:
DATA_DIR="${SCRATCH}/mia/data"

# This script is used to obtain the predictions of the attack on the target models
seed=$1

# if [ $# -eq 1 ]; then  # if the number of arguments is 1, the argument is the seed
#     seed=$1
# fi

echo "obtain_pred.sh seed = $seed"

data_dir="${DATA_DIR}/miae_standard_exp/target"
preds_dir="${DATA_DIR}/miae_standard_exp/preds_sd${seed}"
target_model_path="$data_dir/target_models"
prepare_path="${preds_dir}/prepare_sd${seed}"

mkdir -p "$preds_dir"

arch=$2 #("densenet121" "resnet50" "vgg19") ("resnet56" "mlp_for_texas_purchase")
dataset=$3 #datasets=("purchase100" "texas100")
mia="lira" #("lira" "reference" "shokri" "losstraj" "calibration" "yeom" "aug")

# if assign different num_epoch for different dataset
if [[ "$dataset" == "cifar10_32" || "$dataset" == "cifar10_256" ]]; then
  num_epoch=60
elif [ "$dataset" == "cifar100" ]; then
  num_epoch=100
elif [ "$dataset" == "cinic10" ]; then
  num_epoch=60
elif [ "$dataset" == "purchase100" ]; then
  num_epoch=30
elif [ "$dataset" == "texas100" ]; then
  num_epoch=30
else
  echo "Error: Unknown dataset $dataset"
  exit 1
fi

# for a given dataset and architecture, save the predictions
mkdir -p "$preds_dir/$dataset/$arch/${mia}"

result_dir="$preds_dir/$dataset/$arch/${mia}"
# if the predictions are already saved, skip
if [ -f "$result_dir/pred_$mia.npy" ]; then
    echo "Predictions already saved for $dataset $arch $mia at $result_dir/pred_$mia.npy"
    continue
else
    echo "Predictions not saved for $dataset $arch $mia at $result_dir/pred_$mia.npy"
fi

# if the preparation directory is not empty, delete it
if [ -d "$prepare_path" ] ; then
    rm -r "$prepare_path"
fi

mkdir -p "$result_dir"
prepare_dir="$prepare_path"

echo "Running $dataset $arch $mia"
target_model_save_path="$target_model_path/$dataset/$arch"

if [[ "$arch" == *"_"* ]]; then
  lira_shadow_dir="$preds_dir/$dataset/${arch%%_*}_dp/lira_shadow_ckpts"
else
  lira_shadow_dir="$preds_dir/$dataset/$arch/lira_shadow_ckpts"
fi

python3 obtain_pred.py \
  --dataset "$dataset" \
  --target_model "$arch" \
  --attack "$mia" \
  --result_path "$result_dir" \
  --seed "$seed" \
  --delete-files "True" \
  --preparation_path "$prepare_dir" \
  --data_aug "False"  \
  --target_model_path "$target_model_save_path" \
  --attack_epochs "$num_epoch" \
  --data_path "$data_dir" \
  --device "cuda:0" \
  --dataset_file_root="$data_dir" \
  --lira_shadow_path "$lira_shadow_dir" \
  --pretrained "True"

rm -r "$prepare_path"