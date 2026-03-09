#!/bin/bash

#SBATCH -A zghodsi -q normal --mem=256G -p ai -c 112 --gpus-per-node=8 --time=1440

export TORCH_HOME=${SCRATCH}/torch/$1
export HF_HUB_DISABLE_PROGRESS_BARS=1

# modify this to set up directory:
DATA_DIR="${SCRATCH}/mia/data"

# This script is used to partition the dataset into target dataset and shadow dataset, then train the target model
seed=0 # keep seed = 0
# for repeat training, we do shuffle_seed from 2 to 4 to reshuffle the target-auxiliary dataset partition

# for regular training, we do shuffle_seed from 1
shuffle_seed=1
data_dir="${DATA_DIR}/miae_standard_exp/target"
#data_dir="${DATA_DIR}/repeat_miae_standard_exp/miae_standard_exp_0/target"
mkdir -p "$data_dir"

num_epoch=10
batch_size=512

archs=("resnet56") #("densenet121" "resnet50" "alexnet" "vgg19")
dataset="cifar10"
mias=("lira") #"reference" "shokri" "losstraj" "calibration" "yeom" "aug")
lr=0.001 #resnet56: 0.001, resnet50: 0.0001

target_model_path="$data_dir/target_models"

# for each arch, train the target model
for arch in "${archs[@]}"; do
  mkdir -p "$target_model_path/$dataset/$arch"
  target_model_save_path="$target_model_path/$dataset/$arch"
  echo "Obtaining target_model for $dataset $arch"
  # if the target model is already trained, then skip
  if [ -f "$target_model_save_path/target_model_$arch$dataset.pkl" ]; then
    echo "Target model for $dataset $arch already exists, skip"
  else
    python3 obtain_pred.py --train_target_model "True" --dataset "$dataset" --target_model "$arch" \
      --seed "$seed" --delete-files "True" --data_aug "True"  --target_model_path "$target_model_save_path" \
      --attack_epochs "$num_epoch" --target_epochs "$num_epoch" --data_path "$data_dir" --shuffle_seed "$shuffle_seed" \
      --batch_size $batch_size --attack_lr $lr
  fi
done
echo "obtain_pred.sh seed = $seed"

preds_dir="${DATA_DIR}/miae_standard_exp/preds_sd${seed}"
prepare_path="${preds_dir}/prepare_sd${seed}"

mkdir -p "$preds_dir"

for arch in "${archs[@]}"; do
  # for a given dataset and architecture, save the predictions
  mkdir -p "$preds_dir/$dataset/$arch"

  # prepare a directory for lira shadow models so lira and other attacks (RMIA, reference) could share the same shadow models
  lira_shadow_dir="$preds_dir/$dataset/$arch/lira_shadow_ckpts"
  mkdir -p "$preds_dir/$dataset/$arch/lira_shadow_ckpts"

  for mia in "${mias[@]}"; do
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

    python3 obtain_pred.py \
      --dataset "$dataset" --target_model "$arch" --attack "$mia"\
      --result_path "$result_dir" --seed "$seed" --delete-files "True" \
      --preparation_path "$prepare_dir" --data_aug "False"  \
      --target_model_path "$target_model_save_path" --attack_epochs "$num_epoch" \
      --target_epochs "$num_epoch" --data_path "$data_dir" --device "cuda:0" \
      --dataset_file_root="$data_dir" --lira_shadow_path "$lira_shadow_dir" \
      --batch_size $batch_size --attack_lr $lr

    rm -r "$prepare_path"
  done
done
