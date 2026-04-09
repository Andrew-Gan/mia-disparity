#!/bin/bash

#SBATCH -A zghodsi -q normal --mem=16G -p a30 -c 8 --gpus-per-node=1 --time=120

datasets=("cifar10")
archs=$1
# archs=("densenet121" "resnet50" "alexnet" "vgg19")
mia="lira"

export DATA_DIR="${SCRATCH}/mia/data"

seed=$2 # keep seed = 0
preds_dir="${DATA_DIR}/miae_standard_exp/preds_sd${seed}"

for dataset in "${datasets[@]}"; do
  for arch in "${archs[@]}"; do
    for id in {0..19}; do
      lira_shadow_dir="$preds_dir/$dataset/$arch/lira_shadow_ckpts/$id"
      if [ ! -e $lira_shadow_dir ]; then
        sbatch experiment_scripts/prepare_shadows_per_node.sh $dataset $arch $mia $id $seed
      fi
    done
  done
done