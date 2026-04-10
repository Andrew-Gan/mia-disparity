#!/bin/bash

#SBATCH -A zghodsi -q normal --mem=16G -p ai -c 14 --gpus-per-node=1 --time=120

datasets=("cifar10_32")
archs=($1)
seed=$2

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 arch seed" >&2
  exit 1
fi

# archs=("densenet121" "resnet50" "alexnet" "vgg19")
mia="lira"

export DATA_DIR="${SCRATCH}/mia/data"

preds_dir="${DATA_DIR}/miae_standard_exp/preds_sd${seed}"

for dataset in "${datasets[@]}"; do
  for arch in "${archs[@]}"; do
    for id in {0..19}; do
      lira_shadow_dir="$preds_dir/$dataset/$arch/lira_shadow_ckpts/$id"
      if [ ! -e $lira_shadow_dir ]; then
        bash experiment_scripts/prepare_shadows_per_node.sh $dataset $arch $mia $id $seed
      fi
    done
  done
done