#!/bin/bash

arch=$1
dataset=$2
seeds=(0)

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 arch dataset" >&2
  exit 1
fi

for seed in "${seeds[@]}"; do
  sbatch experiment_scripts/infer_shadows_per_node.sh $seed $arch $dataset 0
  sbatch experiment_scripts/infer_shadows_per_node.sh $seed $arch $dataset 30
  sbatch experiment_scripts/infer_shadows_per_node.sh $seed $arch $dataset 60
  sbatch experiment_scripts/infer_shadows_per_node.sh $seed $arch $dataset 90
done