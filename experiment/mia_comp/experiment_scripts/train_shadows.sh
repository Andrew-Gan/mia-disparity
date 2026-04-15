#!/bin/bash

arch=$1
dataset=$2
seeds=(0 20 40 60 80 100)

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 arch dataset" >&2
  exit 1
fi

for seed in "${seeds[@]}"; do
  sbatch experiment_scripts/prepare_shadows_per_node.sh $dataset $arch "lira" $seed
done