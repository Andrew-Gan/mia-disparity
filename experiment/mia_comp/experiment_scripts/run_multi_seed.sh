#!/bin/bash

arch=$1
dataset=$2
seeds=(0 20 40 60 80 100)

# if [ "$#" -lt 2 ]; then
#   echo "Usage: $0 arch dataset" >&2
#   exit 1
# fi

# prepare dataset
# sbatch experiment_scripts/prepare_target.slurm

# NOTE: train target models with BlazeDP before next steps!

# train shadow models
#for seed in "${seeds[@]}"; do
#    sbatch -W ./experiment_scripts/train_shadows.slurm $seed $arch $dataset &
#done

#wait

# obtain predictions
for sd in "${seeds[@]}"; do
    sbatch -W ./experiment_scripts/obtain_pred.slurm $sd $arch $dataset &
done

wait

# plot multi instance graph
sbatch ./experiment_scripts/obtain_multi_seed_conv.slurm $arch $dataset
