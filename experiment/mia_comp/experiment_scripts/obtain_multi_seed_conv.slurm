#!/bin/sh

#SBATCH -A zghodsi -q normal -p smallgpu -c 64 --gpus-per-node=1 --time=30

# modify this to set up directory:
DATA_DIR="${SCRATCH}/mia/data"

experiment_dir="${DATA_DIR}/miae_standard_exp"

plot_dir="${DATA_DIR}/miae_standard_exp/multiseed_convergence"

dataset=$2 #("cifar10" "cifar100")
arch=$1 #("densenet121" "resnet50" "vgg19")
mias=("lira") #("losstraj" "shokri" "yeom" "lira" "calibration" "reference")
fprs=(0.1)
seeds=(0 20 40 60 80 100)

# prepare the list of mias and fprs as arguments
mialist=""
for mia in "${mias[@]}"; do
    mialist+="${mia} "
done
fprlist=""
for fpr in "${fprs[@]}"; do
    fprlist+="${fpr} "
done
seedlist=""
for seed in "${seeds[@]}"; do
    seedlist+="${seed} "
done

# clean the plot directory
rm -rf "${plot_dir:?}/${dataset:?}/${arch:?}"
mkdir -p ${plot_dir}/${dataset}/${arch}
echo "plots at ${plot_dir}/${dataset}/${arch}"

# convert fprlist to space-separated string
fprlist=$(printf "%s " "${fprs[@]}")

# plot the graphs
# common TP (intersection of all seeds)
graph_title="multi-seed common TP for ${dataset} ${arch}"
graph_path="${plot_dir}/${dataset}/${arch}/multi_seed_intersection_TP"
rm -rf "${graph_path}"
mkdir -p ${graph_path}
python3 obtain_graphs.py --graph_type "multi_seed_convergence_intersection"\
                            --dataset "${dataset}"\
                            --graph_title "${graph_title}"\
                            --data_path "${experiment_dir}"\
                            --graph_path "${graph_path}"\
                            --architecture "${arch}"\
                            --attacks ${mialist}\
                            --fpr ${fprlist}\
                            --seed ${seedlist}

# attack coverage (union of all seeds)
graph_title="multi-seed attack coverage for ${dataset} ${arch}"
graph_path="${plot_dir}/${dataset}/${arch}/multi_seed_union_TP"
rm -rf "${graph_path}"
mkdir -p ${graph_path}
python3 obtain_graphs.py --graph_type "multi_seed_convergence_union"\
                            --dataset "${dataset}"\
                            --graph_title "${graph_title}"\
                            --data_path "${experiment_dir}"\
                            --graph_path "${graph_path}"\
                            --architecture "${arch}"\
                            --attacks ${mialist}\
                            --fpr ${fprlist}\
                            --seed ${seedlist}