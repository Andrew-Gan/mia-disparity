#!/bin/bash

export TORCH_HOME=${SCRATCH}/torch
export HF_HUB_DISABLE_PROGRESS_BARS=1

# run it by: `bash run_multi_seed.sh {0..5}`
# List of arguments
seeds=("$@")
arch=alexnet

# modify this to set up directory:
DATA_DIR="${SCRATCH}/mia/data"

script_out_dir=$DATA_DIR``

# for each seed
for sd in "${seeds[@]}"; do
    log_file="${script_out_dir}/output_${sd}.log"

    # Remove the log file if it exists
    # if [ -f "$log_file" ]; then
    #     rm "$log_file"
    #     echo "Removed existing log file: $log_file"
    # fi

    # Launch the experiment and save output to log file
    sbatch --output="$log_file" --error="$log_file" ./experiment_scripts/obtain_pred.sh "$sd" $arch
done

# # Wait for all background processes to complete
# wait

echo "All tasks launched. Check output files in $script_out_dir"