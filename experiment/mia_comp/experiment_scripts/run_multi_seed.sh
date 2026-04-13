#!/bin/bash

seeds=(0 20 40 60 80 100)
arch=$1

# modify this to set up directory:
DATA_DIR="${SCRATCH}/mia/data"

# for each seed
for sd in "${seeds[@]}"; do
    # Remove the log file if it exists
    # if [ -f "$log_file" ]; then
    #     rm "$log_file"
    #     echo "Removed existing log file: $log_file"
    # fi

    # Launch the experiment and save output to log file
    sbatch ./experiment_scripts/obtain_pred.sh $sd $arch
done

# # Wait for all background processes to complete
# wait