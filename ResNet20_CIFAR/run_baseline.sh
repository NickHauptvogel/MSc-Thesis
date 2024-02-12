#!/bin/bash

# Use conda
#conda activate TF_GPU_2_1
#export NVIDIA_VISIBLE_DEVICES=all
#export CUDA_VISIBLE_DEVICES=0
#export NVIDIA_DRIVER_CAPABILITIES=compute,utility

nvidia-smi

# Hyperparameters to sweep
num_runs=1 # Number of repeated runs

# Generate parameter lists to sweep
seed_range=($(seq 1 $num_runs))

# Run experiment
for seed in ${seed_range[@]}; do
    # run experiment
    printf "\n\n* * * Run SGD for seed = $seed. * * *\n\n\n"
    python -m sgd_baseline --seed=$seed
done

printf "\n\n* * * All runs finished. * * *\n"
