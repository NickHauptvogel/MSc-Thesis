#!/bin/bash

#export NVIDIA_VISIBLE_DEVICES=all
#export CUDA_VISIBLE_DEVICES=0
#export NVIDIA_DRIVER_CAPABILITIES=compute,utility
#SBATCH -o baseline-%A_%a.out
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:titanx:1
#SBATCH --array=21-30

nvidia-smi

# Run experiment
printf "\n\n* * * Run SGD for seed = $SLURM_ARRAY_TASK_ID. * * *\n\n\n"
python -m sgd_baseline --seed=${SLURM_ARRAY_TASK_ID}
