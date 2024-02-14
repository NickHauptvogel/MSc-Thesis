#!/bin/bash

#export NVIDIA_VISIBLE_DEVICES=all
#export CUDA_VISIBLE_DEVICES=0
#export NVIDIA_DRIVER_CAPABILITIES=compute,utility
#SBATCH -o %a.out
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:titanx:1
#SBATCH --array=1-30

nvidia-smi

# Run experiment
printf "\n\n* * * Run SGD for ID = $SLURM_ARRAY_TASK_ID. * * *\n\n\n"
python -m sgd_baseline \
    --id=${SLURM_ARRAY_TASK_ID} \
    #--augm_shift=0.1 \
    #--initial_lr=1e-3 \
    #--l2_reg=1e-4 \
    #--optimizer=adam \
    --validation_split=0.1 \
    --checkpointing=True