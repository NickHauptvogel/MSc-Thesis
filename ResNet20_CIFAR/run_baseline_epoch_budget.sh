#!/bin/bash

#export NVIDIA_VISIBLE_DEVICES=all
#export CUDA_VISIBLE_DEVICES=0
#export NVIDIA_DRIVER_CAPABILITIES=compute,utility
#SBATCH -o epoch_budget_%a.out
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:titanx:1
#SBATCH --array=1-20

nvidia-smi

# Run experiment
printf "\n\n* * * Run SGD for cluster size = $SLURM_ARRAY_TASK_ID. * * *\n\n\n"

# Compute budget as 1500 / cluster size
budget=$((1500 / $SLURM_ARRAY_TASK_ID))
echo "Budget: $budget"

# Train cluster size models in a for loop
for i in $(seq 1 $SLURM_ARRAY_TASK_ID)
do
    printf "\n\n* * * Run SGD for ID = ${SLURM_ARRAY_TASK_ID}_$i. * * *\n\n\n"
    python -m sgd_baseline \
        --id=$(printf "%02d_%02d" $SLURM_ARRAY_TASK_ID $i) \
        --epochs=$budget \
        --data_augmentation \
        --nesterov \
        --checkpointing
        #--augm_shift=0.1 \
        #--initial_lr=1e-3 \
        #--l2_reg=1e-4 \
        #--optimizer=adam \
done
