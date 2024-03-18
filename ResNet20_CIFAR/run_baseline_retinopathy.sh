#!/bin/bash

#SBATCH -o results/retinopathy/resnet50/30_independent_band_full_val/log_%a.out
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:titanx:1
#SBATCH --array=1-30

# Declare output folder as variable
out_folder="results/retinopathy/resnet50/30_independent_band_full_val"
max_ensemble_size=30

#export NVIDIA_VISIBLE_DEVICES=all
#export CUDA_VISIBLE_DEVICES=0
#export NVIDIA_DRIVER_CAPABILITIES=compute,utility

# If SLURM_ARRAY_TASK_ID is not set, set it to 1
if [ -z ${SLURM_ARRAY_TASK_ID+x} ]; then
    SLURM_ARRAY_TASK_ID=1
fi

nvidia-smi

# Run experiment
printf "\n\n* * * Run SGD for ID = $SLURM_ARRAY_TASK_ID. * * *\n\n\n"
python -m sgd_baseline \
    --id=$(printf "%02d" $SLURM_ARRAY_TASK_ID) \
    --seed=$SLURM_ARRAY_TASK_ID \
    --out_folder=$out_folder \
    --batch_size=32 \
    --epochs=90 \
    --model_type="ResNet50v1" \
    --initial_lr=0.023072 \
    --l2_reg=0.00010674 \
    --momentum=0.9901533 \
    --nesterov \
    --use_case="retinopathy" \
    --lr_schedule="retinopathy"