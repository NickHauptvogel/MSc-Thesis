#!/bin/bash

#SBATCH -o log_%a.out4
#SBATCH --time=6-00:00:00
#SBATCH --gres=gpu:titanx:1
#SBATCH --array=1-10

# Declare output folder as variable
out_folder="results/retinopathy/resnet50/10_independent_smalllr_full_val_512_fullbatch"
max_ensemble_size=10

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
    --batch_size=8 \
    --epochs=90 \
    --model_type="ResNet50v1" \
    --initial_lr=0.0023072 \
    --l2_reg=0.00010674 \
    --checkpointing \
    --hold_out_validation_split=0.8 \
    --optimizer=adam \
    --use_case="retinopathy" \
    --lr_schedule="retinopathy" \
    --accumulation_steps=4
