#!/bin/bash

#SBATCH -o log_%a.out2
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:titanx:1
#SBATCH --array=1-30

# Declare output folder as variable
folder="ResNet20_CIFAR/"
out_folder="results/cifar100/resnet110/30_independent_wenzel_no_checkp_no_val"
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
    --validation_split=0.0 \
    --model_type="ResNet110v1" \
    --data_augmentation \
    --nesterov \
    --num_classes=100 \
    --initial_lr=0.01
    #--store_models
    #--test_time_augmentation
    #--checkpointing
    #--checkpoint_every=40
    #--hold_out_validation_split=0.5
    #--initial_lr=0.1
    #--SSE_lr
    #--bootstrapping

# If id is last id in array, run ensemble prediction
if [ $SLURM_ARRAY_TASK_ID -eq $max_ensemble_size ]
then
    cd ..
    printf "\n\n* * * Run Prediction for ensemble = $SLURM_ARRAY_TASK_ID. * * *\n\n\n"
    python -m ensemble_prediction \
        --folder=$(printf "%s%s" $folder $out_folder) \
        --max_ensemble_size=$SLURM_ARRAY_TASK_ID \
        --use_case="cifar10"
fi
