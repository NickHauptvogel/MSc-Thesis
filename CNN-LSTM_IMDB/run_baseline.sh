#!/bin/bash

#SBATCH -o baseline_mle_%a.out
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:titanx:1
#SBATCH --array=1-50

# Declare output folder as variable
folder="CNN-LSTM_IMDB/"
out_folder="results/50_independent_wenzel_no_bootstr_mle"
max_ensemble_size=50

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
    --nesterov \
    --checkpointing \
    --validation_split=0.2
    #--map_optimizer


# If id is last id in array, run ensemble prediction
if [ $SLURM_ARRAY_TASK_ID -eq $max_ensemble_size ]
then
    cd ..
    printf "\n\n* * * Run Prediction for ensemble = $SLURM_ARRAY_TASK_ID. * * *\n\n\n"
    python -m ensemble_prediction \
        --folder=$(printf "%s%s" $folder $out_folder) \
        --max_ensemble_size=$SLURM_ARRAY_TASK_ID \
        --use_case="imdb"
fi
