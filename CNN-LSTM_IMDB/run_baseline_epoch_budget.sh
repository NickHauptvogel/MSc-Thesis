#!/bin/bash

#SBATCH -o epoch_budget_%a.out
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:titanx:1
#SBATCH --array=1-20

# Declare output folder as variable
folder="CNN-LSTM_IMDB/"
out_folder="results/epoch_budget"

nvidia-smi

# Run experiment
printf "\n\n* * * Run SGD for cluster size = $SLURM_ARRAY_TASK_ID. * * *\n\n\n"

# Compute budget as 1500 / cluster size
budget=$((500 / $SLURM_ARRAY_TASK_ID))
echo "Budget: $budget"

# Train cluster size models in a for loop
for i in $(seq 1 $SLURM_ARRAY_TASK_ID)
do
    printf "\n\n* * * Run SGD for ID = ${SLURM_ARRAY_TASK_ID}_$i. * * *\n\n\n"
    python -m sgd_baseline \
        --id=$(printf "%02d_%02d" $SLURM_ARRAY_TASK_ID $i) \
        --seed=$i \
        --epochs=$budget \
        --out_folder=$out_folder \
        --nesterov \
        --checkpointing \
        --validation_split=0.2 \
        --map_optimizer
done
