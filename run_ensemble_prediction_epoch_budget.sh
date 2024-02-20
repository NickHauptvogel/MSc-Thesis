#!/bin/bash

max_ensemble_size=20

for i in $(seq 1 $max_ensemble_size)
do
    printf "\n\n* * * Run Prediction for ensemble = $i. * * *\n\n\n"
    python -m ensemble_prediction \
        --folder="ResNet20_CIFAR/results/epoch_budget_checkp/$(printf "%02d" $i)" \
        --max_ensemble_size=$i \
        --use_case="cifar10"
done
