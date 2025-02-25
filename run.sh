#!/bin/bash

DATASETS=("CLINC" "BANKING" "DBPEDIA" "HWU")
SEEDS=(0 1 2)

D=5000
SIGMA=1e-4
NUM_ENSEMBLES=5
MODEL_NAME="facebook/bart-base"


for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Running model on $dataset with seed $seed..."
        python evaluate.py "$dataset" --D $D --sigma $SIGMA --num_ensembles $NUM_ENSEMBLES --seed $seed --model_name "$MODEL_NAME"
    done
done

echo "All experiments completed!"