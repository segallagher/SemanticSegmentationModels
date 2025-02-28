#!/bin/bash

# Login to wandb
python3 -m wandb login $WANDB_API_KEY

if [ $? -ne 0 ]; then
    echo "wandb api key not provided"
    exit 1
fi

# Train model
echo -e "\nTraining Model\n"
python3 train.py

if [ $? -ne 0 ]; then
    echo "Training failed, stopping program"
    exit 1
fi

# Inference model
echo -e "\nInferencing Model\n"
python3 inference.py  --model_path="./output_dir/model.keras"
