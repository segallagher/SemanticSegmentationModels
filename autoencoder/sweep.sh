#!/bin/bash

# Login to wandb
python3 -m wandb login $WANDB_API_KEY

if [ $? -ne 0 ]; then
    echo "wandb api key not provided"
    exit 1
fi

# Sweep
output=$(python3 -m wandb sweep --project autoencoder sweep.yaml 2>&1)

sweep_command=$(printf '%s\n' "$output" |
  sed -nE 's/.*Run sweep agent with: (.*)/\1/p' |
  head -n 1)

python3 -m $sweep_command

echo -e "Sweep done"