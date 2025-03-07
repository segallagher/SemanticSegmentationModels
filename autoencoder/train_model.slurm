#!/bin/bash
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --tasks-per-node=4
#SBATCH --time=10:00:00
#SBATCH --output=model_train%j.out
#SBATCH --gpus=8

module load CUDA

module load GCCcore/12.3.0
module load Python/3.11.3

bash ./install.sh
bash ./run.sh