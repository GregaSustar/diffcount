#!/bin/sh

#SBATCH --job-name=main
#SBATCH --output=/d/hpc/projects/FRI/DL/gs1121/logs/main.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=32G
#SBATCH --partition=gpu

export PROJECT_ROOT="/d/hpc/home/gs1121/alpha"
export DATA_ROOT="/d/hpc/projects/FRI/DL/gs1121/FSC147_384_V2/"
export CONFIG="$PROJECT_ROOT/configs/mnist_toy.yaml"
export LOG_DIR="/d/hpc/projects/FRI/DL/gs1121/logs"

cd $PROJECT_ROOT

srun --kill-on-bad-exit=1 python -u main.py --base $CONFIG --logdir $LOG_DIR