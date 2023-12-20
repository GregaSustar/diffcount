#!/bin/bash

#SBATCH --job-name=gendm
#SBATCH --output=/d/hpc/projects/FRI/DL/gs1121/logs/gendm.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=2GB

export PROJECT_ROOT="/d/hpc/home/gs1121/alpha"
export DATA_ROOT="/d/hpc/projects/FRI/DL/gs1121/FSC147_384_V2/"

ksize=3
sigma=0.25

echo "Generating density maps with ksize=$ksize and sigma=$sigma"
cd $PROJECT_ROOT

echo "PWD: ${PWD}"

srun --kill-on-bad-exit=1 python -u -c "from sgm.data.fsc147 import generate_density_maps; generate_density_maps('${DATA_ROOT}', ${ksize}, ${sigma})"

echo "Done"