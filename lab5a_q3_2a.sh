#!/bin/bash

#SBATCH --job-name=lab5a_q3_2a

#SBATCH --nodes=1

#SBATCH --cpus-per-task=1

#SBATCH --output=%x.out

#SBATCH --mem=4GB

#SBATCH --time=00:10:00



module purge

module load cuda/11.6.1
module --ignore-cache load "cuda/11.6.1"
eval "$(conda shell.bash hook)"
                                                             
conda activate idle

cd /scratch/ns5429/lab5

nvcc -o lab5a_q3_2a.cu

