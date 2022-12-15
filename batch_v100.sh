#!/bin/bash
#SBATCH --job-name=lab5a_q2_v100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --output=%x.out
#SBATCH --mem=8GB
#SBATCH --gres=gpu:v100
#SBATCH --time=02:00:00

#module purge
module load anaconda3/2020.07
module load python/intel/3.8.6
module load tqdm
#eval "$(conda shell.bash hook)"
#conda activate idls
cd /scratch/ns5429/lab5

python lab5a_q2.py
