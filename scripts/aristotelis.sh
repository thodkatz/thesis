#!/bin/bash

#SBATCH --partition=ampere
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH -o /home/k/katzalis/repos/thesis/slurm_scripts/%x_%j.out
#SBATCH -e /home/k/katzalis/repos/thesis/slurm_scripts/%x_%j.err

nvidia-smi
source ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/k/katzalis/miniforge3/envs/thesis/lib

conda activate thesis
python /home/k/katzalis/repos/thesis/scripts/multi_task.py --model simple --seed 1 --batch 4 --dataset nault-multi --perturbation tcdd
