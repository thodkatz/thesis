#!/bin/bash
#SBATCH -o ./slurm_scripts/%x_%j.out
#SBATCH -e ./slurm_scripts/%x_%j.err
#SBATCH --nodelist=gpu40
#SBATCH -A kreshuk
#SBATCH -N 1
#SBATCH --mem 8050
#SBATCH -t 2880
#SBATCH --qos=normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=theodoros.katzalis@embl.de
#SBATCH -p gpu-el8

nvidia-smi
