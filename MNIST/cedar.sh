#!/bin/bash

#SBATCH --account=def-sutton
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dashley@ualberta.ca
#SBATCH --error=slurm-%j-%n-%a.err
#SBATCH --output=slurm-%j-%n-%a.out
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32000M
#SBATCH --time=0-12:00

source ~/.bashrc
conda activate tensorflow-1.13.1
'./tasks_'"$SLURM_ARRAY_TASK_ID"'.sh'
