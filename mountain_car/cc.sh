#!/bin/bash

#SBATCH --account=def-sutton
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dashley@ualberta.ca
#SBATCH --error=slurm-%j-%n-%a.err
#SBATCH --output=slurm-%j-%n-%a.out
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=0-12:00

source ~/.bashrc
module load python/3.7
module load scipy-stack/2019b
source env/bin/activate
'./tasks_'"$SLURM_ARRAY_TASK_ID"'.sh'
