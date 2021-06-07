#!/bin/bash

#SBATCH --error=slurm-%j-%n-%a.err
#SBATCH --output=slurm-%j-%n-%a.out
#SBATCH --cpus-per-task=6
#SBATCH --mem=64000M
#SBATCH --time=0-03:00

source ~/.bashrc
source ../../env/bin/activate
module load nixpkgs/16.09 scipy-stack/2019b python/3.7
./merge.sh $1
