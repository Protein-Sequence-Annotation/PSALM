#!/bin/bash

#SBATCH -J rebuild_profiles
#SBATCH -p eddy
#SBATCH -c 1
#SBATCH -t 30-00:00
#SBATCH -o logs/rebuild.out
#SBATCH -e logs/rebuild.err
#SBATCH --mem 20000 #20GB


module load gcc
module load openmpi
module load Mambaforge

mamba deactivate
mamba activate TreeHMM

py_dir="../library"

python ${py_dir}/build_profiles.py