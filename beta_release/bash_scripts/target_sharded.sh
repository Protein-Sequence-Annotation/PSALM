#!/bin/bash

#SBATCH -J targets
#SBATCH -p eddy
#SBATCH -c 5
#SBATCH -t 1-00:00
#SBATCH -o logs/processing.out
#SBATCH -e logs/processing.err
#SBATCH --mem 10000 #20GB
# #SBATCH --array=61-100

module load gcc #/13.2.0-fasrc01
module load openmpi
module load Mambaforge

mamba deactivate
mamba activate protllm

py_dir = "py_scripts"

python ${py_dir}/target_precomputing.py #${SLURM_ARRAY_TASK_ID}