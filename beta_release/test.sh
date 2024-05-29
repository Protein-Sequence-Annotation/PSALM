#!/bin/bash

#SBATCH -J beta_test
#SBATCH -p eddy
#SBATCH -c 1
#SBATCH -t 5-00:00
#SBATCH -o logs/beta_test.out
#SBATCH -e logs/beta_test.err
#SBATCH --mem 40000 #80GB
#SBATCH --gres=gpu:1

module load cuda/12.0.1-fasrc01
module load cudnn/8.8.0.121_cuda12-fasrc01
module load gcc/13.2.0-fasrc01
module load Mambaforge

mamba deactivate
mamba activate protllm

python psalm_viz.py -rt ../data -i R6SWH5.fasta