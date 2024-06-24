#!/bin/bash

#SBATCH -J beta_train
#SBATCH -p eddy
#SBATCH -c 1
#SBATCH -t 5-00:00
#SBATCH -o logs/beta_trainfull.out
#SBATCH -e logs/beta_trainfull.err
#SBATCH --mem 40000 #80GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1

module load cuda #/12.0.1-fasrc01
module load cudnn #/8.8.0.121_cuda12-fasrc01
module load gcc #/13.2.0-fasrc01
module load Mambaforge

mamba deactivate
mamba activate protllm

python psalm_train.py -m clan -rt ../dev/data -o batch_clan -ne 30 -lr 0.001 -ns 4 -v