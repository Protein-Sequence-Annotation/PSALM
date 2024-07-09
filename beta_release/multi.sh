#!/bin/bash

#SBATCH -J multi
#SBATCH -p eddy
#SBATCH -c 1
#SBATCH -t 1-00:00
#SBATCH -o logs/multi.out
#SBATCH -e logs/multi.err
#SBATCH --mem 40000 #80GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:2

module load cuda #/12.0.1-fasrc01
module load cudnn #/8.8.0.121_cuda12-fasrc01
module load gcc #/13.2.0-fasrc01
module load Mambaforge

mamba deactivate
mamba activate protllm

python testing.py