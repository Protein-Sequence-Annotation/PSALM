#!/bin/bash

#SBATCH -J beta_multitrain
#SBATCH -p eddy
#SBATCH -c 16
#SBATCH -t 7-00:00
#SBATCH -o logs/beta_multitrain_inf.out
#SBATCH -e logs/beta_multitrain_inf.err
#SBATCH --mem 60000 #80GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:4

module load cuda #/12.0.1-fasrc01
module load cudnn #/8.8.0.121_cuda12-fasrc01
module load gcc #/13.2.0-fasrc01
module load Mambaforge

mamba deactivate
mamba activate protllm

python psalm_train_multi.py -m fam -o batch_fam -ne 10 -lr 0.001 -x _PSALM_1b -nl