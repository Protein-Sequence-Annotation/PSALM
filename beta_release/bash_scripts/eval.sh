#!/bin/bash

#SBATCH -J OnlyFams
#SBATCH -p eddy
#SBATCH -c 8
#SBATCH -t 0-10:00
#SBATCH -o logs/OnlyFams__eval.out
#SBATCH -e logs/OnlyFams__eval.err
#SBATCH --mem 40000 

module load cuda #/12.0.1-fasrc01
module load cudnn #/8.8.0.121_cuda12-fasrc01
module load gcc #/13.2.0-fasrc01
module load Mambaforge

mamba deactivate
mamba activate protllm

py_dir = "py_scripts"

# python eval.py results/test_onlyfams_8M
# python eval.py results/test_onlyfams_35M
# python eval.py results/test_onlyfams_150M
# python eval.py results/test_onlyfams_650M
python ${py_dir}/eval.py results/test_onlyfams_oht


