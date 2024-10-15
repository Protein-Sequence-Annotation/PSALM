#!/bin/bash

#SBATCH -J test
#SBATCH -p eddy
#SBATCH -c 16
#SBATCH -t 7-00:00
#SBATCH -o logs/full_1b.out
#SBATCH -e logs/full_1b.err
#SBATCH --mem 180000 #180GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:4

module load cuda #/12.0.1-fasrc01
module load cudnn #/8.8.0.121_cuda12-fasrc01
module load gcc #/13.2.0-fasrc01
module load Mambaforge

mamba deactivate
mamba activate protllm

py_dir = "py_scripts"

# python psalm_train_multi.py -m fam -o fam_1b_val_acc -ne 100 -lr 0.00005 -x _1b_final
python ${py_dir}/psalm_train_multi.py -m clan -o clan_1b_val_acc -ne 100 -lr 0.0005 -x _1b_final
# python psalm_train_multi.py -m only -o only_fams_lr_5e5_full_35M -ne 1000 -lr 0.00005 -es t12_35M
# python psalm_train_multi.py -m only -o only_fams_lr_5e5_full_8M -ne 1000 -lr 0.00005 -es t6_8M
# python psalm_train_multi.py -m only -o only_fams_lr_5e5_full_oh2 -ne 1000 -lr 0.00005


# python psalm_train_multi.py -m fam -o fam_oh_matched_lr_1e3 -ne 5 -lr 0.001
# python psalm_train_multi.py -m fam -o fam_oh_matched_lr_5e4 -ne 5 -lr 0.0005
# python psalm_train_multi.py -m fam -o fam_oh_matched_lr_1e4 -ne 5 -lr 0.0001
# python psalm_train_multi.py -m fam -o fam_lr_5e5_full_8M -ne 1000 -lr 0.00005
# python psalm_train_multi.py -m fam -o fam_oh_matched_lr_1e5 -ne 5 -lr 0.00001

# python psalm_test_multi.py -m eval -o test_onlyfams_150M -cf results/clan_lr_5e4_full_150M/epoch_7.pth -ff results/only_fams_lr_5e5_full_150M/epoch_9.pth -x _20 -nl -nv -es t30_150M
# python psalm_test_multi.py -m eval -o test_onlyfams_150M -cf results/clan_lr_5e4_full_150M/epoch_7.pth -ff results/only_fams_lr_5e5_full_150M/epoch_9.pth -x _40 -nl -nv -es t30_150M
# python psalm_test_multi.py -m eval -o test_onlyfams_150M -cf results/clan_lr_5e4_full_150M/epoch_7.pth -ff results/only_fams_lr_5e5_full_150M/epoch_9.pth -x _60 -nl -nv -es t30_150M
# python psalm_test_multi.py -m eval -o test_onlyfams_150M -cf results/clan_lr_5e4_full_150M/epoch_7.pth -ff results/only_fams_lr_5e5_full_150M/epoch_9.pth -x _80 -nl -nv -es t30_150M
# python psalm_test_multi.py -m eval -o test_onlyfams_150M -cf results/clan_lr_5e4_full_150M/epoch_7.pth -ff results/only_fams_lr_5e5_full_150M/epoch_9.pth -x _100 -nl -nv -es t30_150M

# python psalm_test_multi.py -m eval -o test_onlyfams_35M -cf results/clan_lr_5e4_full_35M/epoch_7.pth -ff results/only_fams_lr_5e5_full_35M/epoch_14.pth -x _20 -nl -nv -es t12_35M
# python psalm_test_multi.py -m eval -o test_onlyfams_35M -cf results/clan_lr_5e4_full_35M/epoch_7.pth -ff results/only_fams_lr_5e5_full_35M/epoch_14.pth -x _40 -nl -nv -es t12_35M
# python psalm_test_multi.py -m eval -o test_onlyfams_35M -cf results/clan_lr_5e4_full_35M/epoch_7.pth -ff results/only_fams_lr_5e5_full_35M/epoch_14.pth -x _60 -nl -nv -es t12_35M
# python psalm_test_multi.py -m eval -o test_onlyfams_35M -cf results/clan_lr_5e4_full_35M/epoch_7.pth -ff results/only_fams_lr_5e5_full_35M/epoch_14.pth -x _80 -nl -nv -es t12_35M
# python psalm_test_multi.py -m eval -o test_onlyfams_35M -cf results/clan_lr_5e4_full_35M/epoch_7.pth -ff results/only_fams_lr_5e5_full_35M/epoch_14.pth -x _100 -nl -nv -es t12_35M

# python psalm_test_multi.py -m eval -o test_onlyfams_8M -cf results/clan_lr_5e4_full_8M/epoch_7.pth -ff results/only_fams_lr_5e5_full_8M/epoch_14.pth -x _20 -nl -nv -es t6_8M
# python psalm_test_multi.py -m eval -o test_onlyfams_8M -cf results/clan_lr_5e4_full_8M/epoch_7.pth -ff results/only_fams_lr_5e5_full_8M/epoch_14.pth -x _40 -nl -nv -es t6_8M
# python psalm_test_multi.py -m eval -o test_onlyfams_8M -cf results/clan_lr_5e4_full_8M/epoch_7.pth -ff results/only_fams_lr_5e5_full_8M/epoch_14.pth -x _60 -nl -nv -es t6_8M
# python psalm_test_multi.py -m eval -o test_onlyfams_8M -cf results/clan_lr_5e4_full_8M/epoch_7.pth -ff results/only_fams_lr_5e5_full_8M/epoch_14.pth -x _80 -nl -nv -es t6_8M
# python psalm_test_multi.py -m eval -o test_onlyfams_8M -cf results/clan_lr_5e4_full_8M/epoch_7.pth -ff results/only_fams_lr_5e5_full_8M/epoch_14.pth -x _100 -nl -nv -es t6_8M

# python psalm_test_multi.py -m eval -o test_onlyfams_oht -cf results/clan_oh_lr_1e4_full/epoch_7.pth -ff results/only_fams_lr_5e5_full_oh/epoch_14.pth -x _20 -nl -nv
# python psalm_test_multi.py -m eval -o test_onlyfams_oh -cf results/clan_oh_lr_1e4_full/epoch_7.pth -ff results/only_fams_lr_5e5_full_oh/epoch_14.pth -x _40 -nl -nv
# python psalm_test_multi.py -m eval -o test_onlyfams_oh -cf results/clan_oh_lr_1e4_full/epoch_7.pth -ff results/only_fams_lr_5e5_full_oh/epoch_14.pth -x _60 -nl -nv
# python psalm_test_multi.py -m eval -o test_onlyfams_oh -cf results/clan_oh_lr_1e4_full/epoch_7.pth -ff results/only_fams_lr_5e5_full_oh/epoch_14.pth -x _80 -nl -nv
# python psalm_test_multi.py -m eval -o test_onlyfams_oh -cf results/clan_oh_lr_1e4_full/epoch_7.pth -ff results/only_fams_lr_5e5_full_oh/epoch_14.pth -x _100 -nl -nv