import numpy as np
import pickle
from pathlib import Path
import os, sys
from tqdm import tqdm
import wandb

import torch
from torch import nn
import hmmscan_utils as hu
import ml_utils as mu
import classifiers as cf
# import visualizations as vz

"""
Set device for execution
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

"""
For reproducible code
"""
seed = 42 # Because ...
mu.set_torch_seeds(seed)

"""
Instantiate dataset helper object
"""
length_limit = 4096 # Covers 99.75% sequences
model_name =  'esm2_t33_650M_UR50D' #'esm2_t36_3B_UR50D'
num_shards = 50 ###### Replace with appropriate number ###################################
data_utils = mu.DataUtils('../data', num_shards, model_name, length_limit, 'test', device, '_100_shuffled') # Running on old test for now

"""
Model choice based on user input
Not implemented as dictionary so that only one model is created
"""
if sys.argv[3] not in ['joint', 'jointp', 'roc']:
    if sys.argv[1] == 'CLN3': # Clan - 3 Linear Norm head
        classifier = cf.LinearHead3Normed(data_utils.embedding_dim, 2*data_utils.embedding_dim,data_utils.clan_count).to(device)
    elif sys.argv[1] == 'CLN4': # Clan - 4 Linear Norm head
        classifier = cf.LinearHead4Normed(data_utils.embedding_dim, 2*data_utils.embedding_dim,data_utils.clan_count).to(device)
    elif sys.argv[1] == 'CMLN3': # Clan - lstM 3 Linear Norm head
        classifier = cf.ClanLSTM(data_utils.embedding_dim,data_utils.clan_count).to(device)
    elif sys.argv[1] == 'CMLN3_onehot': # Clan - lstM 3 Linear Norm head
        classifier = cf.ClanLSTM(len(data_utils.onehot_alphabet),data_utils.clan_count).to(device)
    elif sys.argv[1] == "CMLN3_onehot_dim_matched":
        classifier = cf.ClanLSTM_onehot_dim_matched(len(data_utils.onehot_alphabet),data_utils.clan_count).to(device)
    elif sys.argv[1] == 'FamMoE':
        classifier = cf.FamModelMoE(data_utils.embedding_dim, data_utils.maps, device).to(device)
    elif sys.argv[1] == 'FamSimple':
        classifier = cf.FamModelSimple(data_utils.embedding_dim, data_utils.maps, device).to(device)
    elif sys.argv[1] == 'FamMoELSTM':
        classifier = cf.FamModelMoELSTM(data_utils.embedding_dim, data_utils.maps, device).to(device)
    elif sys.argv[1] == "FamSimple_onehot_dim_matched":
        classifier = cf.FamModelSimple_onehot_dim_matched(len(data_utils.onehot_alphabet), data_utils.maps, device).to(device)
    elif sys.argv[1] == 'ClanFamSimple':
        classifier = cf.ClanFamLSTM(data_utils.embedding_dim, data_utils.clan_count, data_utils.maps, device).to(device)
    else:
        print('Incorrect Model choice')
        sys.exit(2)

    """
    Parameters for test loop
    """

    classifier_path = Path(sys.argv[2])
    save_path = classifier_path.parent / f'predictions_{model_name}'
    os.makedirs(save_path, exist_ok=True)

    classifier.load_state_dict(torch.load(classifier_path))

else:
    # classifier_clan = cf.ClanLSTM(data_utils.embedding_dim,data_utils.clan_count).to(device)
    # classifier_fam = cf.FamModelSimple(data_utils.embedding_dim, data_utils.maps, device).to(device)
    classifier_clan = cf.ClanLSTM_onehot_dim_matched(len(data_utils.onehot_alphabet),data_utils.clan_count).to(device)
    classifier_fam = cf.FamModelSimple_onehot_dim_matched(len(data_utils.onehot_alphabet), data_utils.maps, device).to(device)
    # classifier_clan.load_state_dict(torch.load('../data/results/clan_finetune/epoch_5.pth'))
    classifier_clan.load_state_dict(torch.load('../data/results/clan_onehot_dm_ft/epoch_4.pth'))

    classifier_path = Path(sys.argv[2])
    # save_path = classifier_path.parent / f'predictions_{model_name}'
    # save_path = Path('../data/results/onehot_fts/')
    # os.makedirs(save_path, exist_ok=True)

    classifier_fam.load_state_dict(torch.load(classifier_path))
    
"""
Testing loop
"""

all_preds = {}
    
for shard in tqdm(range(1, data_utils.num_shards+1) ,total=data_utils.num_shards, desc='Shards completed'):

    hmm_dict = data_utils.parse_shard(shard) 
    dataset = data_utils.get_dataset(shard)
    dataset = data_utils.filter_batches(dataset, hmm_dict.keys())

    data_loader = data_utils.get_dataloader(dataset)

    if sys.argv[3] == 'fam':
        shard_preds, n_batches = mu.test_stepFam(data_loader, 
                                                    classifier,
                                                    device,
                                                    data_utils,
                                                    hmm_dict)
    elif sys.argv[3] == 'clan':
        shard_preds, n_batches = mu.test_stepClan(data_loader, 
                                                    classifier,
                                                    device,
                                                    data_utils,
                                                    hmm_dict)
    elif sys.argv[3] == 'joint':
        shard_preds, n_batches = mu.test_stepFamJoint(data_loader, 
                                                    classifier_clan,
                                                    classifier_fam,
                                                    device,
                                                    data_utils,
                                                    hmm_dict)
    elif sys.argv[3] == 'onehotf':
        shard_preds, n_batches = mu.test_stepFamOneHot(data_loader, 
                                                    classifier,
                                                    device,
                                                    data_utils,
                                                    hmm_dict)
    elif sys.argv[3] == 'onehotc':
        shard_preds, n_batches = mu.test_stepClanOneHot(data_loader, 
                                                    classifier,
                                                    device,
                                                    data_utils,
                                                    hmm_dict)
    elif sys.argv[3] == 'jointp':
        shard_preds, n_batches = mu.test_stepJointPreds(data_loader, 
                                                    classifier_clan,
                                                    classifier_fam,
                                                    device,
                                                    data_utils,
                                                    hmm_dict)
    elif sys.argv[3] == 'roc':
        pred_dict = hu.parse_hmmscan_results(f'../data/test_hmmer_preds_100_shuffled/split_{shard}_test_ids_full.fasta_scan.txt',
                                             e_value_threshold=1000,
                                             score_threshold=0.0,
                                             pred=True)
        shard_preds, n_batches = mu.test_stepROC_OH(data_loader, 
                                                    classifier_clan,
                                                    classifier_fam,
                                                    device,
                                                    data_utils,
                                                    hmm_dict,
                                                    pred_dict)
    else:
        print('Invalid argument')
        sys.exit(2)
        
    # if sys.argv[3] == 'jointp':
    #     with open(save_path / f'xshard_{shard}.pkl', 'wb') as f:
    #         pickle.dump(shard_preds, f)
    # elif sys.argv[3] == 'roc':
        # with open(save_path / f'{sys.argv[3]}_results_roc_full_{shard}.pkl', 'wb') as f:
        #     pickle.dump(shard_preds, f)
        # with open('../data/results/Clan_finetune/{sys.argv[3]}_results_roc_2.pkl', 'wb') as f:
        #     pickle.dump(shard_preds, f)
    # else:
    all_preds = all_preds | shard_preds

    print(f'Shard {shard} predictions processed')
    

"""
Save all predictions
"""
if sys.argv[3] != 'jointp':
#     with open(save_path / f'{sys.argv[3]}_results_roc_full.pkl', 'wb') as f:
#         pickle.dump(all_preds, f)
    # with open(f'../data/results/clan_finetune/{sys.argv[3]}_results_20_nu.pkl', 'wb') as f:
    #     pickle.dump(all_preds, f)
    with open(f'../data/results/onehot_fts/{sys.argv[3]}_results_100.pkl', 'wb') as f:
        pickle.dump(all_preds, f)