import numpy as np
import pickle
from pathlib import Path
import os, sys
from tqdm import tqdm
import wandb

import torch
from torch import nn

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
mu.set_seeds(seed)

"""
Instantiate dataset helper object
"""
length_limit = 4096 # Covers 99.75% sequences
model_name =  'esm2_t33_650M_UR50D' #'esm2_t36_3B_UR50D'
num_shards = 50 ###### Replace with appropriate number ###################################
data_utils = mu.DataUtils('../data', num_shards, model_name, length_limit, 'test', device)

"""
Model choice based on user input
Not implemented as dictionary so that only one model is created
"""

if sys.argv[1] == 'CLN3': # Clan - 3 Linear Norm head
    classifier = cf.LinearHead3Normed(data_utils.embedding_dim, 2*data_utils.embedding_dim,data_utils.clan_count).to(device)
elif sys.argv[1] == 'CLN4': # Clan - 4 Linear Norm head
    classifier = cf.LinearHead4Normed(data_utils.embedding_dim, 2*data_utils.embedding_dim,data_utils.clan_count).to(device)
elif sys.argv[1] == 'CMLN3': # Clan - lstM 3 Linear Norm head
    classifier = cf.ClanLSTM(data_utils.embedding_dim,data_utils.clan_count).to(device)
elif sys.argv[1] == 'FamMoE':
    classifier = cf.FamModelMoE(data_utils.embedding_dim, data_utils.maps, device).to(device)
elif sys.argv[1] == 'FamSimple':
    classifier = cf.FamModelSimple(data_utils.embedding_dim, data_utils.maps, device).to(device)
elif sys.argv[1] == 'FamMoELSTM':
    classifier = cf.FamModelMoELSTM(data_utils.embedding_dim, data_utils.maps, device).to(device)
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

"""
Testing loop
"""
    
for shard in tqdm(range(1, data_utils.num_shards+1) ,total=data_utils.num_shards, desc='Shards completed'):

    hmm_dict = data_utils.parse_shard(shard)
    dataset = data_utils.get_dataset(shard)
    dataset = data_utils.filter_batches(dataset, hmm_dict.keys())

    data_loader = data_utils.get_dataloader(dataset)

    shard_preds, n_batches = mu.test_step(data_loader, ###########################
                                                classifier,
                                                device,
                                                data_utils,
                                                hmm_dict)
    
    with open(save_path / f'shard_{shard}.pkl', 'wb') as f:
        pickle.dump(shard_preds, f)

    print(f'Shard {shard} predictions saved')

"""
Visualize results
"""

# vz.clan_accuracies(save_path)
    