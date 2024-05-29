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
mu.set_torch_seeds(seed)

"""
Instantiate dataset helper object
"""
length_limit = 4096 # Covers 99.75% sequences
model_name =  'esm2_t33_650M_UR50D' #'esm2_t36_3B_UR50D'
num_shards = 1 ###### Replace with appropriate number ###################################
data_utils = mu.DataUtils('../data', num_shards, model_name, length_limit, 'test', device, '_negative') # Running on old test for now

"""
Model choice based on user input
Not implemented as dictionary so that only one model is created
"""

classifier_clan = cf.ClanLSTM(data_utils.embedding_dim,data_utils.clan_count).to(device)
classifier_fam = cf.FamModelSimple(data_utils.embedding_dim, data_utils.maps, device).to(device)
classifier_clan.load_state_dict(torch.load('../data/results/clan_finetune/epoch_5.pth'))

classifier_path = Path(sys.argv[2])
save_path = classifier_path.parent / f'predictions_{model_name}'
os.makedirs(save_path, exist_ok=True)

classifier_fam.load_state_dict(torch.load(classifier_path))

"""
Testing loop
"""

shard = 1

dataset = data_utils.get_dataset(shard)

data_loader = data_utils.get_dataloader(dataset)

shard_preds, n_batches = mu.test_stepNegatives(data_loader, 
                                            classifier_clan,
                                            classifier_fam,
                                            device,
                                            data_utils,
                                            None)

"""
Save all predictions
"""

with open(save_path / f'negative_results_c5.pkl', 'wb') as f:
    pickle.dump(shard_preds, f)