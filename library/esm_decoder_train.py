import numpy as np
from pathlib import Path
import os, sys
from tqdm import tqdm
import wandb

import torch
from torch import nn

import ml_utils as mu
import classifiers as cf

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
num_shards = 50
data_utils = mu.DataUtils('../data', num_shards, model_name, length_limit, 'train', device)
data_utils.maps['clan_family_matrix'].to(device)

"""
Model choice based on user input
Not implemented as dictionary so that only one model is created
"""

if sys.argv[1] == 'CL1': # Clan - 1 linear head
    classifier = cf.LinearHead1(data_utils.embedding_dim, data_utils.clan_count).to(device)
elif sys.argv[1] == 'CL3': # Clan - 3 linear heads
    classifier = cf.LinearHead3(data_utils.embedding_dim, 2*data_utils.embedding_dim, data_utils.clan_count).to(device)
elif sys.argv[1] == 'CLN3': # Clan - 3 Linear Norm head
    classifier = cf.LinearHead3Normed(data_utils.embedding_dim, 2*data_utils.embedding_dim,data_utils.clan_count).to(device)
elif sys.argv[1] == 'CLN4': # Clan - 4 Linear Norm head
    classifier = cf.LinearHead4Normed(data_utils.embedding_dim, 2*data_utils.embedding_dim,data_utils.clan_count).to(device)
elif sys.argv[1] == 'FLN3': # Family - 3 Linear Norm head
    classifier = cf.LinearHead3NormedFam(data_utils.embedding_dim,data_utils.clan_count,data_utils.fam_count,1, attend=False).to(device)
elif sys.argv[1] == 'FLN3A': # Family - 3 Linear Norm head With Attention
    classifier = cf.LinearHead3NormedFam(data_utils.embedding_dim,data_utils.clan_count,data_utils.fam_count,1).to(device)
elif sys.argv[1] == 'FLN3S': # Family - 3 Linear Norm head With Softmax
    classifier = cf.LinearHead3NormedFamSoft(data_utils.embedding_dim,data_utils.clan_count,data_utils.fam_count).to(device)
elif sys.argv[1] == 'CLWC': # Clan - Linear Weighted Context
    classifier = cf.ContextWeightedSum(data_utils.embedding_dim, data_utils.clan_count).to(device)
elif sys.argv[1] == 'CLC': # Clan - Concatenated Linear
    classifier = cf.ContextConcatLinear3(data_utils.embedding_dim, data_utils.clan_count).to(device)
elif sys.argv[1] == 'CMLN3': # Clan - lstM 3 Linear Norm head
    classifier = cf.TryLSTM(data_utils.embedding_dim,data_utils.clan_count).to(device)
elif sys.argv[1] == 'FamMoE':
    classifier = cf.FamModelMoE(data_utils.embedding_dim, data_utils.maps, device).to(device)
elif sys.argv[1] == 'FamSimple':
    classifier = cf.FamModelSimple(data_utils.embedding_dim, data_utils.maps, device).to(device)
elif sys.argv[1] == 'FamMoELSTM':
    classifier = cf.FamModelMoELSTM(data_utils.embedding_dim, data_utils.maps, device).to(device)
else:
    print('Incorrect Model choice')
    sys.exit(2)

resume = True
if resume:
    classifier_path = Path(f'../data/results/Fam_Simple_run2/epoch_3.pth')
    classifier.load_state_dict(torch.load(classifier_path))

"""
Parameters for training loop
"""

loss_fn = nn.CrossEntropyLoss() ############ Changed for weighted LSTM
# loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0001) ############### Changed for LSTM continuation!!!!
num_epochs = 5
save_path = Path(f'../data/results/{sys.argv[2]}')
os.makedirs(save_path, exist_ok=True)

"""
Initialize wandb
"""
run = wandb.init(project='esm2-linear3', 
                 entity='eddy_lab',
                 config={"epochs": num_epochs,
                         "lr": 1e-3,
                         "Architecture": "Simple Fam w/ non IDR",
                         "dataset": 'Pfam Seed'})

"""
Training loop
"""

for epoch in range(num_epochs):

    epoch_loss = 0
    
    for shard in tqdm(range(1, data_utils.num_shards+1) ,total=data_utils.num_shards, desc='Shards completed'):

        hmm_dict = data_utils.parse_shard(shard)
        dataset = data_utils.get_dataset(shard)
        dataset = data_utils.filter_batches(dataset, hmm_dict.keys())

        data_loader = data_utils.get_dataloader(dataset)

        shard_loss, n_batches = mu.train_stepFamSimple(data_loader, ###########################
                                                  classifier,
                                                  loss_fn,
                                                  optimizer,
                                                  device,
                                                  data_utils,
                                                  hmm_dict)
        
        epoch_loss += shard_loss
        
        print(f'Epoch {epoch} Shard {shard} Loss {shard_loss / n_batches}')
        wandb.log({'Shard loss': shard_loss / n_batches})
    
    print(f'Epoch {epoch} Loss {epoch_loss / data_utils.num_shards}')
    print('------------------------------------------------')
    wandb.log({'Epoch loss': epoch_loss / data_utils.num_shards})

    torch.save(classifier.state_dict(), save_path / f'epoch_{epoch}.pth')