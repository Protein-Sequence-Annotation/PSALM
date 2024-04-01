import numpy as np
from pathlib import Path
import os, sys
from tqdm import tqdm
import wandb

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
mu.set_torch_seeds(seed)

"""
Instantiate dataset helper object
"""
length_limit = 4096 # Covers 99.75% sequences
model_name =  'esm2_t33_650M_UR50D' 
# model_name = 'esm2_t36_3B_UR50D'
num_shards = 50
data_utils = mu.DataUtils('../data', num_shards, model_name, length_limit, 'train', device)
data_utils.maps['clan_family_matrix'].to(device)

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
elif sys.argv[1] == 'CMLN3_onehot': # Clan - lstM 3 Linear Norm head
    classifier = cf.ClanLSTM(len(data_utils.onehot_alphabet),data_utils.clan_count).to(device)
elif sys.argv[1] == "CMLN3_onehot_dim_matched":
    classifier = cf.ClanLSTM_onehot_dim_matched(len(data_utils.onehot_alphabet),data_utils.clan_count).to(device)
elif sys.argv[1] == 'FamMoE':
    classifier = cf.FamModelMoE(data_utils.embedding_dim, data_utils.maps, device).to(device)
elif sys.argv[1] == 'FamSimple':
    classifier = cf.FamModelSimple(data_utils.embedding_dim, data_utils.maps, device).to(device)
elif sys.argv[1] == "FamSimple_onehot_dim_matched":
    classifier = cf.FamModelSimple_onehot_dim_matched(len(data_utils.onehot_alphabet), data_utils.maps, device).to(device)
elif sys.argv[1] == 'FamMoELSTM':
    classifier = cf.FamModelMoELSTM(data_utils.embedding_dim, data_utils.maps, device).to(device)
elif sys.argv[1] == 'ClanFamSimple':
    classifier = cf.ClanFamLSTM(data_utils.embedding_dim, data_utils.clan_count, data_utils.maps, device).to(device)
else:
    print('Incorrect Model choice')
    sys.exit(2)

resume = False
if resume:
    classifier_path = Path(f'../data/results/no_l1_part3/epoch_5.pth')
    classifier.load_state_dict(torch.load(classifier_path))

"""
Parameters for training loop
"""

loss_fn = nn.CrossEntropyLoss() ############ Changed for weighted LSTM
# loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
lr = 0.001
optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.1, threshold_mode="rel") # lower LR if less than 10% decrease

num_epochs = 30
save_path = Path(f'../data/results/{sys.argv[2]}')
os.makedirs(save_path, exist_ok=True)

"""
Initialize wandb
"""
run = wandb.init(project='esm2-linear3', 
                 entity='eddy_lab',
                 config={"epochs": num_epochs,
                         "lr": lr,
                         "Architecture": "one_hot basic",
                         "dataset": 'Pfam Seed'})

"""
Training loop
"""

shard_seed = 42
shard_gen = np.random.default_rng(shard_seed)

for epoch in range(num_epochs):

    epoch_loss = 0
    
    shard_order = np.arange(1,51)
    # shard_gen.shuffle(shard_order) # Use this for reprodcibility of shard order
    np.random.shuffle(shard_order)

    for shard in tqdm(shard_order ,total=data_utils.num_shards, desc='Shards completed'):

        hmm_dict = data_utils.parse_shard(shard)
        dataset = data_utils.get_dataset(shard)
        dataset = data_utils.filter_batches(dataset, hmm_dict.keys())

        data_loader = data_utils.get_dataloader(dataset)

        shard_loss, n_batches = mu.train_stepFamOneHot(data_loader, ###########################
                                                  classifier,
                                                  loss_fn,
                                                  optimizer,
                                                  device,
                                                  data_utils,
                                                  hmm_dict)
        
        epoch_loss += shard_loss # (shard_loss / n_batches)
        
        print(f'Epoch {epoch} Shard {shard} Loss {shard_loss / n_batches}')
        wandb.log({'Shard loss': shard_loss / n_batches})
    
    print(f'Epoch {epoch} Loss {epoch_loss / data_utils.num_shards}')
    print('------------------------------------------------')
    wandb.log({'Epoch loss': epoch_loss / data_utils.num_shards, 'Learning rate': optimizer.param_groups[0]['lr']})

    torch.save(classifier.state_dict(), save_path / f'epoch_{epoch}.pth')
    scheduler.step(epoch_loss / data_utils.num_shards)