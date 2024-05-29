import numpy as np
from pathlib import Path
import os, sys
from tqdm import tqdm
import wandb

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from parsers import train_parser

import ml_utils as mu
import classifiers as cf

"""
Parser Initialization
"""

parser = train_parser()
args = parser.parse_args()

"""
Set device for execution
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

"""
For reproducible code
"""

if args.seeded:
    seed = 42 # Because ...
    mu.set_torch_seeds(seed)

"""
Instantiate dataset helper object
"""
length_limit = 4096 # Covers 99.75% sequences
esm_model_name =  'esm2_t33_650M_UR50D' 

num_shards = args.num_shards
data_utils = mu.DataUtils(args.root, num_shards, esm_model_name, length_limit, 'train', device, alt_suffix = "_finetune")
data_utils.maps['clan_family_matrix'].to(device)

"""
Model choice based on user input
Not implemented as dictionary so that only one model is created
"""

if args.model == 'clan': # Clan - lstM 3 Linear Norm head
    classifier = cf.ClanLSTM(data_utils.embedding_dim,data_utils.clan_count).to(device)
    train_fn = mu.train_step_clan
elif args.model == 'fam':
    classifier = cf.FamLSTM(data_utils.embedding_dim, data_utils.maps, device).to(device)
    train_fn = mu.train_step_fam
else:
    print('Incorrect Model choice')
    sys.exit(2)

if args.resume != 'none':
    classifier_path = Path(f'{args.resume}')
    classifier.load_state_dict(torch.load(classifier_path))

"""
Parameters for training loop
"""

loss_fn = nn.CrossEntropyLoss() # fam loss is hard coded
lr = args.learning_rate # lr 1e-3 for train with L1,  1e-4 for train without L1, 1e-5 for fine tune
optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.1, threshold_mode="rel") # lower LR if less than 10% decrease

num_epochs = args.num_epochs
save_path = Path(f'results/{args.output}')
os.makedirs(save_path, exist_ok=True)

"""
Initialize wandb
"""
# For wandb logging - change project name and credentials

if args.log:
    run = wandb.init(project='esm2-linear3', 
                    entity='eddy_lab',
                    config={"epochs": num_epochs,
                            "lr": lr,
                            "Architecture": "Fam",
                            "dataset": 'Pfam Seed'})

"""
Training loop
"""

if args.seeded:
    shard_seed = 42
    shard_gen = np.random.default_rng(shard_seed)

for epoch in range(num_epochs):

    epoch_loss = 0
    
    shard_order = np.arange(1,num_shards+1)
    shard_gen.shuffle(shard_order) # Use this for reprodcibility of shard order
    np.random.shuffle(shard_order)

    for shard in tqdm(shard_order ,total=data_utils.num_shards, desc='Shards completed'):

        hmm_dict = data_utils.parse_shard(shard)
        dataset = data_utils.get_dataset(shard)
        dataset = data_utils.filter_batches(dataset, hmm_dict.keys())

        data_loader = data_utils.get_dataloader(dataset)

        shard_loss, n_batches = mu.train_fn(data_loader = data_loader,
                                                  classifier = classifier,
                                                  loss_fn = loss_fn,
                                                  optimizer = optimizer,
                                                  device = device,
                                                  data_utils = data_utils,
                                                  hmm_dict = hmm_dict)
        
        epoch_loss += (shard_loss / n_batches)
        
        print(f'Epoch {epoch} Shard {shard} Loss {shard_loss / n_batches}')
        
        if args.log:
            wandb.log({'Shard loss': shard_loss / n_batches})
    
    print(f'Epoch {epoch} Loss {epoch_loss / data_utils.num_shards}')
    print('------------------------------------------------')
    
    if args.log:
        wandb.log({'Epoch loss': epoch_loss / data_utils.num_shards, 'Learning rate': optimizer.param_groups[0]['lr']})

    torch.save(classifier.state_dict(), save_path / f'epoch_{epoch}.pth')
    scheduler.step(epoch_loss / data_utils.num_shards)