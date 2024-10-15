import pickle
from pathlib import Path
import os
from tqdm import tqdm

import torch
import ml_utils as mu
import classifiers as cf
from parsers import test_parser
from functools import partial

"""
Parser Initialization
"""

parser = test_parser()
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
num_shards = args.num_shards # Number of shards
data_utils = mu.DataUtils(args.root, num_shards, esm_model_name, length_limit, 'test', device, alt_suffix=args.suffix) # Running on old test for now

"""
Model choice based on user input
Not implemented as dictionary so that only one model is created
"""
if args.model == 'clan':

    classifier = cf.ClanLSTM(data_utils.embedding_dim,data_utils.clan_count).to(device)

    classifier_path = Path(args.clan_filename)

    classifier.load_state_dict(args.clan_filename)

    test_fn = partial(mu.test_step_clan, classifier=classifier)

else:
    classifier_clan = cf.ClanLSTM(data_utils.embedding_dim,data_utils.clan_count).to(device)
    classifier_fam = cf.FamLSTM(data_utils.embedding_dim, data_utils.maps, device).to(device)

    classifier_clan.load_state_dict(torch.load(args.clan_filename)) # Clan model path

    classifier_path = Path(args.fam_filename)

    classifier_fam.load_state_dict(torch.load(args.fam_filename))

    test_fn = partial(mu.test_step_fam, classifier_clan=classifier_clan, classifier_fam=classifier_fam)

save_path = classifier_path.parent / f'predictions_{args.model}' # Edit for different save locations
os.makedirs(save_path, exist_ok=True)
    
"""
Testing loop
"""

all_preds = {}
    
for shard in tqdm(range(1, data_utils.num_shards+1) ,total=data_utils.num_shards, desc='Shards completed'):

    hmm_dict = data_utils.parse_shard(shard) 
    dataset = data_utils.get_dataset(shard)
    dataset = data_utils.filter_batches(dataset, hmm_dict.keys())

    data_loader = data_utils.get_dataloader(dataset)

    shard_preds, n_batches = test_fn(data_loader = data_loader, 
                                                device = device,
                                                data_utils = data_utils,
                                                hmm_dict = hmm_dict)
         
    # Save all predictions for this shard

    with open(save_path / f'{args.model}_shard_{shard}.pkl', 'wb') as f:
        pickle.dump(shard_preds, f)


    print(f'Shard {shard} predictions processed')