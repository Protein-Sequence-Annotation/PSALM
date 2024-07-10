from pathlib import Path

import torch
import torch.multiprocessing as mp

from parsers import train_parser
import ml_utils as mu
import distributed_utils as du
import os
import pickle

if __name__ == '__main__':
    
    # Parser Initialization

    parser = train_parser()
    args = parser.parse_args()

    # Set device for execution

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}, {num_gpus} GPUs')

    # For reproducible code

    if not args.no_seed:
        seed = 42 # Because ...
        mu.set_torch_seeds(seed)

    # Save path for result

    save_path = Path(f'results/{args.output}')
    os.makedirs(save_path, exist_ok=True)

    # Load scans for train and validation

    with open(f'{args.root}/PSALM_1b_train_scans.pkl', 'rb') as f: # Path to train scans pkl
        train_dict = pickle.load(f)

    with open(f'{args.root}/PSALM_1b_validation_scans.pkl', 'rb') as f: # Path to validation scans pkl
        val_dict = pickle.load(f)

    # Spawn processes and begin training

    mp.spawn(du.loadAll, args=(num_gpus, args, save_path, train_dict, val_dict), nprocs=num_gpus)
