from pathlib import Path

import torch
import torch.multiprocessing as mp

from parsers import train_parser
import ml_utils as mu
import distributed_utils as du
import os, sys
import pickle

if __name__ == '__main__':

    # Parser Initialization

    parser = train_parser()
    args = parser.parse_args()

    # Set device for execution

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}, {num_gpus} GPUs')
    sys.stdout.flush()

    # Save path for result

    save_path = Path(f'results/{args.output}')
    os.makedirs(save_path, exist_ok=True)

    # Load scans for test

    test_dict_path = f'{args.root}/Test{args.suffix}_targets.pkl'

    # Spawn processes and begin training

    mp.spawn(du.loadAll, args=(num_gpus, args, save_path, None, None, test_dict_path), nprocs=num_gpus)

    