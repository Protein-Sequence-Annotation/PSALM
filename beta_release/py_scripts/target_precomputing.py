import hmmscan_utils as hu
import pickle
import numpy as np
import torch
import sys

def main(mode, shard_num):

    # db_path = f'datasets/{mode}_{shard_num}.pkl' # For train
    db_path = f'../dev/data/full_data_curation/PSALM_1b_Test_Scan_FINAL.pkl' # For validation

    all_targets = {}

    with open('info_files/maps.pkl', 'rb') as f:
        maps = pickle.load(f)

    with open(db_path, 'rb') as f:
        hmm_dict = pickle.load(f)

    for idx, key in enumerate(hmm_dict.keys()):

        stop_index = min(hmm_dict[key]['length'], 4096)
        fam_vector, clan_vector = hu.generate_domain_position_list(hmm_dict, key, maps)
        fam_vector = np.argmax(fam_vector, axis=1)[:stop_index]
        clan_vector = np.argmax(clan_vector, axis=1)[:stop_index]

        all_targets[key] = {}
        all_targets[key]['stop'] = stop_index
        all_targets[key]['fam_vector'] = torch.tensor(fam_vector, dtype=torch.long)
        all_targets[key]['clan_vector'] = torch.tensor(clan_vector, dtype=torch.long)
    
    # with open(f'datasets/{mode}_targets_{shard_num}.pkl', 'wb') as f: # For train
    with open(f'datasets/{mode}_targets_1b.pkl', 'wb') as f: # For validation
        pickle.dump(all_targets, f)

    return

def sharder(mode, shards):

    db_path = f'datasets/{mode}.pkl'

    with open(db_path, 'rb') as f:
        hmm_dict = pickle.load(f)

    total = len(hmm_dict)
    each = int(total // shards)        

    shard_dict = {}
    ctr = 0
    shard_num = 0

    for key in hmm_dict.keys():
        shard_dict[key] = hmm_dict[key]
        ctr += 1

        if ctr == each:
            with open(f'datasets/{mode}_{shard_num}.pkl', 'wb') as f:
                pickle.dump(shard_dict, f)

            ctr = 0
            shard_num += 1
            shard_dict = {}

    if ctr > 0:
        with open(f'datasets/{mode}_{shard_num}.pkl', 'wb') as f:
            pickle.dump(shard_dict, f)

    return

def merger(mode, shards):

    all_targets = {}

    for i in range(shards):

        with open(f'datasets/{mode}_targets_{i}.pkl', 'rb') as f:
            shard_dict = pickle.load(f)

        for key in shard_dict:
            all_targets[key] = shard_dict[key]

    with open(f'datasets/{mode}_targets.pkl', 'wb') as f:
        pickle.dump(all_targets, f)

    return

if __name__ == '__main__':

    mode = 'Test'
    # shard_num = int(sys.argv[1])
    shard_num = 100 # Dummy for validation
    main(mode, shard_num)
    # sharder(mode, 100)
    # merger(mode, 101)