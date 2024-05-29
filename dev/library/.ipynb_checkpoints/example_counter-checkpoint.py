# Map between fam idx and examples seen
from collections import defaultdict
from Bio import SeqIO
import hmmscan_utils as hu
from tqdm import tqdm
import numpy as np
import pickle
import sys

with open(f'../data/maps.pkl', 'rb') as f:
    maps = pickle.load(f)

train_examples = defaultdict(int)

shard = sys.argv[1]

hmm_dict = hu.parse_hmmscan_results(f'../data/train_scan/split_{shard}_train_ids_full.fasta_scan.txt')

for key in hmm_dict.keys():

    target, _ = hu.generate_domain_position_list2(hmm_dict, key, maps)
    target = np.argmax(target[:min(target.shape[0], 4096), :], axis=1)

    fams = np.unique(target)
    for fam in fams:
        train_examples[fam] += 1

with open(f'../data/train_examples_{shard}.pkl', 'wb') as f:
    pickle.dump({'counts': train_examples}, f)