from Bio import SeqIO
from Bio.Seq import Seq
import os
import glob
import pickle
import random
from tqdm import tqdm
import sys
sys.path.insert(0, '../library')
import hmmscan_utils as hu

random.seed(100)

shard = sys.argv[1]

# Load maps
with open("../data/maps.pkl", "rb") as f:
    maps = pickle.load(f)

# Directory containing the fasta files
scan_path = "../data/train"
out_path = "../data/train_fasta_finetune"


# Get the corresponding scan
hmmscan_dict = hu.parse_hmmscan_results(f"{scan_path}_scan/split_{shard}_train_ids_full.fasta_scan.txt")

# Use BioPython to load the fasta file
sequences = list(SeqIO.parse(f"{scan_path}_fasta/split_{shard}_train_ids_full.fasta", "fasta"))

for seq in sequences:
    # print(seq.id)
    # Get clan_vector
    try:
        _, clan_vector = hu.generate_domain_position_list(hmmscan_dict, seq.id, maps)
    except:
        # print(f"Error with {seq.id}")
        continue
    # find the indices where the clan vector is 656
    indices = [i for i, x in enumerate(clan_vector) if x == 656]

    # If there are no clan 656 domains, skip
    if len(indices) == 0:
        continue

    # Shuffle seq.seq only where clan 656 is present
    sequence = list(seq.seq)
    values = [sequence[i] for i in indices]
    random.shuffle(values)
    for i, index in enumerate(indices):
        sequence[index] = values[i]
    seq.seq = Seq("".join(sequence))
    
# Write the shuffled sequences to a new fasta file
with open(f"{out_path}/split_{shard}_train_ids_full.fasta", "w") as f:
    SeqIO.write(sequences, f, "fasta")
