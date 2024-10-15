import pandas as pd
import pdb
import subprocess
from tqdm import tqdm
from Bio import SeqIO, AlignIO
from Bio.Seq import Seq
import pickle
import numpy as np
import random
from random import shuffle
import os
import sys
sys.path.insert(0, '../../library')
import hmmscan_utils as hu

def load_hmmstats(file_path):
    """
    Load HMM statistics from a file generated from the HMMER hmmstat command into a DataFrame.

    Parameters:
    file_path (str): The path to the file containing the HMM statistics.

    Returns:
    pandas.DataFrame: A DataFrame containing the loaded HMM statistics.

    """
    column_names = ['idx', 'name', 'accession', 'nseq', 'eff_nseq', 'M', 'relent', 'info', 'p relE', 'compKL']
    df = pd.read_csv(file_path, sep='\s+', comment='#', names=column_names)

    return df

def filter_families(hmmstats, min_sequence, nseq_max=2000):
    """
    Filter the families in the HMM statistics DataFrame based on a minimum number of sequences.

    Parameters:
    hmmstats (pandas.DataFrame): The DataFrame containing the HMM statistics.
    min_sequences (int): The minimum number of sequences a family must have to be included in the output.

    Returns:
    list: A list of family names that have at least `min_sequences` sequences.
    """
    # Filter the DataFrame
    filtered_df = hmmstats[(hmmstats['nseq'] > min_sequence) & (hmmstats['nseq'] <= nseq_max)]

    # Return the 'name' column as a list
    return filtered_df['name'].tolist()

def pick_sequence(filtered_families, save_path, pid_upper_bound=100, incE=0.01, nseq_max=2000):
    hmmfetch_command = '/n/eddy_lab/software/bin/hmmfetch'
    hmmsearch_command = '/n/eddy_lab/software/bin/hmmsearch'
    alipid_command = '/n/eddy_lab/software/bin/esl-alipid'
    hmm_db = '/n/eddy_lab/data/pfam-35.0/Pfam-A.hmm'
    seq_db = '../data/train_test_splitting/Pfam_A.seed.full_seqs.fasta'
    tmp_hmm = f"{save_path}/tmp_hmm.hmm"
    tmp_msa = f"{save_path}/tmp_msa.sto"
    tmp_alipid = f"{save_path}/tmp_alipid.txt"
    results = {}
    fam_hits = {}
    incE = incE/(19632/len(filtered_families)) # to make this match up with E-value for hmmscan E-value
    with open(f"{save_path}/seqname_fam_id.csv", "w") as f:
        f.write("# Sequence Max_%id Family\n")
        for fam in tqdm(filtered_families):
            subprocess.run([hmmfetch_command, "-o", tmp_hmm, hmm_db, fam], stdout=subprocess.DEVNULL, check=True)
            subprocess.run([hmmsearch_command,"--cpu",str(8), "-A", tmp_msa, "--incE", str(incE), tmp_hmm, seq_db], stdout=subprocess.DEVNULL, check=True)
            # Add a check that identifies how many sequences returned and then skip if more than some upper bound
            msa = list(SeqIO.parse(tmp_msa, "stockholm"))
            num_sequences = len(msa)
            with open(tmp_alipid, "w") as g:
                subprocess.run([alipid_command, tmp_msa], stdout=g, check=True)
            unique_seqnames = [x.id for x in msa]
            fam_hits[fam] = unique_seqnames
            if num_sequences > nseq_max:
                continue

            df = pd.read_csv(tmp_alipid, sep="\s+", usecols=[0,1,2])
            df.columns = ['seqname1', 'seqname2', '%id']
            unique_seqnames = pd.concat([df['seqname1'], df['seqname2']]).unique()

            # Initialize a dictionary to store the maximum %id for each sequence name
            max_ids = {}
            # For each unique sequence name
            for seqname in unique_seqnames:
                # Find all rows that have that sequence name in either column and pull the corresponding %id
                ids = df.loc[(df['seqname1'] == seqname) | (df['seqname2'] == seqname), '%id']
                # Store the maximum %id for this sequence name
                max_ids[seqname] = ids.max()
            # Convert the dictionary to a Series
            max_ids_series = pd.Series(max_ids)

            # Filter sequences with max %id <= pid_upper_bound
            filtered_max_ids_series = max_ids_series[max_ids_series <= pid_upper_bound]

            # Check if there are at least 5 sequences
            if len(filtered_max_ids_series) >= 5:
                # Get the sequence name and corresponding id of the highest max %id from the filtered sequences
                seqname_with_max_id = filtered_max_ids_series.idxmax()
                corresponding_id = filtered_max_ids_series.max()
            else:
                print("There are less than 5 sequences with max %id <= pid_upper_bound.")
                continue

            f.write(f"{seqname_with_max_id} {corresponding_id} {fam}\n")
            f.flush()

            if seqname_with_max_id in results:
                results[seqname_with_max_id].append((corresponding_id, fam))
            else:
                results[seqname_with_max_id] = [(corresponding_id, fam)]
    return results, fam_hits


def get_val_seqs():
    save_path = '../data/full_data_curation'
    nseq_threshold = 50
    nseq_max = 150
    max_pid = 30

    # Load the HMM statistics
    hmmstats = load_hmmstats("/n/eddy_lab/data/pfam-35.0/Pfam-A.hmm.stats")

     # Filter the families
    filtered_families = filter_families(hmmstats, nseq_threshold, nseq_max)

    results, fam_hits = pick_sequence(filtered_families, save_path, max_pid,nseq_max)
    
    with open(f"{save_path}/fam_hits.pkl", "wb") as f:
        pickle.dump(fam_hits, f)
    with open(f"{save_path}/results.pkl", "wb") as f:
        pickle.dump(results, f)

def check_seeds_for_val_seqs():
    save_path = '../data/full_data_curation'
    hmmstats = "/n/eddy_lab/data/pfam-35.0/Pfam-A.hmm.stats"
    afetch_command = "/n/eddy_lab/software/bin/esl-afetch"
    msa_db = '/n/eddy_lab/data/pfam-35.0/Pfam-A.seed'

    val_df = pd.read_csv(f'{save_path}/seqname_fam_id.csv',sep='\s+',comment='#')
    val_df.columns = ['seqname','pid','fam_id']
    val_df['seqname'] = val_df['seqname'].str.split('/').str[0]
    val_df_seqnames = val_df['seqname'].tolist()
    families = load_hmmstats(hmmstats)['name'].tolist()

    # Initialize the dictionary to store the results
    family_seq_occurrences = {}
    family_total_counts = {}
    flagged_families = []

    for family in tqdm(families):
        # Retrieve the MSA for the family using esl-afetch
        msa_file = "tmp.sto"  # Specify the output file name
        subprocess.run([afetch_command, "-o", msa_file, msa_db, family], stdout=subprocess.DEVNULL, check=True)
        
        # Parse the MSA file
        try:
            with open(msa_file, 'r', encoding='latin-1') as handle:
                alignment = AlignIO.read(handle, "stockholm")
        except:
            print(f"Error reading {family}")
            continue
        
        # Process each record in the alignment
        seqs_in_msa = []
        for record in alignment:
            record = str(record)
            # Extracting the accession code
            accession_start = record.find("/accession=") + len("/accession=")
            accession_end = record.find("\n", accession_start)
            accession_code = record[accession_start:accession_end]
            # print(accession_code)
            # break
            if accession_code in val_df_seqnames:
                seqs_in_msa.append(accession_code)
        
        # Store the information in the dictionary
        family_seq_occurrences[family] = seqs_in_msa
        family_total_counts[family] = len(alignment)
        
        # Initialize an empty list to store flagged families based on the new criteria

    for family, seqs in family_seq_occurrences.items():
        total_seqs = family_total_counts[family]
        num_hits = len(seqs)
        
        # Condition 1: Less than 20 seqs in alignment and at least one hit from val_df
        if total_seqs < 50 and num_hits > 0:
            flagged_families.append(family)
        # Condition 2: More than 50 seqs in alignment and remaining seqs after hits is less than 50
        elif total_seqs >= 50 and (total_seqs - num_hits) < 50:
            flagged_families.append(family)

    # At this point, family_seq_occurrences contains the required information
    # Save the results to a file
    with open(f"{save_path}/family_seq_occurrences.pkl", "wb") as f:
        pickle.dump(family_seq_occurrences, f)

    # Save the total counts to a file
    with open(f"{save_path}/family_total_counts.pkl", "wb") as f:
        pickle.dump(family_total_counts, f)

    # Save the flagged families to a file
    with open(f"{save_path}/flagged_families.txt", "w") as f:
        for family in flagged_families:
            f.write(f"{family}\n")

    unique_sequences = set()
    for seqs in family_seq_occurrences.values():
        unique_sequences.update(seqs)
    unique_sequences_list = list(unique_sequences)
    print(len(unique_sequences_list))

    # Initialize a set to track unique sequences
    unique_flagged_sequences = set()
    # Iterate through each family and its sequences
    for family, seqs in family_seq_occurrences.items():
        if family in flagged_families:
            # Add sequences to the set if the family is flagged
            unique_flagged_sequences.update(seqs)
    final_val_ids = list(unique_sequences.difference(unique_flagged_sequences))
    val_df = pd.read_csv(f'{save_path}/seqname_fam_id.csv',sep='\s+',comment='#')
    val_df.columns = ['seqname','pid','fam_id']

    results_dict = {}

    for final_seq in tqdm(final_val_ids):
        # Initialize the dictionary key with an empty list to store matches
        results_dict[final_seq] = []
        for index, row in val_df.iterrows():
            if final_seq in row['seqname']:
                # Append a tuple with the matching details to the list for this key
                results_dict[final_seq].append((row['seqname'], row['pid'], row['fam_id']))
    
    #save
    with open(f"{save_path}/final_val_seqs.pkl", "wb") as f:
        pickle.dump(results_dict, f)

def get_val_best_hits():
    save_path = '../data/full_data_curation'
    hmmscan_command = '/n/eddy_lab/software/bin/hmmscan'
    hmm_db = '/n/eddy_lab/data/pfam-35.0/Pfam-A.hmm'
    seq_db = '../data/train_test_splitting/Pfam_A.seed.full_seqs.fasta'

    # Load final_val_seqs.pkl, a dict where the keys are the sequence names
    with open(f"{save_path}/final_val_seqs.pkl", "rb") as f:
        val_seqs = pickle.load(f)
    
    # Use BioPython to load full db and retrieve only the val_seq sequences to create a new fasta file
    full_db = list(SeqIO.parse(seq_db, "fasta"))
    val_db = [seq for seq in full_db if seq.id in val_seqs.keys()]
    val_db_fasta = f"{save_path}/val_pre_scan_db.fasta"
    with open(val_db_fasta, "w") as f:
        SeqIO.write(val_db, f, "fasta")
    print("Val db fasta created")

    # Run hmmscan on the val_db_fasta
    val_hmmscan_output = f"{save_path}/val_hmmscan_output.txt"
    subprocess.run([hmmscan_command,"--acc", "--cpu", str(8), "-E",str(1), "-o", val_hmmscan_output, hmm_db, val_db_fasta],stdout=subprocess.DEVNULL, check=True)

def filter_overlapping_domains(hmmscan_dict: dict, min_match_state: float, min_overlap: float) -> dict:
    def calculate_match_percentage(match_string: str) -> float:
        return (match_string.count('M') / len(match_string)) * 100

    def calculate_overlap(start1: int, end1: int, start2: int, end2: int) -> float:
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        if overlap_start < overlap_end:
            overlap_length = overlap_end - overlap_start + 1
            total_length = min(end1 - start1 + 1, end2 - start2 + 1)
            return (overlap_length / total_length) * 100
        return 0.0

    filtered_dict = {}

    for seq, data in hmmscan_dict.items():
        hit_domains = data['hit_domains']
        filtered_domains = []

        for i in range(len(hit_domains)):
            start1, end1, score1, family1, match_string1 = hit_domains[i]
            match_percentage1 = calculate_match_percentage(match_string1)
            if match_percentage1 < min_match_state:
                continue

            keep_domain = True
            for j in range(len(hit_domains)):
                if i == j:
                    continue
                start2, end2, score2, family2, match_string2 = hit_domains[j]
                match_percentage2 = calculate_match_percentage(match_string2)
                if match_percentage2 < min_match_state:
                    continue

                overlap_percentage = calculate_overlap(start1, end1, start2, end2)
                if overlap_percentage >= min_overlap:
                    if score1 < score2:
                        keep_domain = False
                        break

            if keep_domain:
                filtered_domains.append(hit_domains[i])

        filtered_dict[seq] = {'length': data['length'], 'hit_domains': filtered_domains}

    return filtered_dict

def filter_overlapping_domains_sorted(hmmscan_dict: dict, min_match_state: float, min_overlap: float) -> dict:
    def calculate_match_percentage(match_string: str) -> float:
        return (match_string.count('M') / len(match_string)) * 100

    def calculate_overlap(start1: int, end1: int, start2: int, end2: int) -> float:
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        if overlap_start < overlap_end:
            overlap_length = overlap_end - overlap_start + 1
            total_length = min(end1 - start1 + 1, end2 - start2 + 1)
            return (overlap_length / total_length) * 100
        return 0.0

    filtered_dict = {}

    for seq, data in hmmscan_dict.items():
        hit_domains = sorted(data['hit_domains'], key=lambda x: x[2], reverse=True)  # Sort by score in decreasing order
        filtered_domains = []
        removed_indices = set()

        for i in range(len(hit_domains)):
            if i in removed_indices:
                continue  # Skip domains already identified for removal

            start1, end1, score1, family1, match_string1 = hit_domains[i]
            match_percentage1 = calculate_match_percentage(match_string1)
            if match_percentage1 < min_match_state:
                continue

            for j in range(i + 1, len(hit_domains)):  # Only compare with domains not yet processed
                if j in removed_indices:
                    continue
                start2, end2, score2, family2, match_string2 = hit_domains[j]
                match_percentage2 = calculate_match_percentage(match_string2)
                if match_percentage2 < min_match_state:
                    continue

                overlap_percentage = calculate_overlap(start1, end1, start2, end2)
                if overlap_percentage >= min_overlap:
                    removed_indices.add(j)  # Mark the lower scoring domain for removal

            filtered_domains.append(hit_domains[i])  # Add the current domain as it's not overlapping with a higher score

        # Sort filtered_domains by start position (the first element of the tuple) in increasing order
        filtered_domains_sorted = sorted(filtered_domains, key=lambda x: x[0])
        filtered_dict[seq] = {'length': data['length'], 'hit_domains': filtered_domains_sorted}

    return filtered_dict

def filter_val_best_hits():
    save_path = '../data/full_data_curation'
    
    # Load val_hmmscan_output.txt
    hmmscan_dict = hu.parse_hmmscan_results(f"{save_path}/val_hmmscan_output.txt", score_threshold=0)

    # Apply filtering for overlapping domains to each sequence in hmmscan_dict
    min_match_state = 80.0
    min_overlap = 15.0
    filtered_dict = filter_overlapping_domains_sorted(hmmscan_dict, min_match_state, min_overlap)

    # Load final_val_seqs.pkl, a dict where the keys are the sequence names
    with open(f"{save_path}/final_val_seqs.pkl", "rb") as f:
        val_seqs = pickle.load(f)
    
    # Load the mapping from name to accession
    with open(f"{save_path}/name_to_accession.pkl", "rb") as f:
        name_to_accession = pickle.load(f)

    # Initialize the filtered version of val_seqs
    filtered_val_seqs = {}
    # Iterate over each sequence and its associated domains in val_seqs
    for seq, domains in val_seqs.items():
        # Skip sequences that are not in filtered_dict
        if seq not in filtered_dict:
            continue
        
        filtered_domains = []
        # Iterate over each domain in the current sequence
        for domain in domains:
            seqname_start_stop, score, family_name = domain
            # Extract start and stop positions from the domain string
            start, stop = map(int, seqname_start_stop.split('/')[1].split('-'))
            # Get the accession number for the family name
            accession = name_to_accession.get(family_name)

            # Skip if accession is not found
            if not accession:
                continue

            # Iterate over hit domains in the filtered dictionary for the current sequence
            for hit_domain in filtered_dict[seq]['hit_domains']:
                hit_start, hit_stop, hit_score, hit_family, hit_match_string = hit_domain

                # Check if the hit domain family matches the accession
                if hit_family == accession:
                    # Calculate the overlap length between the current domain and the hit domain
                    overlap_length = min(stop, hit_stop) - max(start, hit_start) + 1
                    # Calculate the length of the current domain
                    domain_length = stop - start + 1
                    # Calculate the overlap percentage
                    overlap_percentage = (overlap_length / domain_length) * 100

                    # If the overlap percentage is 80% or more, add the domain to filtered_domains
                    if overlap_percentage >= 80:
                        filtered_domains.append(domain)
                        break

        # If there are any filtered domains, add them to filtered_val_seqs
        if filtered_domains:
            filtered_val_seqs[seq] = filtered_domains

    # Save the filtered val_seqs
    with open(f"{save_path}/hmmscan_filtered_val_seqs.pkl", "wb") as f:
        pickle.dump(filtered_val_seqs, f)

    # Save filtered hmmscan_dict subset only to the keys in filtered_val_seqs
    filtered_hmmscan_dict = {seq: filtered_dict[seq] for seq in filtered_val_seqs.keys()}
    with open(f"{save_path}/filtered_val_hmmscan.pkl", "wb") as f:
        pickle.dump(filtered_hmmscan_dict, f)

    return filtered_val_seqs

def train_val_split_fasta():
    # load the seq_db and hmmscan_filtered_val_seqs dictionary
    # Any sequence not in vals dict (as a key) should be added to the train fasta file

    save_path = '../data/full_data_curation'
    seq_db = '../data/train_test_splitting/Pfam_A.seed.full_seqs.fasta'
    val_seqs = f"{save_path}/hmmscan_filtered_val_seqs.pkl"
    output_dir = '../data/train_fasta_PSALM_1b'

    with open(val_seqs, "rb") as f:
        val_seqs_dict = pickle.load(f)

    full_db = list(SeqIO.parse(seq_db, "fasta"))
    train_db = [seq for seq in full_db if seq.id not in val_seqs_dict.keys()]
    train_db_fasta = f"{save_path}/train_db.fasta"
    with open(train_db_fasta, "w") as f:
        SeqIO.write(train_db, f, "fasta")

    # Shuffle sequences
    shuffle(train_db)

    # Determine shard size
    total_sequences = len(train_db)
    shard_size = total_sequences // 100

    os.makedirs(output_dir, exist_ok=True)

    # Write to shards
    for shard_index in range(100):
        start_index = shard_index * shard_size
        # For the last shard, include all remaining sequences
        end_index = (shard_index + 1) * shard_size if shard_index < 99 else total_sequences
        shard_sequences = train_db[start_index:end_index]
        
        # Construct shard filename
        shard_filename = os.path.join(output_dir, f'shard_{shard_index+1}_tr.fasta')
        
        # Write shard sequences to file
        with open(shard_filename, 'w') as output_file:
            SeqIO.write(shard_sequences, output_file, 'fasta')

def calculate_coverage(hmmscan):
    coverage_dict = {}
    for seq_name, details in hmmscan.items():
        length = details['length']
        hit_domains = details['hit_domains']
        
        # Sort hit_domains by start position
        hit_domains_sorted = sorted(hit_domains, key=lambda x: x[0])
        
        # Merge overlapping intervals and calculate total covered length
        total_covered_length = 0
        current_start, current_stop = hit_domains_sorted[0][:2]
        for start, stop, *_ in hit_domains_sorted[1:]:
            if start <= current_stop:
                # Overlapping, extend the current interval
                current_stop = max(current_stop, stop)
            else:
                # Non-overlapping, add the interval length to total and start a new interval
                total_covered_length += current_stop - current_start
                current_start, current_stop = start, stop
        # Add the last interval
        total_covered_length += current_stop - current_start
        
        # Calculate coverage percentage
        coverage_percentage = (total_covered_length / length) * 100
        coverage_dict[seq_name] = coverage_percentage
    
    return coverage_dict

def final_val_formatting():
    save_path = '../data/full_data_curation'
    seq_db = '../data/train_test_splitting/Pfam_A.seed.full_seqs.fasta'
    val_seqs = f"{save_path}/filtered_val_hmmscan.pkl"
    # Load the mapping from name to accession
    # with open(f"{save_path}/name_to_accession.pkl", "rb") as f:
    #     name_to_accession = pickle.load(f)
    
    # Load the val_seqs dictionary
    with open(val_seqs, "rb") as f:
        val_seqs_dict = pickle.load(f)

    # Load the sequence database
    full_db = list(SeqIO.parse(seq_db, "fasta"))

    # Create a dictionary for quick lookup of SeqRecords by ID
    seq_db_dict = {record.id: record for record in full_db}

    # Get coverage_dict
    coverage_dict = calculate_coverage(val_seqs_dict)

    # subset the hmmscan_dict to only include sequences with coverage >= 80%
    coverage_thresh = 80
    high_coverage_dict = {seq: val_seqs_dict[seq] for seq in coverage_dict if coverage_dict[seq] >= coverage_thresh}
    test_dict = {seq: val_seqs_dict[seq] for seq in coverage_dict if coverage_dict[seq] < coverage_thresh}

    # write the seqs that are keys in high_coverage_dict to a fasta file
    high_coverage_seqs = [seq_db_dict[seq] for seq in high_coverage_dict.keys()]
    high_coverage_fasta = f"{save_path}/PSALM_1b_Validation_FINAL.fasta"

    with open(high_coverage_fasta, "w") as f:
        SeqIO.write(high_coverage_seqs, f, "fasta")
    
    # write the seqs that are keys in test_dict to a fasta file
    test_seqs = [seq_db_dict[seq] for seq in test_dict.keys()]
    test_fasta = f"{save_path}/PSALM_1b_Test_FINAL.fasta"

    with open(test_fasta, "w") as f:
        SeqIO.write(test_seqs, f, "fasta")
    
    # save the high_coverage_dict and test_dict
    with open(f"{save_path}/PSALM_1b_Validation_Scan_FINAL.pkl", "wb") as f:
        pickle.dump(high_coverage_dict, f)
    
    with open(f"{save_path}/PSALM_1b_Test_Scan_FINAL.pkl", "wb") as f:
        pickle.dump(test_dict, f)
    
    # # Randomly select 20% of sequences
    # num_to_select = int(0.2 * len(modified_sequences))
    # selected_sequences = random.sample(modified_sequences, num_to_select)

    # # Add shuffled sequences back to modified_sequences with new IDs
    # for i, seq in enumerate(selected_sequences):
    #     new_id = f"shuffled_{seq.id}"
        
    #     # Shuffle the sequence itself
    #     seq_list = list(seq.seq)
    #     random.shuffle(seq_list)
    #     shuffled_seq = Seq(''.join(seq_list))
        
    #     new_record = SeqIO.SeqRecord(shuffled_seq, id=new_id, description="")
    #     modified_sequences.append(new_record)

    #     # Add corresponding entry in custom_hmmscan_dict with family_id "IDR"
    #     custom_hmmscan_dict[new_id] = {
    #         'length': len(shuffled_seq),
    #         'hit_domains': [(0, len(shuffled_seq), 1000, "IDR", 'M' * len(shuffled_seq))]
    #     }

    # # Save 
    # new_fasta = f'{save_path}/PSALM_1b_Validation_FINAL.fasta'
    # with open(new_fasta, "w") as f:
    #     SeqIO.write(modified_sequences, f, "fasta")

    # with open(f"{save_path}/PSALM_1b_Validation_Scan_FINAL.pkl", "wb") as f:
    #     pickle.dump(custom_hmmscan_dict, f) 

def is_significant_overlap(new_start, new_stop, hit_domains, overlap_factor=0.5):
    """
    Check if there's a significant overlap between the new domain and existing domains.
    overlap_factor is the fraction of the new domain's length that must be overlapped
    for the overlap to be considered significant.
    """
    new_length = new_stop - new_start + 1
    min_overlap_length = new_length * overlap_factor

    for start, stop, _, _, _ in hit_domains:
        # Calculate overlap
        overlap = min(new_stop, stop) - max(new_start, start) + 1
        if overlap >= min_overlap_length:
            return True  # Significant overlap found
    return False  # No significant overlap

def fix_nohits():
    random.seed(100)
    # Load train fasta file
    train_file = '../data/full_data_curation/train_db.fasta'
    save_path = '../data/full_data_curation'
    hmmstats = "/n/eddy_lab/data/pfam-35.0/Pfam-A.hmm.stats"
    afetch_command = "/n/eddy_lab/software/bin/esl-afetch"
    msa_db = '/n/eddy_lab/data/pfam-35.0/Pfam-A.seed'

    train_seqs = list(SeqIO.parse(train_file, 'fasta'))
    # Create a dictionary mapping sequence IDs to their lengths
    seq_lengths = {seq.id: len(str(seq.seq).replace('-', '').replace('.', '')) for seq in train_seqs}

    hmmdict = {}
    for i in range(1,100):
        dict = hu.parse_hmmscan_results(f'../data/train_scan_PSALM_1b/shard_{i}_tr.txt',score_threshold=0)
        hmmdict = hmmdict | dict
    #save hmmdict
    # with open(f'{save_path}/allscans3.pkl', 'wb') as f:
    #     pickle.dump(hmmdict, f)
    with open(f'{save_path}/allscans3.pkl', 'rb') as f:
        hmmdict = pickle.load(f)
    
    # Find the list of sequences that are not in the hmmdict
    nohits = list(set(seq_lengths.keys()).difference(hmmdict.keys()))

    # Step 1: Load families
    families = load_hmmstats(hmmstats)['accession'].tolist()

    # Step 2: Fetch MSA for each family and process
    for family in tqdm(families):
        msa_file = f"{save_path}/tmp.sto"  # Temporary file for MSA
        # Fetch MSA using esl-afetch
        subprocess.run([afetch_command, "-o", msa_file, msa_db, family], stdout=subprocess.DEVNULL, check=True)
        
        # Parse the MSA file
        try:
            with open(msa_file, 'r', encoding='latin-1') as handle:
                alignment = AlignIO.read(handle, "stockholm")
        except:
            print(f"Error reading {family}")
            continue
        # pdb.set_trace()
        # Step 3: Check each sequence in the alignment against hmmdict
        for record in alignment:
            _, start_stop = str(record.id).split('/')[0], str(record.id).split('/')[1]
            start, stop = map(int, start_stop.split('-'))
            start -=1
            # stop -=1
            record_str = str(record)
            # Extracting the accession code
            accession_start = record_str.find("/accession=") + len("/accession=")
            accession_end = record_str.find("\n", accession_start)
            accession_code = record_str[accession_start:accession_end]
            # pdb.set_trace()
            # add to hmmdict if seq in train but not in hmmdict
            if accession_code in nohits:      
                seq_length = seq_lengths[accession_code]
                print(f"Adding no-hit sequence: {accession_code} from family {family} with length {seq_length}")  # Debug print statement
                # Create a new entry for hmmdict
                hmmdict[accession_code] = {
                    'length': seq_length,
                    'hit_domains': []
                }
                # Assuming domain information can be extracted or is known; here we use placeholders
                
                hmmdict[accession_code]['hit_domains'].append((start-1, stop, 1000.0, family, 'M' * (stop - start + 1))
                )

                # update nohits
                nohits.remove(accession_code)
                # pdb.set_trace()
            
            # even if seq in train, check that the corresponding family is present in a tuple in the hit_domains list of the seq
            elif accession_code in seq_lengths:
                if family not in [x[3] for x in hmmdict[accession_code]['hit_domains']]:
                    # if not is_significant_overlap(start, stop, hmmdict[accession_code]['hit_domains']):
                        print(f"Adding family {family} to sequence {accession_code}")
                        # pdb.set_trace()
                        hmmdict[accession_code]['hit_domains'].append((start-1, stop, 1000.0, family, 'M' * (stop - start + 1))
                )

    # Sample and get negatives and add back into fasta and hmmdict
    total_sequences = len(train_seqs)
    sample_size = int(np.ceil(0.01*total_sequences))
    sampled_sequences = random.sample(train_seqs, sample_size)

    shuffled_sequences = []
    for seq in sampled_sequences:
        shuffled_seq = ''.join(random.sample(str(seq.seq), len(seq.seq)))
        new_id = f"shuffled_{seq.id}"
        shuffled_sequences.append(SeqIO.SeqRecord(Seq(shuffled_seq), id=new_id, description=""))
    
    # Append shuffled sequences to the fasta file
    new_fasta = f'{save_path}/PSALM_1b_Train_FINAL.fasta'
    with open(new_fasta, "w") as new_file:
        SeqIO.write(train_seqs, new_file, "fasta")
    with open(new_fasta, "a") as f:
        SeqIO.write(shuffled_sequences, f, "fasta")
    
    # Add shuffled sequences to hmmdict
    for seq in shuffled_sequences:
        accession_code = seq.id.split('/')[0]
        seq_length = len(seq.seq)
        hmmdict[accession_code] = {
            'length': seq_length,
            'hit_domains': [(0, seq_length, 100.0, "IDR", 'M' * seq_length)]
        }

    # Apply filtering for overlapping domains to each sequence in hmmscan_dict
    min_match_state = 80.0
    min_overlap = 15.0
    filtered_dict = filter_overlapping_domains_sorted(hmmdict, min_match_state, min_overlap)

    #save hmmdict
    with open(f'{save_path}/PSALM_1b_Train_Scan_FINAL.pkl', 'wb') as f:
        pickle.dump(filtered_dict, f)
    
if __name__ == "__main__":
    # get_val_seqs()
    # check_seeds_for_val_seqs()
    # get_val_best_hits()
    filter_val_best_hits()
    train_val_split_fasta()
    final_val_formatting()
    fix_nohits()