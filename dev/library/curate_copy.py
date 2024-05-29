import subprocess
from pathlib import Path
from Bio import SeqIO
import numpy as np
from tqdm import tqdm
from os import stat


# awk -F ' ' '{for (i=1; i<=2; i++) {if ($i ~ /\//) {$i=substr($i, 1, index($i, "/")-1)}}} 1' alipid_out.txt

full_seqs_path = Path('../data/train_test_splitting/Pfam_A.seed.full_seqs.fasta')
save_path = Path('../data/train_test_splitting') # full_seqs_path.parent # for now - until permission is changed on original

phmmer_command = '/n/eddy_lab/software/bin/phmmer'
alipid_command = '/n/eddy_lab/software/bin/esl-alipid'

train_path = Path('../data/train_test_splitting/full_train.fasta')
test_path = Path('../data/train_test_splitting/full_test.fasta')
phmmer_out_path = Path('../data/train_test_splitting/phmmer_out.sto')
alipid_out_path = Path('../data/train_test_splitting/alipid_out.txt')
query_path = Path('../data/train_test_splitting/query.fasta')

incE = str(0.01)
np.random.seed(42) # That's the only number I know

def checkAssignment(record, handle, fpath, threshold):

    """
    Check whether given sequence has pid less than threshold with the group being compared against

    Args:
        record (Bio.SeqIO): A query record being tested for feasibility
        handle (File): File handle to the group of sequences query is being tested against
        fpath (Path): file path to the group being tested against

    Returns:
        bool: True or False indicating whether the pid criterion is satisfied
    """

    current_pos = handle.tell() # Store current position of file pointer
    SeqIO.write(record, handle, 'fasta') # Add the query sequence to the file

    with open(query_path, 'w') as query: # Tmp file for query
        SeqIO.write(record, query, 'fasta')

    query_id = record.id # Get name of the query

    # Run pHMMER to get best hits to either train/test group
    subprocess.run([phmmer_command, "--incE", incE, "-A", phmmer_out_path, query_path, fpath], stdout=subprocess.DEVNULL, check=True)
    
    handle.seek(current_pos) # Go to file pointer location before query was added
    handle.truncate() # Truncate file to this position

    # Check if phmmer output file is empty i.e. no hits satisfying incE found, then return True
    file_size = stat(phmmer_out_path).st_size
    
    if file_size == 0:
        return True

    subprocess.run([alipid_command, '--amino', phmmer_out_path], stdout=open(alipid_out_path, 'w'), check=True) # Run alipid

    # Check for exactly one column with query_id, sort by pid and return the max
    pid_extract_command = f'awk -v q="{query_id}" \'($1 ~ q || $2 ~ q) && !($1 ~ q && $2 ~ q) {{print $3}}\' {alipid_out_path} | sort -nr | head -n 1'

    # Extract max pid across all comparisons involving query from the alipid output file
    pid_output = subprocess.Popen(pid_extract_command, stdout=subprocess.PIPE, shell=True)
    pid_result, _ = pid_output.communicate()
    max_pid = pid_result.decode('utf-8').strip()

    # Check if alipid output file is empty i.e. best hit was to itself, so not present in output

    if not max_pid:
        return True

    return float(max_pid) < threshold

def split_seqs(threshold, halt=-1):

    """
    Split all sequences from Pfam seed into train and test

    Args:
        threshold (float): Cutoff for pid - ensure that train and test are at most threshold % apart
        halt (int): Break after halt number of sequences, useful for debugging
    """

    with open(full_seqs_path, 'r') as seqs_db:

        num_records =  1104223 if halt == -1 else halt # Number of sequences hard coded based on precomputed value
        train_ctr = 1 # Track train sequences
        test_ctr = 1 # Track test seqeuences

        group_idx = np.random.choice([0,1], size=num_records, p=[0.75, 0.25]) # 0 - train, 1 - test
        train_file = open(train_path, 'a') # Use append mode because 'w' truncates to 0 length before writing
        test_file = open(test_path, 'a')

        for idx, record in tqdm(enumerate(SeqIO.parse(seqs_db, "fasta")), total=num_records, desc='Sequences Parsed'):

            if idx < 909775:
                continue

            elif group_idx[idx] == 0: # Inclusion in train if there are no seqs in test OR assignment is possible
                if (test_ctr != 0 and checkAssignment(record, test_file, test_path, threshold)) or test_ctr == 0:
                    SeqIO.write(record, train_file, 'fasta') # Write sequence
                    train_ctr += 1
                # else:
                #     print('Failed assignment to train')

            else: # Inclusion in test if there are no seqs in train OR assignment is possible
                if (train_ctr != 0 and checkAssignment(record, train_file, train_path, threshold)) or train_ctr == 0:
                    SeqIO.write(record, test_file, 'fasta') # Write sequence
                    test_ctr += 1
                # else:
                #     print('Failed assignment to test')

            if idx == halt-1: # For testing puposes, if we want to stop after a certain number sequences
                break
        
        train_file.close()
        test_file.close()

    print('Sequence splitting completed')
    print(f'Train: {train_ctr} Test: {test_ctr}')

    return

if __name__ == '__main__':

    split_seqs(25, -1)