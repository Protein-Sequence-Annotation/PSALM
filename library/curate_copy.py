import subprocess
from pathlib import Path
from Bio import SeqIO
import numpy as np
from tqdm import tqdm

# # Define the shell command
# command = f'awk -v query="{query_name}" \'$1 ~ query || $2 ~ query {{print $3}}\' alipid.txt | sort -nr | head -n 1'

# # Run the shell command
# process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
# output, error = process.communicate()

# # Convert the output to a float
# max_id = float(output.decode('utf-8').strip())

# # Print the maximum %id
# print(max_id)

# alipid
# /n/eddy_lab/software/bin/esl-alipid data/train_test_splitting/align.sto > data/train_test_splitting/alipid.txt

# phmmer
# /n/eddy_lab/software/bin/phmmer --incE 0.001 -A data/train_test_splitting/align.sto data/train_test_splitting/query.fasta data/train_test_splitting/combo.fasta 

'''
1. Select query
2. Insert query into train
3. phmmer query against train
4. If query maxpid < 0.25, then insert query into test
5. remove query from train
6. Else repeat 2-5 with test

Details for step 4.
phmmer query with incE of 0.001 against train/test (should include query) and output to alignment
esl-alipid the resulting alignment
awk to get the max %id
'''

full_seqs_path = Path('../data/train_test_splitting/Pfam_A.seed.full_seqs.fasta')
save_path = full_seqs_path.parent

phmmer_command = '/n/eddy_lab/software/bin/phmmer'
alipid_command = '/n/eddy_lab/software/bin/esl-alipid'

train_path = Path('../data/train_test_splitting_tmp/full_train.fasta')
test_path = Path('../data/train_test_splitting_tmp/full_test.fasta')
phmmer_out_path = Path('../data/train_test_splitting_tmp/phmmer_out.sto')
alipid_out_path = Path('../data/train_test_splitting_tmp/alipid_out.txt')
query_path = Path('../data/train_test_splitting_tmp/query.fasta')

incE = str(0.001)
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
    handle.write(record) # Add the query sequence to the file

    with open(query_path, 'w') as query: # Tmp file for query
        SeqIO.write(record, query)

    query_id = record.id # Get name of the query

    # Run pHMMER to get best hits to either train/test group
    subprocess.run([phmmer_command, "--incE", incE, "-A", phmmer_out_path, query_path, fpath], stdout=subprocess.DEVNULL, check=True)
    
    handle.seek(current_pos) # Go to file pointer location before query was added
    handle.truncate() # Truncate file to this position

    subprocess.run([alipid_command, phmmer_out_path], stdout=open(alipid_out_path, 'w'), check=True) # Run alipid

    # Check any column with query_id, sort by pid and return the max
    pid_extract_command = f'awk -v query="{query_id}" \'$1 ~ query || $2 ~ query {{print $3}}\' {alipid_out_path} | sort -nr | head -n 1'

    # Extract max pid from the alipid output file
    pid_output = subprocess.Popen(pid_extract_command, stdout=subprocess.PIPE, shell=True)
    pid_result, _ = pid_output.communicate()

    # Convert the output to a float
    max_pid = float(pid_result.decode('utf-8').strip())

    # Print the maximum %id
    return max_pid < threshold

def split_seqs(threshold, halt=-1):

    """
    Split all sequences from Pfam seed into train and test

    Args:
        threshold (float): Cutoff for pid - ensure that train and test are at most threshold % apart
        halt (int): Break after halt number of sequences, useful for debugging
    """

    with open(full_seqs_path, 'r') as seqs_db:

        num_records = SeqIO.count(seqs_db, "fasta") # Number of sequences
        train_ctr = 0 # Track train sequences
        test_ctr = 0 # Track test seqeuences

        group_idx = np.random.choice([0,1], size=num_records, p=[0.75, 0.25]) # 0 - train, 1 - test

        train_file = open(train_path, 'a') # Use append mode because 'w' truncates to 0 length before writing
        test_file = open(test_path, 'a')

        for idx, record in tqdm(enumerate(SeqIO.parse(seqs_db, "fasta"), total=num_records, desc='Sequences Parsed')):

            if group_idx[idx] == 0: # Inclusion in train
                if (test_ctr != 0 and checkAssignment(record, test_file, threshold)) or test_ctr == 0:
                    SeqIO.write(record, train_file) # Write sequence
                    train_ctr += 1

            else: # Inclusion in test
                if (train_ctr != 0 and checkAssignment(record, train_file, threshold)) or train_ctr == 0:
                    SeqIO.write(record, test_file) # Write sequence
                    test_ctr += 1

            if idx == halt: # For testing puposes, if we want to stop after a certain number sequences
                break
        
        train_file.close()
        test_file.close()

    print('Sequence splitting completed')
    print(f'Train: {train_ctr} Test: {test_ctr}')

    return

if __name__ == '__main__':

    split_seqs(25, 20)