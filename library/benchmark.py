import subprocess
from Bio import SeqIO
import pandas as pd
import pickle
import csv
import os
import glob
from tqdm import tqdm

hmmfetch_command = "/n/eddy_lab/software/bin/hmmfetch"
hmmsearch_command = "/n/eddy_lab/software/bin/hmmsearch"
reformat_command = "/n/eddy_lab/software/bin/esl-reformat"
align_command = "/n/eddy_lab/software/bin/hmmalign"
alipid_command = "/n/eddy_lab/software/bin/esl-alipid"
incE = str(0.001) #0.001

hmm_db = "/n/eddy_lab/data/pfam-35.0/Pfam-A.hmm"
train_db = "/n/eddy_lab/Lab/protein_annotation_dl/PSALM/data/train_fasta/train_ids_full.fasta"
test_db = "/n/eddy_lab/Lab/protein_annotation_dl/PSALM/data/test_fasta/test_ids_full.fasta"

save_path = "/n/eddy_lab/Lab/protein_annotation_dl/PSALM/data/benchmarking"
output_hmm = f"{save_path}/tmp.hmm"
output_tr_sto = f"{save_path}/tmp_tr_msa.sto"
output_te_sto = f"{save_path}/tmp_te_msa.sto"
output_tr_fasta = f"{save_path}/tmp_tr.fasta"
output_te_fasta = f"{save_path}/tmp_te.fasta"
tr_MMSA = f"{save_path}/train.seed.sto"
tmp_align = f"{save_path}/tmp_tr_te.sto"
tmp_alipid = f"{save_path}/tmp_alipid.txt"

# Open log file
with open(f"{save_path}/log.csv", "w") as log_file:
    writer = csv.writer(log_file)

    # Get list of family IDs
    with open(f"{save_path}"+"/../maps.pkl", "rb") as f:
        maps = pickle.load(f)
    family_ids = list(maps['fam_idx'].keys())

    # Iterate through each family
    for family_id in tqdm(family_ids):
        # family_id = "PF10417.12"

        # hmmfetch profile for selected family
        subprocess.run([hmmfetch_command, "-o", output_hmm, hmm_db, family_id], stdout=subprocess.DEVNULL)

        # hmmsearch profile against test database and return MSA of hits
        subprocess.run([hmmsearch_command, "-A", output_te_sto, "--incE", incE, output_hmm, test_db], stdout=subprocess.DEVNULL)

        # hmmsearch profile against training database and return MSA of hits
        subprocess.run([hmmsearch_command, "-A", output_tr_sto, "--incE", incE, output_hmm, train_db], stdout=subprocess.DEVNULL)

        # reformat train and test stocholm files to fasta format
        subprocess.run([reformat_command, "--informat", "stockholm", "-o", output_tr_fasta, "fasta", output_tr_sto])
        subprocess.run([reformat_command, "--informat", "stockholm", "-o", output_te_fasta, "fasta", output_te_sto])

        # Take each sequence from test and gets its max pid with train
        tr_sequences = list(SeqIO.parse(output_tr_fasta, "fasta"))
        te_sequences = list(SeqIO.parse(output_te_fasta, "fasta"))
        # for seq in te_sequences:
        #     # Write all of train and a single test sequence to a file
        filename = f"{save_path}/tmp_{seq.id.replace('/', '_')}.fasta"
        with open(filename, "w+") as f:
            SeqIO.write(tr_sequences, f, "fasta") 
            SeqIO.write(te_sequences, f, "fasta")
            
        # hmmalign the file to the corresponding profile hmm
        subprocess.run([align_command, "-o", tmp_align, output_hmm, filename])

        # get alipid for the alignment
        with open(tmp_alipid, "w") as f:
            subprocess.run([alipid_command, tmp_align], stdout=f)

        # Retrieve the max pid from the alipid file
        df = pd.read_csv(tmp_alipid, sep="\s+", usecols=[0, 1, 2], names = ["seq1","seq2","pid"],header=0)

        # Subset where one of the sequences is from train and the other from test
        subset_df1 = df[(df['seq1'].isin(tr_sequences)) & (df['seq2'].isin(te_sequences))]
        subset_df2 = df[(df['seq1'].isin(te_sequences)) & (df['seq2'].isin(tr_sequences))]

        subset_df1 = subset_df1[['seq2', 'pid']]
        subset_df2 = subset_df2[['seq1', 'pid']]
        concat_df = pd.concat([subset_df1, subset_df2], ignore_index=True)
        '''
        FIX THIS PART AND CHECK COLUMN NAMES FOR CONCAT_DF
        '''
        # Get the max pid for each sequence
        max_pid = concat_df.groupby('seq2')['pid'].max().max()
        max_pid = subset_df1['pid'].max()

        # Log <seq_id, domain, max_pid>
        writer.writerow([seq.id, family_id, max_pid])

        # Remove tmp files from save_path
        tmp_files = glob.glob(f"{save_path}/tmp*")
        for file in tmp_files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting file {file}: {e}")