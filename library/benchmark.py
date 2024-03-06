import subprocess
from Bio import SeqIO
import pandas as pd
import csv

hmmfetch_command = "/n/eddy_lab/software/bin/hmmfetch"
hmmsearch_command = "/n/eddy_lab/software/bin/hmmsearch"
reformat_command = "/n/eddy_lab/software/bin/esl-reformat"
align_command = "/n/eddy_lab/software/bin/hmmalign"
alipid_command = "/n/eddy_lab/software/bin/esl-alipid"
incE = str(1)

hmm_db = "/n/eddy_lab/data/pfam/pfam-35.0/Pfam-A.hmm"
train_db = "/n/eddy_lab/Lab/protein_annotation_dl/data/train_fasta/train_ids_full.fasta"
test_db = "/n/eddy_lab/Lab/protein_annotation_dl/data/test_fasta/test_ids_full.fasta"

save_path = "/n/eddy_lab/Lab/protein_annotation_dl/PSALM/data/benchmarking"
output_hmm = f"{save_path}/temp.hmm"
output_tr_sto = f"{save_path}/temp_tr_msa.sto"
output_te_sto = f"{save_path}/temp_te_msa.sto"
output_tr_fasta = f"{save_path}/temp_tr.fasta"
output_te_fasta = f"{save_path}/temp_te.fasta"
tr_MMSA = f"{save_path}/train.seed.sto"
temp_align = f"{save_path}/temp_tr_te.sto"
temp_alipid = f"{save_path}/temp_alipid.txt"

# Open log file
with open(f"{save_path}/log.csv", "w") as log_file:
    writer = csv.writer(log_file)

# For a given family ID
family_id = "PF10417.12"

# hmmfetch profile for selected family
subprocess.run([hmmfetch_command, "-o", output_hmm, hmm_db, family_id], stdout=subprocess.DEVNULL)

# hmmsearch profile against test database and return MSA of hits
subprocess.run([hmmsearch_command, "-A", output_te_sto, "--incE", incE, output_hmm, test_db], stdout=subprocess.DEVNULL)
'''
****IF NO HITS ARE RETURNED, SKIP THE REST OF THE CODE****
'''

# hmmsearch profile against training database and return MSA of hits
subprocess.run([hmmsearch_command, "-A", output_tr_sto, "--incE", incE, output_hmm, train_db], stdout=subprocess.DEVNULL)

# reformat train and test stocholm files to fasta format
subprocess.run([reformat_command, "--informat", "stockholm", "-o", output_tr_fasta, "fasta", output_tr_sto])
subprocess.run([reformat_command, "--informat", "stockholm", "-o", output_te_fasta, "fasta", output_te_sto])

# Take each sequence from test and gets its max pid with train
tr_sequences = list(SeqIO.parse(output_tr_fasta, "fasta"))
te_sequences = SeqIO.parse(output_te_fasta, "fasta")
for seq in te_sequences:
    # Write all of train and a single test sequence to a file
    filename = f"{save_path}/temp_{seq.id.replace('/', '_')}.fasta"
    with open(filename, "w+") as f:
        SeqIO.write(tr_sequences, f, "fasta") 
        SeqIO.write(seq, f, "fasta")
    
    # hmmalign the file to the corresponding profile hmm
    subprocess.run([align_command, "-o", temp_align, output_hmm, filename])

    # get alipid for the alignment
    with open(temp_alipid, "w") as f:
        subprocess.run([alipid_command, temp_align], stdout=f)

    # Retrieve the max pid from the alipid file
    df = pd.read_csv(temp_alipid, sep="\s+", usecols=[0, 1, 2], names = ["seq1","seq2","pid"],header=0)
    subset_df = df[(df['seq1'] == seq.id) | (df['seq2'] == seq.id)]
    max_pid = subset_df['pid'].max()

    # Log <seq_id, domain, family_pid>
    writer.writerow([seq.id, family_id, max_pid])