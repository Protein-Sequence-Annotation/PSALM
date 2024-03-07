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
        try:
            subprocess.run([hmmfetch_command, "-o", output_hmm, hmm_db, family_id], stdout=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f"Error fetching {family_id} hmm from database")
        
        # hmmsearch profile against test database and return MSA of hits      
        try:
            subprocess.run([hmmsearch_command, "-A", output_te_sto, "--incE", incE, output_hmm, test_db], stdout=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f"Error searching {family_id} hmm against test database")

        # hmmsearch profile against training database and return MSA of hits
        try:
            subprocess.run([hmmsearch_command, "-A", output_tr_sto, "--incE", incE, output_hmm, train_db], stdout=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f"Error searching {family_id} hmm against train database")
        
        # reformat train and test stocholm files to fasta format
        try:
            subprocess.run([reformat_command, "--informat", "stockholm", "-o", output_tr_fasta, "fasta", output_tr_sto], check=True)
        except:
            print(f"Error reformatting {family_id} train stockholm to fasta")
        try:
            subprocess.run([reformat_command, "--informat", "stockholm", "-o", output_te_fasta, "fasta", output_te_sto], check=True)
        except:
            print(f"Error reformatting {family_id} test stockholm to fasta")

        # Take each sequence from test and gets its max pid with train
        tr_sequences = list(SeqIO.parse(output_tr_fasta, "fasta"))
        te_sequences = list(SeqIO.parse(output_te_fasta, "fasta"))
        # for seq in te_sequences:
        #     # Write all of train and a single test sequence to a file
        filename = f"{save_path}/tmp_{family_id}.fasta"
        with open(filename, "w+") as f:
            SeqIO.write(tr_sequences, f, "fasta") 
            SeqIO.write(te_sequences, f, "fasta")
            
        # hmmalign the file to the corresponding profile hmm
        try:
            subprocess.run([align_command, "-o", tmp_align, output_hmm, filename], check=True)
        except:
            print(f"Error aligning {family_id} sequences")

        # get alipid for the alignment
        with open(tmp_alipid, "w") as f:
            try:
                subprocess.run([alipid_command, tmp_align], stdout=f, check=True)
            except:
                print(f"Error getting alipid for {family_id} alignment")

        # Retrieve the max pid from the alipid file
        df = pd.read_csv(tmp_alipid, sep="\s+", usecols=[0, 1, 2], names = ["seq1","seq2","pid"],header=0)

        # Subset where one of the sequences is from train and the other from test
        tr_sequences = [seq.id for seq in tr_sequences]
        te_sequences = [seq.id for seq in te_sequences]
        subset_df1 = df[(df['seq1'].isin(tr_sequences)) & (df['seq2'].isin(te_sequences))]
        subset_df2 = df[(df['seq1'].isin(te_sequences)) & (df['seq2'].isin(tr_sequences))]

        subset_df1 = subset_df1[['seq2', 'pid']]
        subset_df1.columns = ['seq', 'pid']
        subset_df2 = subset_df2[['seq1', 'pid']]
        subset_df2.columns = ['seq', 'pid']
        concat_df = pd.concat([subset_df1, subset_df2], ignore_index=True)

        # Group by sequence id and get the max pid for each sequence
        grouped_df = concat_df.groupby('seq')['pid'].max().reset_index()
        
        # Write each sequence and its max pid to the log csv
        for index, row in grouped_df.iterrows():
            writer.writerow([row['seq'], family_id, row['pid']])
        log_file.flush()

        # Remove tmp files from save_path
        tmp_files = glob.glob(f"{save_path}/tmp*")
        for file in tmp_files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting file {file}: {e}")
