from Bio import SeqIO, SearchIO
from tqdm import tqdm
import pickle
import subprocess
import os
import glob
# import pdb

hmmfetch_command = "/n/eddy_lab/software/bin/hmmfetch"
hmmsearch_command = "/n/eddy_lab/software/bin/hmmsearch"
reformat_command = "/n/eddy_lab/software/bin/esl-reformat"
align_command = "/n/eddy_lab/software/bin/hmmalign"
hmmbuild_command = "/n/eddy_lab/software/bin/hmmbuild"
hmmpress_command = "/n/eddy_lab/software/bin/hmmpress"
hmmstat_command = "/n/eddy_lab/software/bin/hmmstat"
incE = str(0.001)
bitscore_threshold = 30

save_path = "/n/eddy_lab/Lab/protein_annotation_dl/PSALM/data/rebuilt_hmms"
hmm_db = "/n/eddy_lab/data/pfam-35.0/Pfam-A.hmm"
new_hmm_db = f"{save_path}/rebuilt_profiles.hmm"
train_fasta_path = "/n/eddy_lab/Lab/protein_annotation_dl/PSALM/data/train_fasta/train_ids_full.fasta"
train_db = f"{save_path}/cropped_train.fasta"
maps_path = "/n/eddy_lab/Lab/protein_annotation_dl/PSALM/data/maps.pkl"

output_hmm = f"{save_path}/tmp.hmm"
output_tr_sto = f"{save_path}/tmp_tr_msa.sto"
output_tr_domtbl = f"{save_path}/tmp_tr_domtbl.txt"
output_tr_fasta = f"{save_path}/tmp_tr.fasta"
new_hmm = f"{save_path}/tmp_train.hmm"

# Truncate train sequences to 4096 residues
with open(train_db, "w") as output_handle:
    for record in SeqIO.parse(train_fasta_path, "fasta"):
        if len(record.seq) > 4096:
            record.seq = record.seq[:4096]
        SeqIO.write(record, output_handle, "fasta")

# Get list of family IDs
with open(maps_path, "rb") as f:
    maps = pickle.load(f)
family_ids = list(maps['fam_idx'].keys())[:-1] #ignore the IDR key
# Open new hmm database file
with open(new_hmm_db, "w") as hmmfile:
    # Iterate through each family
    for family_id in tqdm(family_ids):
    # family_id = "PF10417.12"
    # family_id = "PF00008.30"

        # hmmfetch profile for selected family
        try:
            subprocess.run([hmmfetch_command, "-o", output_hmm, hmm_db, family_id], stdout=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f"Error fetching {family_id} hmm from database")

        # hmmsearch profile against training database and return MSA of hits
        try:
            subprocess.run([hmmsearch_command, "-A", output_tr_sto, "--incE", incE, "--domtblout", output_tr_domtbl, output_hmm, train_db], stdout=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f"Error searching {family_id} hmm against train database")

        # Open the train domain table file and extract the IDs of the hits that have a bitscore > 30
        tr_high_score_hits = []
        for query in SearchIO.parse(output_tr_domtbl, 'hmmsearch3-domtab'):
            for hit in query.hits:
                for hsp in hit.hsps:
                    if hsp.evalue <= float(incE) and hsp.bitscore >= bitscore_threshold:
                        tr_high_score_hits.append(f"{hit.id}/{hsp.hit_start+1}-{hsp.hit_end}")
                # # Find the hsp with the smallest evalue and filter by bitscore ***MODEL ONLY SEES SINGLE STRONGEST HIT***
                # best_hsp = min((hsp for hsp in hit.hsps if hsp.bitscore >= bitscore_threshold), key=lambda hsp: hsp.evalue, default=None)
                # if best_hsp is not None:
                #     tr_high_score_hits.append(f"{hit.id}/{best_hsp.hit_start+1}-{best_hsp.hit_end}")
        if not tr_high_score_hits:
            print("Skipped family: ", family_id)
            continue
        # reformat train stockholm files to fasta format
        try:
            subprocess.run([reformat_command, "--informat", "stockholm", "-o", output_tr_fasta, "fasta", output_tr_sto], check=True)
        except:
            print(f"Error reformatting {family_id} train stockholm to fasta")

        # Remove sequences are not in high_score_hits
        tr_sequences = list(SeqIO.parse(output_tr_fasta, "fasta"))
        tr_sequences = [seq for seq in tr_sequences if seq.id in tr_high_score_hits]

        # Using famil_id in names to be 1) less confusing and 2) have the description of the new hmm be correct (for retrieval purposes)
        hsh_fasta = f"{save_path}/{family_id}.fasta"
        with open(hsh_fasta, "w+") as f:
            SeqIO.write(tr_sequences, f, "fasta") 

        # hmmalign the file to the corresponding profile hmm
        hsh_align = f"{save_path}/{family_id}.sto"
        try:
            subprocess.run([align_command, "-o", hsh_align, output_hmm, hsh_fasta], check=True)
        except:
            print(f"Error aligning {family_id} sequences")

        # Build a new profile from the aligned sequences
        try:
            subprocess.run([hmmbuild_command,  new_hmm, hsh_align],stdout=subprocess.DEVNULL, check=True)
        except:
            print(f"Error building {family_id} profile")
        
        # Append the contents of new_hmm to hmmfile
        with open(new_hmm, 'r') as f:
            hmmfile.write(f.read())
        hmmfile.flush()

        # Remove temporary files from save_path
        # Remove temporary and family specific files from save_path
        files_to_remove = glob.glob(f"{save_path}/tmp*") + glob.glob(f"{save_path}/{family_id}*")
        for file in files_to_remove:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting file {file}: {e}")

# Index and press the new database
try:
    subprocess.run([hmmfetch_command,"--index", new_hmm_db], check=True)
    subprocess.run([hmmpress_command, new_hmm_db], check=True)
except:
    print(f"Error indexing and pressing new database")

# Get stats
with open(f"{new_hmm_db}.stats", "w") as f:
    try:
        subprocess.run([hmmstat_command, new_hmm_db], stdout=f, check=True)
    except:
        print(f"Error getting stats for new database")