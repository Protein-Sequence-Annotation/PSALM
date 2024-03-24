from Bio import SeqIO, SearchIO
from tqdm import tqdm
import pickle
import subprocess
import os
import glob
# import pdb

hmmfetch_command = "/n/eddy_lab/software/bin/hmmfetch"
sfetch_command = "/n/eddy_lab/software/bin/esl-sfetch"
hmmsearch_command = "/n/eddy_lab/software/bin/hmmsearch"
reformat_command = "/n/eddy_lab/software/bin/esl-reformat"
align_command = "/n/eddy_lab/Lab/protein_annotation_dl/mafft-7.525-with-extensions/scripts/mafft"
hmmbuild_command = "/n/eddy_lab/software/bin/hmmbuild"
hmmpress_command = "/n/eddy_lab/software/bin/hmmpress"
hmmstat_command = "/n/eddy_lab/software/bin/hmmstat"
incE = str(0.001)
bitscore_threshold = 30

save_path = "/n/eddy_lab/Lab/protein_annotation_dl/PSALM/data/clan_hmms"
hmm_db = "/n/eddy_lab/data/pfam-35.0/Pfam-A.hmm"
new_hmm_db = f"{save_path}/clan_profiles.hmm"
train_fasta_path = "/n/eddy_lab/Lab/protein_annotation_dl/PSALM/data/train_fasta/train_ids_full.fasta"
train_db = f"{save_path}/cropped_train.fasta"
maps_path = "/n/eddy_lab/Lab/protein_annotation_dl/PSALM/data/maps.pkl"

output_hmm = f"{save_path}/tmp.hmm"
output_tr_sto = f"{save_path}/tmp_tr_msa.sto"
output_tr_domtbl = f"{save_path}/tmp_tr_domtbl.txt"
output_tr_fasta = f"{save_path}/tmp_tr.fasta"
new_hmm = f"{save_path}/tmp_train.hmm"


# # Truncate train sequences to 4096 residues
# with open(train_db, "w") as output_handle:
#     for record in SeqIO.parse(train_fasta_path, "fasta"):
#         if len(record.seq) > 4096:
#             record.seq = record.seq[:4096]
#         SeqIO.write(record, output_handle, "fasta")

# Index the train_db
try:
    subprocess.run([sfetch_command, "--index", train_db], check=True)
except:
    print(f"Error indexing {train_db}")

# Get list of family IDs
with open(maps_path, "rb") as f:
    maps = pickle.load(f)
clan_ids = list(maps['clan_fam'].keys())[:-1] #ignore the IDR key
# Open new hmm database file
with open(new_hmm_db, "w") as hmmfile:
    # Iterate through each clan
    for clan_id in tqdm(clan_ids):
        family_ids = maps['clan_fam'][clan_id]
        # High scoring hits for the clan
        tr_high_score_hits = []
        # Iterate through each family
        for family_id in family_ids:

            # hmmfetch profile for selected family
            try:
                subprocess.run([hmmfetch_command, "-o", output_hmm, hmm_db, family_id], stdout=subprocess.DEVNULL, check=True)
            except subprocess.CalledProcessError:
                print(f"Error fetching {family_id} hmm from database")

            # hmmsearch profile against training database and return MSA of hits
            try:
                subprocess.run([hmmsearch_command, 
                                "-A", output_tr_sto, 
                                "--incE", incE, 
                                "--domtblout", output_tr_domtbl, output_hmm, train_db], stdout=subprocess.DEVNULL, check=True)
            except subprocess.CalledProcessError:
                print(f"Error searching {family_id} hmm against train database")

            # Open the train domain table file and extract the IDs of the hits that have a bitscore > 30
            for query in SearchIO.parse(output_tr_domtbl, 'hmmsearch3-domtab'):
                for hit in query.hits:
                    for hsp in hit.hsps:
                        if hsp.evalue <= float(incE) and hsp.bitscore >= bitscore_threshold:
                            # Format allows esl-sfetch -Cf to be used to extract the subsequences
                            tr_high_score_hits.append(f"{hit.id}/{hsp.hit_start+1}-{hsp.hit_end} {hsp.hit_start+1} {hsp.hit_end} {hit.id}")
            if not tr_high_score_hits:
                print("Skipped family: ", family_id)
                continue
            
            # Remove temporary files needed in fam processing
            files_to_remove = glob.glob(f"{save_path}/tmp*") + glob.glob(f"{save_path}/{family_id}*")
            for file in files_to_remove:
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"Error deleting file {file}: {e}")
    
        # Write tr_high_score_hits to a file
        with open(f"{save_path}/{clan_id}_high_score_hits.txt", "w") as f:
            f.write("\n".join(tr_high_score_hits))
        
        # sfetch subsequences from train_db
        try:
            subprocess.run([sfetch_command,
                            "-o",f"{save_path}/{clan_id}_high_score_hits.fasta", 
                            "-Cf", train_db, f"{save_path}/{clan_id}_high_score_hits.txt"], check=True)
        except:
            print(f"Error sfetching subsequences from train_db")

        # Align all hits in clan using MAFFT w/ G-INS-i, which assumes that entire region can be aligned w/ Needleman-Wunsch algorithm
        # Higher accuracy, lower speed, memory intensive
        with open(f"{save_path}/{clan_id}.afa", "w") as f:
            try:
                subprocess.run([align_command, "--globalpair", "--maxiterate", "1", "--thread", "-8", 
                                f"{save_path}/{clan_id}_high_score_hits.fasta"],stdout=f, check=True)
            except:
                print(f"Error aligning {clan_id} high score hits")

        # Build a new profile from the aligned sequences
        try:
            subprocess.run([hmmbuild_command,  new_hmm, f"{save_path}/{clan_id}.afa"],stdout=subprocess.DEVNULL, check=True)
        except:
            print(f"Error building {clan_id} profile")
        # break
        
        # Append the contents of new_hmm to hmmfile
        with open(new_hmm, 'r') as f:
            hmmfile.write(f.read())
        hmmfile.flush()

        # Remove temporary files from save_path
        # Remove temporary and family specific files from save_path
        files_to_remove = glob.glob(f"{save_path}/tmp*") + glob.glob(f"{save_path}/{clan_id}*")
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