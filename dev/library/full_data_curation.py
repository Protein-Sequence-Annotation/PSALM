import pandas as pd
import pdb
import subprocess
from tqdm import tqdm
from Bio import SeqIO, AlignIO
import pickle

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
            # fam_hits[fam] = unique_seqnames.tolist()

            # Initialize a dictionary to store the maximum %id for each sequence name
            max_ids = {}
            # For each unique sequence name
            for seqname in unique_seqnames:
                # Find all rows that have that sequence name in either column and pull the corresponding %id
                ids = df.loc[(df['seqname1'] == seqname) | (df['seqname2'] == seqname), '%id']
                # Store the maximum %id for this sequence name
                max_ids[seqname] = ids.max()
            # # Convert the dictionary to a Series
            max_ids_series = pd.Series(max_ids)
            # # Find the sequence name with the minimum maximum %id
            # seqname_with_min_max_id = max_ids_series.idxmin()
            # corresponding_id = max_ids_series.min()
            # Sort the series in ascending order
            sorted_max_ids_series = max_ids_series.sort_values()

            # Check if there are at least 5 sequences
            if len(sorted_max_ids_series) >= 5:
                # Get the sequence name and corresponding id of the 5th lowest maximum %id
                seqname_with_min_max_id = sorted_max_ids_series.index[4]
                corresponding_id = sorted_max_ids_series.iloc[4]
            else:
                print("There are less than 5 sequences.")
                continue

            if corresponding_id > pid_upper_bound:
                print(seqname_with_min_max_id, corresponding_id, fam)
                continue
            else:
                f.write(f"{seqname_with_min_max_id} {corresponding_id} {fam}\n")
                f.flush()
            # seqname_with_min_max_id = seqname_with_min_max_id.split("/")[0]
            if seqname_with_min_max_id in results:
                results[seqname_with_min_max_id].append((corresponding_id, fam))
            else:
                results[seqname_with_min_max_id] = [(corresponding_id, fam)]
    return results, fam_hits


def get_val_seqs():
    save_path = '../data/full_data_curation'
    nseq_threshold = 20
    nseq_max = 100
    max_pid = 30

    # Load the HMM statistics
    hmmstats = load_hmmstats("/n/eddy_lab/data/pfam-35.0/Pfam-A.hmm.stats")

     # Filter the families
    filtered_families = filter_families(hmmstats, nseq_threshold, nseq_max)

    results, fam_hits = pick_sequence(filtered_families, save_path, max_pid,nseq_max)

    # with open(f"{save_path}/unique_seqnames.txt", "w") as f:
    #     for seqname in results.keys():
    #         f.write(f"{seqname}\n")
    
    with open(f"{save_path}/fam_hits.pkl", "wb") as f:
        pickle.dump(fam_hits, f)
    with open(f"{save_path}/results.pkl", "wb") as f:
        pickle.dump(results, f)

def check_val_seqs():
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
        if total_seqs < 20 and num_hits > 0:
            flagged_families.append(family)
        # Condition 2: More than 20 seqs in alignment and remaining seqs after hits is less than 20
        elif total_seqs >= 20 and (total_seqs - num_hits) < 20:
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

if __name__ == "__main__":
    # get_val_seqs()
    check_val_seqs()