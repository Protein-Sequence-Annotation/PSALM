from Bio import SeqIO, SearchIO
import numpy as np
from numba import jit

##############################################################################################################
# Functions helpful in generating domain and clan vectors
##############################################################################################################

def identify_domains_at_position(hmmscan_dict, query_sequence, position):
    
    """
    Identify all domain hits at given residue position

    Args:
        hmmscan_dict (dict): A dictionary containing the parsed hmmscan results. The keys are the sequence IDs, and the values are
                            dictionaries containing the sequence length and a list of hit domains. Each hit domain is represented as
                            a tuple containing the query start position, query end position, bit score, hit ID, and translated query
                            sequence.
        query_sequence (str): A string with the ID of the query sequence
        position (int): The current residue position being evaluated

    Returns:
        domains (list): List of tuples containing ID, bit score, M/I state for each hit
    """
    
    domains = []

    # Scan through all hits
    for domain in hmmscan_dict[query_sequence]["hit_domains"]:
        
        domain_start, domain_stop, score, name, match_states = domain
        
        # Identify if position is within domain bounds
        if domain_start <= position < domain_stop:
            domains.append((name, score, match_states[position-domain_start]))
    
    return domains

def calculate_prob(scores):
    
    """
    Calculates the domain probabilities of scores at a given position.

    Parameters:
    scores (list or numpy.ndarray): The scores to calculate probabilities for.

    Returns:
    numpy.ndarray: An array of probabilities corresponding to the input scores.
    """
    
    # Convert scores to numpy array
    scores = np.array(scores)
    # Find maximum score
    max_score = np.max(scores)
    # Subtract maximum score from all scores
    scores = scores - max_score
    # Calculate exponentiated scores
    exp_scores = np.exp2(scores)
    # Calculate sum of exponentiated scores
    sum_exp_scores = np.sum(exp_scores)
    # Calculate probabilities
    probs = exp_scores / sum_exp_scores
    
    return probs

def modify_clan_vector(clans, scores, clan_vector):

    """
    Adjusts probabilities for families belonging to same clan.
    Computes the sum over family probabilities and uses that for all
    entries of the corresponding clan

    Args:
        clans (np.ndarray): a vector containing clan index for each family
        scores (np.ndarray): a vector containing probability for each family
        clan_vector (np.ndarray): placeholder for vector of adjusted clan probabilities 

    Returns:
        clan_vector (np.ndarray): vector containing adjusted scores for each clan index 
    """

    clan_values = np.unique(clans)
    
    for val in clan_values:

        clan_vector[val] = scores[clans == val].sum()

    return clan_vector

def generate_domain_position_list(hmmscan_dict, query_sequence, maps, indices = False):

    """
    Create domain and clan probability vectors for a given query sequence

    Args:
        hmmscan_dict (dict): A dictionary containing the parsed hmmscan results. The keys are the sequence IDs, and the values are
                            dictionaries containing the sequence length and a list of hit domains. Each hit domain is represented as
                            a tuple containing the query start position, query end position, bit score, hit ID, and translated query
                            sequence.
        query_sequence (str): A string with the ID of the query sequence
        maps (dict): A dictionary containing maps from family -> index, family -> clan
                    and clan -> index, each stored as dictionaries
    
    Returns:
        domain_vector (np.ndarray): Matrix of dimension num_families x query_length with
                                    probability score for each family
        clan_vector (np.ndarray): Matrix of dimension num_clans x query_length with
                                    probability score for each clan
    """


    # Generate domain map (change to family map later)
    family_mapping = maps['fam_idx']
    clan_mapping = maps['clan_idx']
    fam_clan = maps['fam_clan']
    
    # Change this to torch tensor later
    domain_vector = np.zeros(shape=(len(family_mapping),hmmscan_dict[query_sequence]["length"]),dtype=np.float32)
    clan_vector = np.zeros(shape=(len(clan_mapping),hmmscan_dict[query_sequence]["length"]),dtype=np.float32)

    # Generate domain vector
    for i in range(hmmscan_dict[query_sequence]["length"]):
        domains = identify_domains_at_position(hmmscan_dict, query_sequence, i)

        # If there are no domains at this position, set to 0
        if len(domains) == 0:
            domains = [("IDR",50, "M")] #high score just to make sure IDR has highest mass
        # Otherwise, calculate the probabilities for the relevant domains
        else:
            # If one of the domains at this position contain both M and I states, drop the I states from the domains list
            if any('I' in domain[2] for domain in domains) and any('M' in domain[2] for domain in domains):
                domains = [domain for domain in domains if 'M' in domain[2]]
            
        # Unzip the domains list
        domain_names, scores, __ = zip(*domains)
        # Calculate the probabilities
        scores = calculate_prob(scores)

        # Get the index of the domains in the family mapping
        domain_idx = [family_mapping[domain] for domain in domain_names]
        clan_dom = [clan_mapping[fam_clan[domain]] for domain in domain_names]

        # Set the probabilities in the domain and clan vectors
        domain_vector[domain_idx,i] = scores
        clan_vector[:, i] = modify_clan_vector(clan_dom, scores, clan_vector[:,i])
    clan_vector = np.argmax(clan_vector, axis=0) #return indices of max clan probabilities
    if indices:
        domain_vector = np.argmax(domain_vector, axis=0) #return indices of max family probabilities
        return domain_vector, clan_vector
    else:
        return domain_vector.T, clan_vector
def generate_domain_position_list2(hmmscan_dict, query_sequence, maps):

    """
    Create domain and clan probability vectors for a given query sequence

    Args:
        hmmscan_dict (dict): A dictionary containing the parsed hmmscan results. The keys are the sequence IDs, and the values are
                            dictionaries containing the sequence length and a list of hit domains. Each hit domain is represented as
                            a tuple containing the query start position, query end position, bit score, hit ID, and translated query
                            sequence.
        query_sequence (str): A string with the ID of the query sequence
        maps (dict): A dictionary containing maps from family -> index, family -> clan
                    and clan -> index, each stored as dictionaries
    
    Returns:
        domain_vector (np.ndarray): Matrix of dimension num_families x query_length with
                                    probability score for each family
        clan_vector (np.ndarray): Matrix of dimension num_clans x query_length with
                                    probability score for each clan
    """


    # Generate domain map (change to family map later)
    family_mapping = maps['fam_idx']
    clan_mapping = maps['clan_idx']
    fam_clan = maps['fam_clan']
    
    # Change this to torch tensor later
    domain_vector = np.zeros(shape=(len(family_mapping),hmmscan_dict[query_sequence]["length"]),dtype=np.float32)
    clan_vector = np.zeros(shape=(len(clan_mapping),hmmscan_dict[query_sequence]["length"]),dtype=np.float32)

    # Generate domain vector
    for i in range(hmmscan_dict[query_sequence]["length"]):
        domains = identify_domains_at_position(hmmscan_dict, query_sequence, i)

        # If there are no domains at this position, set to 0
        if len(domains) == 0:
            domains = [("IDR",50, "M")] #high score just to make sure IDR has highest mass
        # Otherwise, calculate the probabilities for the relevant domains
        else:
            # If one of the domains at this position contain both M and I states, drop the I states from the domains list
            if any('I' in domain[2] for domain in domains) and any('M' in domain[2] for domain in domains):
                domains = [domain for domain in domains if 'M' in domain[2]]
            
        # Unzip the domains list
        domain_names, scores, __ = zip(*domains)
        # Calculate the probabilities
        scores = calculate_prob(scores)

        # Get the index of the domains in the family mapping
        domain_idx = [family_mapping[domain] for domain in domain_names]
        clan_dom = [clan_mapping[fam_clan[domain]] for domain in domain_names]

        # Set the probabilities in the domain and clan vectors
        domain_vector[domain_idx,i] = scores
        clan_vector[:, i] = modify_clan_vector(clan_dom, scores, clan_vector[:,i])
    # clan_vector = np.argmax(clan_vector, axis=0) #return indices of max clan probabilities
    
    return domain_vector.T, clan_vector.T

##############################################################################################################
# Functions helpful in parsing hmmscan results
##############################################################################################################

def parse_hmmscan_results(file_path: str, e_value_threshold: float=0.01, target_prob: float=0.85, length_thresh: int=20, score_threshold: float=30, pred: bool=False) -> dict:
    """
    Parses the results of an hmmscan search from a file and returns a dictionary containing the parsed information.

    Args:
        file_path (str): The path to the hmmscan results file.
        e_value_threshold (float): The threshold value for the E-value. Hit domains with E-values below this threshold will be included in the results.
        target_prob (float): The target probability of an insert state. Should be set according to expert knowledge.
        length_thresh (int): The length threshold for adjusting M/I states in a Maximally Scoring Segment (MSS). Should be set according to expert knowledge.

    Returns:
        dict: A dictionary containing the parsed hmmscan results. The keys are the sequence IDs, and the values are
              dictionaries containing the sequence length and a list of hit domains. Each hit domain is represented as
              a tuple containing the query start position, query end position, bit score, hit ID, and translated query
              sequence.

    """
    hmmscan_results = SearchIO.parse(file_path, "hmmer3-text")
    hmmscan_dict = {}
    for result in hmmscan_results:
        hit_domains = []
        for hit in result.hits:
            for hsps in hit.hsps:
                if hsps.evalue <= e_value_threshold and hsps.bitscore > score_threshold:
                    hit_domains.append((hsps.query_start,
                                        hsps.query_end,
                                        hsps.bitscore,
                                        hit.id,
                                        translate_to_MID(str(hsps.query.seq),target_prob, length_thresh)))
        hit_domains = sorted(hit_domains, key=lambda x: x[0])  # Sort by query_start in increasing order
        if len(hit_domains) > 0:
            hmmscan_dict[result.id] = {}
            hmmscan_dict[result.id]['length'] = result.seq_len
            hmmscan_dict[result.id]["hit_domains"] = hit_domains
        elif pred:
            hmmscan_dict[result.id] = {}
            hmmscan_dict[result.id]['length'] = result.seq_len
            hmmscan_dict[result.id]["hit_domains"] = [(-2,-1,50,'IDR', 'M')] # Some dummy IDR data
        else:
            continue

    return hmmscan_dict

def translate_to_MID(sequence: str, target_prob: float, length_thresh: int) -> str:
    """
    Translates a string by:
    1) replacing each character with 'M' if it is an uppercase letter (match),
    'I' if it is a lowercase letter (insert), or 'D' if it is a hyphen (deletion).
    2) smoothing out the M/I states based on maximal scoring segments (MSS).

    Args:
        sequence (str): The input string to be translated.
        target_prob (float): The target probability of an insert state.
        length_thresh (int): The length threshold for adjusting M/I states.

    Returns:
        str: The translated string.
    """
    translation_table = str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-',
                                      'M'*26 + 'I'*26 + 'D')
    translated_string = sequence.translate(translation_table)
    # remove 'D's from translated_string
    translated_string = translated_string.replace('D','')
    # Smooth M/I states out
    adjusted_string = adjust_state_assignment(translated_string, target_prob, length_thresh)
    return adjusted_string

def adjust_state_assignment(seq: str, target_prob: float, length_thresh: int) -> str:
    """
    Computes the adjusted sequence by assigning match (M) or insert (I) based on the maximum scoring segment.
    The insert state target probability and length threshold can be used to compute the score for match and insert states.

    Args:
        seq (str): The input sequence.
        target_prob (float): The target probability of an insert state.
        length_thresh (int): The length threshold. 

    Returns:
        str: The adjusted sequence.
    """
  
    # Calculate probability state occurs naturally/by chance
    p_i = seq.count('I') / len(seq)
    if p_i == 0:
        p_i = 10E-6
    if p_i == 1:
        p_i = 1 - 10E-6
    p_m = 1 - p_i
  
    # Set target frequencies
    q_i = target_prob
    if q_i == 0:
        q_i = 10E-6
    if q_i == 1:
        q_i = 1 - 10E-6
    q_m = 1 - q_i

    # Calculate scores (ala Karlin and Altschul (1990))
    # "Methods for assessing the statistical significance of molecular sequence features by using general scoring schemes"
    i_score = np.log2(q_i / p_i)
    m_score = np.log2(q_m / p_m)

    if i_score < m_score:
        base_state_switch = True
    else:
        base_state_switch = False
    # print(f"p_i: {p_i}, p_m: {p_m}, q_i: {q_i}, q_m: {q_m}")
    # print(f"i_score: {i_score}, m_score: {m_score}")

    quantified_seq = np.array([score_states(residue, m_score, i_score) for residue in seq])
    
    running_count = accumulate_scores(quantified_seq)
    
    adjusted_seq = find_MSS(running_count, len(seq), length_thresh, base_state_switch=base_state_switch)

    # print('Adj', adjusted_seq, len(adjusted_seq))
    # print('Org', seq, len(seq))

    return adjusted_seq

def score_states(residue: str, m_score: int, i_score: int) -> int:
    """
    Convert matches and inserts to their corresponding scores

    Args:
        residue (str): The residue type ('M' for match or 'I' for insert)
        m_score (int): The score for a match
        i_score (int): The score for an insert

    Returns:
        int: The corresponding score based on the residue type
    """

    if residue == 'M':
        return m_score
    elif residue == 'I':
        return i_score
    else:
        return 0

@jit(nopython=True)
def accumulate_scores(seq: np.array) -> np.array:
    """
    Modified cumulative sum ensuring that sum does not become negative
    
    Args:
        seq (np.array): The input sequence of scores
        
    Returns:
        np.array: The accumulated sequence where each element is the modified cumulative sum of the input sequence
    """
    counts = np.empty_like(seq) # Create output storage array
    total = 0 # Accumulator

    for i, score in enumerate(seq):
        total += score # Add to accumulator
        
        if total < 0: # Reset if total is negative
            total = 0
            
        counts[i] = total # Assign modified total to accumulator

    return counts

def find_MSS(running_count: np.array, total_length: int, length_thresh: int, base_state_switch: bool=False) -> str:
    """
    Finds the maximum scoring segments and populate a match/insert string accordingly

    Args:
    running_count (np.array): An array containing the running count of scores.
    total_length (int): The total length of the string.
    length_thresh (int): The minimum length threshold for considering a segment.

    Returns:
    str: The adjusted string with 'M' for match and 'I' for insert segments.
    """
    if base_state_switch:
        base_state = 'I'
        replace_state = 'M'
    else:
        base_state = 'M'
        replace_state = 'I'
    adjusted_string = [base_state]* total_length

    zeros = np.where(running_count == 0)[0]
    if zeros[-1] != total_length-1:
        zeros = np.append(zeros,total_length-1) # add end even if it doesnt hit zero

    # print(f"zeros: {zeros}")
    # print(f"running_count: {running_count}")
    for idx, pos in enumerate(zeros[:-1]):
        # If consecutive zeros, skip
        if zeros[idx+1] == zeros[idx] + 1:
            continue

        # Find max score in between zeros
        start = zeros[idx]
        bound = zeros[idx+1]
        # print(start,bound)
        stop = start + np.argmax(running_count[start:bound+1])
        # print(start,stop,bound)
        if stop - start >= length_thresh:
            adjusted_string[start+1:stop+1] = replace_state*(stop-start)
    # If seq starts with replace state, reflect that in adjusted string
    if zeros[0] != 0:
        stop = np.argmax(running_count[:zeros[0]])+1# Find max score before first zero
        # print(f"early stop: {stop}")
        adjusted_string[:stop] = replace_state*(stop) # max score position is in right state, replace until that position
    return ''.join(adjusted_string)

def get_I_lengths(hmm_dict: dict) -> list:
    """
    Get the lengths of all 'I' sections in the 'MID' strings of the hit domains in the hmm_dict.

    Parameters:
    hmm_dict (dict): A dictionary containing hit domains information.

    Returns:
    list: A list of lengths of all 'I' sections.

    """
    MIDs = [domain[4] for key in hmm_dict for domain in hmm_dict[key]['hit_domains']]
    lengths = []
    for mid in MIDs:
        mid = mid.replace('D','')
        sections = mid.split('M')
        for section in sections:
            if 'I' in section:
                lengths.append(len(section))
    return lengths

def get_I_frequency(hmm_dict: dict) -> list:
    """
    Get the lengths of all 'I' sections in the 'MID' strings of the hit domains in the hmm_dict.

    Parameters:
    hmm_dict (dict): A dictionary containing hit domains information.

    Returns:
    list: A list of lengths of all 'I' sections.

    """
    MIDs = [domain[4] for key in hmm_dict for domain in hmm_dict[key]['hit_domains']]
    counts = []
    lengths = []

    for mid in MIDs:   

        count = mid.count('I') / len(mid)
        counts.append(count)
        lengths.append(len(mid))

    return counts, lengths

##############################################################################################################
# Functions helpful in filtering/preprocessing Pfam-A.seed
##############################################################################################################

def split_fasta_file(fasta_file: str, data_dir: str, num_files: int) -> None:
    """
    Splits a fasta file into multiple smaller fasta files.

    Args:
        fasta_file (str): The name of the input fasta file.
        data_dir (str): The directory where the fasta file is located.
        num_files (int): The number of smaller fasta files to create.

    Returns:
        None
    """
    records = list(SeqIO.parse(f"{data_dir}/{fasta_file}" , "fasta"))
    total_records = len(records)
    records_per_file = total_records // num_files

    for i in range(num_files):
        start = i * records_per_file
        end = start + records_per_file if i < (num_files-1) else None  # the last file may contain more records
        SeqIO.write(records[start:end], f"{data_dir}/split_{i+1}_{fasta_file}", "fasta")

def find_missing_sequences(hmmscan_file: str, fasta_file: str) -> list:
    """
    Finds missing sequences in the hmmscan results.

    Args:
        hmmscan_file (str): Path to the hmmscan results file.
        fasta_file (str): Path to the fasta file containing sequences.

    Returns:
        list: List of sequence IDs that are missing in the hmmscan results.
    """
    hmmscan_dict = parse_hmmscan_results(hmmscan_file)
    # for key, value in hmmscan_dict.items():
    #     if len(value.get('hit_domains', [])) == 0:
    #         print(f"No hit domains found for key: {key}")
    
    fasta_sequences = [x.id for x in SeqIO.parse(fasta_file, "fasta")]
    if len(hmmscan_dict) == len(fasta_sequences):
        print("All sequences are present in the hmmscan results.", len(hmmscan_dict), len(fasta_sequences))
        return []
    else:
        missing_sequences = []
        for sequence in fasta_sequences:
            if sequence not in hmmscan_dict:
                missing_sequences.append(sequence)
        
    return missing_sequences

    