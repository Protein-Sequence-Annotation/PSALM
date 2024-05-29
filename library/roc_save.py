from pathlib import Path
import pickle
from Bio import SeqIO
import hmmscan_utils as hu
import analysis_tools as at
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# from itertools import groupby
import sys

split = sys.argv[1]
num_seen = int(sys.argv[2])

# Load results
with open(f"../data/results/clan_finetune/roc_results_{split}.pkl","rb") as f:
    results = pickle.load(f)
with open(f"../data/results/onehot_fts/roc_results_{split}.pkl","rb") as f:
    results_oh = pickle.load(f)

# Get the number of examples seen in train
with open('../data/count_fam_list.pkl','rb') as f:
    count_fam_list = pickle.load(f)

min_lim = num_seen
exclude_all = []
for i in range(min_lim +1):
    exclude_fams = list(count_fam_list[i])
    exclude_all.extend(exclude_fams)

# Get the sequences and append at the residue level
# Initialize empty lists for each array
psalm_f_vals = []
psalm_f_indices = []
psalm_c_vals = []
psalm_c_indices = []
hmmer_f_vals = []
hmmer_f_indices = []
hmmer_c_vals = []
hmmer_c_indices = []
onehot_f_vals = []
onehot_f_indices = []
true_f = []
true_c = []
counter = 0
# Iterate over the results dictionary
for sequence_id in results:
    # if (sequence_id not in clan_seqs):
    if True: #sequence_id in clan_seqs: #True
        trues = np.unique(results[sequence_id]['true_f'])
        if not (np.in1d(trues,exclude_all)).any():
            counter += 1
            # Append each sequence's data to the respective list
            psalm_f_vals += list(results[sequence_id]['psalm_f_vals'])
            psalm_f_indices += list(results[sequence_id]['psalm_f_indices'])
            psalm_c_vals += list(results[sequence_id]['psalm_c_vals'])
            psalm_c_indices += list(results[sequence_id]['psalm_c_indices'])
            hmmer_f_vals += list(results[sequence_id]['hmmer_f_vals'])
            hmmer_f_indices += list(results[sequence_id]['hmmer_f_indices'])
            hmmer_c_vals += list(results[sequence_id]['hmmer_c_vals'])
            hmmer_c_indices += list(results[sequence_id]['hmmer_c_indices'])
            onehot_f_vals += list(results_oh[sequence_id]['psalm_f_vals'])
            onehot_f_indices += list(results_oh[sequence_id]['psalm_f_indices'])
            true_f += list(results[sequence_id]['true_f'])
            true_c += list(results[sequence_id]['true_c'])
            
            if len(true_f) != len(hmmer_f_indices):
                print('Mistmatch', len(list(results[sequence_id]['true_f'])), len(list(results[sequence_id]['hmmer_f_indices'])))

# Convert lists to numpy arrays
psalm_f_vals = np.array(psalm_f_vals)
psalm_f_indices = np.array(psalm_f_indices)
psalm_c_vals = np.array(psalm_c_vals)
psalm_c_indices = np.array(psalm_c_indices)
hmmer_f_vals = np.array(hmmer_f_vals)
hmmer_f_indices = np.array(hmmer_f_indices)
hmmer_c_vals = np.array(hmmer_c_vals)
hmmer_c_indices = np.array(hmmer_c_indices)
onehot_f_vals = np.array(onehot_f_vals)
onehot_f_indices = np.array(onehot_f_indices)
true_f = np.array(true_f)
true_c = np.array(true_c)

# Define functions needed for ROC plots

def true_positives(true_labels, predicted_labels, no_hit_label):
    mask = true_labels != no_hit_label
    return np.sum((true_labels[mask] == predicted_labels[mask]))

def false_negatives(true_labels, predicted_labels, no_hit_label):
    mask = true_labels != no_hit_label
    return np.sum((predicted_labels[mask] == no_hit_label))

def false_positives(true_labels, predicted_labels, no_hit_label, full_length=False):
    if full_length:
        mask = predicted_labels != no_hit_label
        return np.sum(true_labels[mask] != predicted_labels[mask])
    else:
        mask = true_labels == no_hit_label
        return np.sum(predicted_labels[mask] != no_hit_label)

def true_negatives(true_labels, predicted_labels, no_hit_label):
    mask = true_labels == no_hit_label
    return np.sum((true_labels[mask] == predicted_labels[mask]))

def generate_thresholds(predicted_vals,threshold=0):
    # Get the maximum value in the predicted_vals array
    max_val = np.max(predicted_vals)

    # Generate thresholds at 0.1% increments of the maximum value
    thresholds = np.linspace(threshold, max_val, int(1000/1))
    thresholds = np.append(thresholds, max_val + 0.01)

    return thresholds

def calculate_tpr_fpr(true_labels, predicted_vals, predicted_labels, threshold, no_hit_label, full_length=False):
    true_labels = np.ravel(true_labels)
    predicted_vals = np.ravel(predicted_vals)
    predicted_labels = np.ravel(predicted_labels)
    
    threshold_indices = np.where(predicted_vals >= threshold)
    negative_indices = np.where(predicted_vals < threshold)
    
    predicted_labels_threshold = np.zeros_like(predicted_labels)
    predicted_labels_threshold[threshold_indices] = predicted_labels[threshold_indices]
    predicted_labels_threshold[negative_indices] = no_hit_label

    tp = true_positives(true_labels, predicted_labels_threshold, no_hit_label)
    fn = false_negatives(true_labels, predicted_labels_threshold, no_hit_label)
    fp = false_positives(true_labels, predicted_labels_threshold, no_hit_label, full_length=full_length)
    tn = true_negatives(true_labels, predicted_labels_threshold, no_hit_label)

    # tpr = tp / (tp + fn + 1e-10)
    mask = true_labels != no_hit_label
    tp_denom = np.sum(mask)
    tpr = tp/tp_denom
    fpr = fp / (fp + tn + 1e-10)
    tpr_ci = 1.96*np.sqrt(tpr*(1-tpr)/tp_denom)

    return tpr, fpr, tpr_ci

# Calculate clan rocs
thresholds = generate_thresholds(psalm_c_vals,0.72)
roc_values = [calculate_tpr_fpr(true_c, psalm_c_vals, psalm_c_indices,threshold, no_hit_label=656,full_length=False) for threshold in thresholds]
thresholds2 = generate_thresholds(hmmer_c_vals)[:len(thresholds)]
roc_values2 = [calculate_tpr_fpr(true_c, hmmer_c_vals, hmmer_c_indices,threshold, no_hit_label=656,full_length=False) for threshold in thresholds2]
thresholds3 = generate_thresholds(onehot_f_vals)[:len(thresholds)]
roc_values3 = [calculate_tpr_fpr(true_c, onehot_f_vals, onehot_f_indices,threshold, no_hit_label=656,full_length=False) for threshold in thresholds3]

# Save this together in f`../data/full_length_rocs/clan_rocs_{split}_{num_seen}.pkl`
with open(f"../data/full_length_rocs/clan_rocs_{split}_{num_seen}.pkl","wb") as f:
    pickle.dump([roc_values,roc_values2,roc_values3],f)

# Calculate fam rocs
thresholds = generate_thresholds(psalm_f_vals,0.72)
roc_values = [calculate_tpr_fpr(true_f, psalm_f_vals, psalm_f_indices,threshold, no_hit_label=19632,full_length=False) for threshold in thresholds]
thresholds2 = generate_thresholds(hmmer_f_vals)[:len(thresholds)]
roc_values2 = [calculate_tpr_fpr(true_f, hmmer_f_vals, hmmer_f_indices,threshold, no_hit_label=19632,full_length=False) for threshold in thresholds2]
thresholds3 = generate_thresholds(onehot_f_vals)[:len(thresholds)]
roc_values3 = [calculate_tpr_fpr(true_f, onehot_f_vals, onehot_f_indices,threshold, no_hit_label=19632,full_length=False) for threshold in thresholds3]

# Save this together in f`../data/full_length_rocs/fam_rocs_{split}_{num_seen}.pkl`
with open(f"../data/full_length_rocs/fam_rocs_{split}_{num_seen}.pkl","wb") as f:
    pickle.dump([roc_values,roc_values2,roc_values3],f)