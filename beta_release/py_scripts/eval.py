import pickle
from Bio import SeqIO
from tqdm import tqdm
import numpy as np
import sys

####################
# Analysis functions
####################

def true_positives(true_labels, predicted_labels, no_hit_label):
    mask = true_labels != no_hit_label
    return np.sum(true_labels[mask] == predicted_labels[mask])

def false_negatives(true_labels, predicted_labels, no_hit_label):
    mask = true_labels != no_hit_label
    return np.sum(predicted_labels[mask] == no_hit_label)

def false_positives(true_labels, predicted_labels, no_hit_label, full_length=False):
    if full_length:
        mask = predicted_labels != no_hit_label
        return np.sum(true_labels[mask] != predicted_labels[mask])
    else:
        mask = true_labels == no_hit_label
        return np.sum(predicted_labels[mask] != no_hit_label)

def true_negatives(true_labels, predicted_labels, no_hit_label):
    mask = true_labels == no_hit_label
    return np.sum(true_labels[mask] == predicted_labels[mask])

def generate_thresholds(predicted_vals, threshold=0):
    max_val = np.max(predicted_vals)
    thresholds = np.linspace(threshold, max_val, int(1000/1))
    thresholds = np.append(thresholds, max_val + 0.01)
    return thresholds

def calculate_f1(tp, fn, fp):
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-5)
    return f1

def calculate_mcc_scaled(tp, fn, fp, tn):
    # Small constant added to prevent division by zero
    epsilon = 0

    # Find the maximum value to scale down TP, FN, FP, TN
    max_val = max(tp, fn, fp, tn, 1)  # Use 1 as a fallback to avoid division by zero
    
    # Scale down TP, FN, FP, TN by the maximum value
    tp_scaled = tp / max_val
    fn_scaled = fn / max_val
    fp_scaled = fp / max_val
    tn_scaled = tn / max_val

    # Calculate the numerator
    numerator = (tp_scaled * tn_scaled) - (fp_scaled * fn_scaled)

    # Calculate the denominator with added epsilon to prevent division by zero
    denominator = np.sqrt((tp_scaled + fp_scaled + epsilon) * (tp_scaled + fn_scaled + epsilon) * 
                          (tn_scaled + fp_scaled + epsilon) * (tn_scaled + fn_scaled + epsilon))

    # Return MCC scaled back by multiplying with max_val
    return numerator / denominator if denominator != 0 else 0

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

    mask = true_labels != no_hit_label
    tp_denom = np.sum(mask)
    tpr = tp / tp_denom
    fpr = fp / (fp + tn + 1e-10)
    tpr_ci = 1.96 * np.sqrt(tpr * (1 - tpr) / tp_denom)
    f1 = calculate_f1(tp,fn,fp)
    mcc = calculate_mcc_scaled(tp,fn,fp,tn)

    return tpr, fpr, tpr_ci,f1,mcc

def calculate_normalized_auc(fprs, tprs):
    # Convert the lists to numpy arrays for easier manipulation
    fprs = np.array(fprs)
    tprs = np.array(tprs)

    # Calculate the area under the curve using the trapezoidal rule
    auc = np.trapz(tprs, fprs)

    # Normalize the AUC by the range of FPR (the x-axis), typically from 0 to 1
    fpr_range = fprs[-1] - fprs[0]  # Usually, fpr_range is 1 if FPR spans from 0 to 1

    normalized_auc = auc / fpr_range

    return normalized_auc

def check(pid, psalm_dict):


    # Load the pickled dictionary
    with open(f'datasets/Val_{pid}.pkl', 'rb') as file:
        pickled_dict = pickle.load(file)

    # Load sequences from Test_20.fasta
    fasta_sequences = SeqIO.parse(f'datasets/Test_{pid}.fasta', 'fasta')
    test_sequences = {record.id for record in fasta_sequences}

    # Filter psalm_dict and pickled_dict to retain only the sequences present in Test_20.fasta
    psalm_dict_filtered = {key: value for key, value in psalm_dict.items() if key in test_sequences}
    pickled_dict_filtered = {key: value for key, value in pickled_dict.items() if key in test_sequences}

    # Check if any keys in the filtered pickled dictionary are present in the filtered psalm_dict
    overlapping_keys = set(pickled_dict_filtered.keys()) & set(psalm_dict_filtered.keys())

    if overlapping_keys:
        print(f"Warning: The following keys are present in both dictionaries: {overlapping_keys}")
        sys.exit(0)
    else:
        print("No overlapping keys found. You're good to go!")
        
#########################
# Processing Utils
#########################

def merge(psalm_dict):

    # Initialize empty lists for each array
    psalm_f_vals = []
    psalm_f_indices = []
    psalm_c_vals = []
    psalm_c_indices = []
    true_f = []
    true_c = []

    for sequence_id in psalm_dict:
        psalm_f_vals += list(psalm_dict[sequence_id]['fam_preds'])
        psalm_f_indices += list(psalm_dict[sequence_id]['fam_idx'])
        psalm_c_vals += list(psalm_dict[sequence_id]['clan_preds'])
        psalm_c_indices += list(psalm_dict[sequence_id]['clan_idx'])
        true_f += list(psalm_dict[sequence_id]['fam_true'])
        true_c += list(psalm_dict[sequence_id]['clan_true'])

    # Convert lists to numpy arrays
    psalm_f_vals = np.array(psalm_f_vals)
    psalm_f_indices = np.array(psalm_f_indices)
    psalm_c_vals = np.array(psalm_c_vals)
    psalm_c_indices = np.array(psalm_c_indices)
    true_f = np.array(true_f)
    true_c = np.array(true_c)

    return psalm_f_vals, psalm_f_indices, psalm_c_vals, psalm_c_indices, true_f, true_c

#########################
# Main
#########################

def main(root, hmmer=False):

    metrics_dict = {}
    dataset_name = root.split('/')[-1]

    for pid in [20]:#['20', '40', '60', '80', '100']:

        metrics_dict[pid] = {}

        pred_path = f"{root}/preds_{pid}.pkl"

        with open(pred_path, "rb") as f:
            result_dict = pickle.load(f)

        check(pid, result_dict)

        psalm_f_vals, psalm_f_indices, psalm_c_vals, psalm_c_indices, true_f, true_c = merge(result_dict)

        thresholds = generate_thresholds(psalm_f_vals,0)
        fam_roc_values = [calculate_tpr_fpr(true_f, psalm_f_vals, psalm_f_indices,threshold, no_hit_label=19632,full_length=False) for threshold in thresholds]

        thresholds = generate_thresholds(psalm_c_vals,0)
        clan_roc_values = [calculate_tpr_fpr(true_c, psalm_c_vals, psalm_c_indices,threshold, no_hit_label=656,full_length=False) for threshold in thresholds]

        metrics_dict[pid]['fam_vals'] = fam_roc_values
        metrics_dict[pid]['clan_vals'] = clan_roc_values

    
    np.save(f'metrics/{dataset_name}.npy', metrics_dict, allow_pickle=True)


if __name__ == '__main__':

    main(sys.argv[1], False)