from itertools import groupby
import numpy as np
import hmmscan_utils as hu

def identify_hit_no_hit(labels, no_hit_label, target_prob, length_thresh):
    """
    Identifies hit and no hit regions and smooths the labels using Maximum Scoring Segment (MSS).

    Args:
        labels (np.ndarray): The original labels.
        no_hit_label (int): The label value that indicates no hit.
        target_prob (float): The target probability of an insert state.
        length_thresh (int): The length threshold.

    Returns:
        np.ndarray: The smoothed labels.
    """
    # Create a numpy array of 'M's and 'I's based on the condition
    # Condition: Is are predicted no hits.
    labels = np.ravel(labels)
    labels_string_array = np.where(labels == no_hit_label, 'I', 'M')

    # Convert the numpy array to a string
    labels_string = ''.join(labels_string_array)

    # Now pass labels_string to adjust_state_assignment
    hit_no_hit_string = hu.adjust_state_assignment(labels_string, target_prob, length_thresh)

    # Make a copy of labels AND update with MSS no hit assignments
    labels_copy = np.array([no_hit_label if label == 'I' else labels[i] for i, label in enumerate(hit_no_hit_string)])

    return labels_copy

def call_hits(labels_copy, no_hit_label, target_prob, length_thresh):
    """
    Identifies hit regions and adjusts the labels accordingly.

    Args:
        labels_copy (np.ndarray): The copy of original labels.
        no_hit_label (int): The label value that indicates no hit.
        target_prob (float): The target probability of an insert state.
        length_thresh (int): The length threshold.

    Returns:
        np.ndarray: The adjusted labels.
    """
    # Initialize the tracking string with 'X' where labels_copy has no_hit_label and 'O' everywhere else
    tracking_string = ['X' if label == no_hit_label else 'O' for label in labels_copy]

    # counter to track number of recursive domain calls
    counter = 0

    # Continue the process as long as there is an 'O' in tracking_string
    while 'O' in tracking_string:
        if counter >= 5:
            break

        # Initialize the start index
        start_index = 0

        # Iterate over the contiguous blocks of elements in tracking_string where there is an 'O'
        for key, group in groupby(tracking_string, lambda x: x == 'O'):
            # Convert the group to a list
            group = list(group)

            if key:  # If the block consists of 'O'
                # Get the corresponding block from labels_copy
                block = labels_copy[start_index : start_index + len(group)]

                # Find the majority class in the block
                majority_class = max(set(block), key=lambda x: np.count_nonzero(block == x))

                # Create a string representation of the block with the majority class as 'M' and everything else as 'I'
                block_string = ''.join(['M' if label == majority_class else 'I' for label in block])

                # Perform adjust_state_assignment on the block string
                adjusted_block = hu.adjust_state_assignment(block_string, target_prob, length_thresh)

                # Replace the 'M' labels in the original block in labels_copy with the majority class label
                # and mark the modified positions in the tracking string with 'X'
                for i, label in enumerate(adjusted_block):
                    if label == 'M':
                        labels_copy[start_index + i] = majority_class
                        tracking_string[start_index + i] = 'X'

            # Update the start index
            start_index += len(group)

        # Update the counter
        counter += 1
    return labels_copy

def rescore_hits(labels_copy, labels, vals):
    """
    Rescores the hit regions.

    Args:
        labels_copy (np.ndarray): The copy of original labels.
        labels (np.ndarray): The original labels.
        vals (np.ndarray): The original values.

    Returns:
        np.ndarray: The rescored values.
    """
    # Initialize the scores array with the same length as labels_copy and filled with 0
    scores = np.zeros(len(labels_copy))

    # Initialize the start index
    start_index = 0

    # Iterate over the contiguous chunks of same labels in labels_copy
    for label, group in groupby(labels_copy):
        # Convert the group to a list
        group = list(group)

        # Get the corresponding region from labels
        region = labels[start_index : start_index + len(group)]

        # Find the indices in the region that have the desired label
        indices = np.where(region == label)[0]

        # Calculate the percentage of the region that have the desired label
        percentage = len(indices) / len(region)

        # Calculate the average score of the indices that have the desired label
        average_score = np.mean(vals[start_index + indices])

        # Assign the average score to the corresponding positions in the scores array
        scores[start_index : start_index + len(group)] = average_score

        # Update the start index
        start_index += len(group)

    return scores

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

def generate_thresholds(predicted_vals):
    max_val = np.max(predicted_vals)
    thresholds = np.linspace(0, max_val, int(1000/1))
    thresholds = np.append(thresholds, max_val + 0.01)  # Add a value slightly higher than the max value
    return thresholds

def calculate_tpr_fpr(true_labels, predicted_vals, predicted_labels, threshold, family = True, full_length=False):
    no_hit_label = 19632 if family else 656

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

    tpr = tp / (tp + fn + 1e-10)
    fpr = fp / (fp + tn + 1e-10)

    return tpr, fpr