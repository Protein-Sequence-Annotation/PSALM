import hmmscan_utils as hu
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm

def shard_predict(shard, target_path, pred_path, maps):

    shard_preds = {}

    target_dict = hu.parse_hmmscan_results(target_path / f'split_{shard}_test_ids_full.fasta_scan.txt')
    pred_dict = hu.parse_hmmscan_results(pred_path / f'split_{shard}_test_ids_full.fasta_scan.txt', pred=True)

    for label in target_dict.keys():

        shard_preds[label] = {}
        
        target_vector, _ = hu.generate_domain_position_list2(target_dict, label, maps)
        pred_vector, _ = hu.generate_domain_position_list2(pred_dict, label, maps) # Account for no hits

        stop_index_target = min(target_dict[label]['length'], 4096) # Hard coded limit currently
        stop_index_pred = min(pred_dict[label]['length'], 4096)
        
        target_vector = np.argmax(target_vector[:stop_index_target,:], axis=1)
        pred_vector = np.argmax(pred_vector[:stop_index_pred,:], axis=1)

        shard_preds[label]['top'] = (target_vector == pred_vector).mean()

        fams = np.unique(target_vector)
        top_total = 0.

        for fam in fams:
            idx = (target_vector == fam)            
            top_total += (target_vector[idx] == pred_vector[idx]).mean()

        shard_preds[label]['fam_top'] = (top_total / fams.shape[0])

        adjusted_score = 0.
        dubious_pos = 0

        for i in range(target_vector.shape[0]):
            if target_vector[i] == pred_vector[i]:
                adjusted_score += 1
            elif (target_vector[i] == pred_vector[max(0,i-5):i+1]).any() or \
                (target_vector[i] == pred_vector[i:min(i+5,target_vector.shape[0])]).any():
                adjusted_score += 1

        shard_preds[label]['adjusted_acc'] = adjusted_score / (target_vector.shape[0]-dubious_pos+1e-3)

        non_idr = target_vector != 19632
        shard_preds[label]['non_idr_top'] = (target_vector[non_idr] == pred_vector[non_idr]).mean()

    return shard_preds

def hmm_predict():

    all_preds = {}

    num_shards = 50
    target_path = Path('../data/test_scan_OLD')
    pred_path = Path('../data/test_hmmer_preds')

    with open('../data/maps.pkl', 'rb') as f:
        maps = pickle.load(f)

    for shard in tqdm(range(1, num_shards+1), total=num_shards, desc='Shards processed:'):

        shard_preds = shard_predict(shard, target_path, pred_path, maps)

        all_preds = all_preds | shard_preds

    with open(f'../data/results/hmmer_preds/fam_results.pkl', 'wb') as f:
        pickle.dump(all_preds, f)

    return

if __name__ == '__main__':

    hmm_predict()