import sys
import numpy as np
sys.path.insert(0, '../library')
sys.path.insert(0, '../py_scripts')
import hmmscan_utils as hu
import classifiers as cf
import ml_utils as mu
import visualizations as vz
import torch
import pickle
from pathlib import Path

with open(f'../data_esm_decoder/maps.pkl', 'rb') as g:
        maps = pickle.load(g)
fam_map = list(maps['fam_idx'].keys())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

length_limit = 4096 # Covers 99.75% sequences
model_name =  'esm2_t33_650M_UR50D' #'esm2_t36_3B_UR50D'
num_shards = 50 ###### Replace with appropriate number ###################################
data_utils = mu.DataUtils('../data', num_shards, model_name, length_limit, 'test', device)

classifier = cf.FamModelSimple(data_utils.embedding_dim, data_utils.maps, device).to(device)
classifier_path = Path('../data/results/Fam_Simple_run2/epoch_3.pth')
classifier.load_state_dict(torch.load(classifier_path))

classifier.eval()

with torch.inference_mode():

    shard = np.random.randint(1,51)
    # shard = 47

    hmm_dict = data_utils.parse_shard(shard)
    keys = list(hmm_dict.keys())
    key_idx = np.random.randint(0,len(keys))
    seq_id = keys[key_idx]
    # seq_id = 'A6GJW3.1'

    dataset = data_utils.get_dataset(shard)
    dataset = data_utils.filter_batches(dataset, [seq_id])
    data_loader = data_utils.get_dataloader(dataset)

    label, seq, token = next(iter(data_loader))
    token = token.to(device)
    embedding = data_utils.get_embedding(token)
    fam_vector, clan_vector = hu.generate_domain_position_list2(hmm_dict, seq_id, data_utils.maps)
                    
    stop_index = min(len(seq[0]), data_utils.length_limit)
    fam_vector = torch.tensor(fam_vector[:stop_index,:]).to(device) # clip the clan_vector to the truncated sequence length
    clan_vector = torch.tensor(clan_vector[:stop_index,:]).to(device)

    preds, _ = classifier(embedding["representations"][data_utils.last_layer][0,1:stop_index+1,:], clan_vector)
    preds = torch.nn.functional.softmax(preds, dim=1)
    preds[preds < 0.01] = 0
    preds = torch.nn.functional.normalize(preds, dim=1)
 
    results = {}
    results[seq_id] = {}
    top_two_vals, top_two_indices = torch.topk(preds, k=2, dim=1)

    for i, row in enumerate(top_two_vals):
         print(i, row)

    results[seq_id]['fam_vals'] = top_two_vals.cpu().numpy()
    results[seq_id]['fam_idx'] = top_two_indices.cpu().numpy()
    results[seq_id]['fam_true'] = torch.argmax(fam_vector, dim=1).cpu().numpy()
    results[seq_id]['fam_true_vals'] = fam_vector.max(dim=1)[1].cpu().numpy()
    results['label'] = seq_id
    results['shard'] = shard

    save_path = classifier_path.parent
    with open(save_path / f'single_pred.pkl', 'wb') as f:
        pickle.dump(results, f)