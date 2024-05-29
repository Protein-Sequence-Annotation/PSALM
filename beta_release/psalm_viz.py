import hmmscan_utils as hu
import classifiers as cf
import ml_utils as mu
from parsers import single_parser
import visualizations as vz

import torch
import pickle
import matplotlib.pyplot as plt

parser = single_parser()
args = parser.parse_args()

with open('info_files/shard_seqs.pkl', 'rb') as f:
    shard_seqs = pickle.load(f)

with open(f'info_files/maps.pkl', 'rb') as g:
        maps = pickle.load(g)

fam_map = list(maps['fam_idx'].keys())
clan_map = list(maps['clan_idx'].keys())

device = 'cuda' if torch.cuda.is_available() else 'cpu'

length_limit = 4096 # Covers 99.75% sequences
esm_model_name =  'esm2_t33_650M_UR50D'
num_shards = args.num_shards
data_utils = mu.DataUtils(f'{args.root}', num_shards, esm_model_name, length_limit, 'test', device,"_OLD")

fam_classifier = cf.FamLSTM(data_utils.embedding_dim, data_utils.maps, device).to(device)
fam_classifier.load_state_dict(torch.load(args.fam_filename))

clan_classifier = cf.ClanLSTM(data_utils.embedding_dim,data_utils.clan_count).to(device)
clan_classifier.load_state_dict(torch.load(args.clan_filename))

fam_classifier.eval()
clan_classifier.eval()

with torch.inference_mode():

    if args.input == 'none':

        # shard = np.random.randint(1,args.num_shards+1) # Choose random shard
        # # shard = 17 # Select specific shard

        # hmm_dict = data_utils.parse_shard(shard)
        
        # keys = list(hmm_dict.keys())
        # key_idx = np.random.randint(0,len(keys))
        # seq_id = keys[key_idx]

        seq_id = 'I4D8X5.1' # Select specific sequence from this shard
        shard = shard_seqs[seq_id]
        hmm_dict = data_utils.parse_shard(shard)
        dataset = data_utils.get_dataset(shard)
        dataset = data_utils.filter_batches(dataset, [seq_id]) ############### Need to fix this filtering
    else:
         dataset = data_utils.get_custom_dataset(args.input)
        
    data_loader = data_utils.get_dataloader(dataset)
    label, seq, token = next(iter(data_loader))
    if args.input != 'none':
         seq_id = label[0]

    token = token.to(device)
    embedding = data_utils.get_embedding(token)
    stop_index = min(len(seq[0]), data_utils.length_limit)

    clan_preds = clan_classifier(embedding["representations"][data_utils.last_layer][0,1:stop_index+1,:])
    clan_preds = torch.nn.functional.softmax(clan_preds, dim=1)

    _, raw_preds = fam_classifier(embedding["representations"][data_utils.last_layer][0,1:stop_index+1,:], clan_preds)

    clan_fam_weights = data_utils.maps['clan_family_matrix'].to(device)
   
    for i in range(clan_fam_weights.shape[0]): #shape is cxf (ignore IDR bc 1:1 map)
        indices = torch.nonzero(clan_fam_weights[i]).squeeze() #indices for the softmax
        if i == 656:
            raw_preds[:,indices] = 1
        else:
            raw_preds[:,indices] = torch.softmax(raw_preds[:,indices],dim=1)

    # Multiply by clan, expand clan preds
    clan_preds_f = torch.matmul(clan_preds,clan_fam_weights)

    # element wise mul of preds and clan
    fam_preds = raw_preds * clan_preds_f

    # Fam Results
    results_fam = {}
    results_fam[seq_id] = {}
    top_two_vals, top_two_indices = torch.topk(fam_preds, k=2, dim=1)

    results_fam[seq_id]['fam_vals'] = top_two_vals.cpu().numpy()
    results_fam[seq_id]['fam_idx'] = top_two_indices.cpu().numpy()

    results_fam['label'] = seq_id
    
    # Clan Results
    results_clan = {}
    results_clan[seq_id] = {}
    top_two_vals, top_two_indices = torch.topk(clan_preds, k=2, dim=1)

    results_clan[seq_id]['clan_vals'] = top_two_vals.cpu().numpy()
    results_clan[seq_id]['clan_idx'] = top_two_indices.cpu().numpy()

    results_clan['label'] = seq_id

fig = vz.viewAll(seq_id, results_clan, results_fam, fam_map, clan_map, 'test_pids.pkl', args.thresh)
fig.savefig(f"results/plots/{seq_id.replace('.', '_')}.png")
plt.close(fig)