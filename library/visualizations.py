import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import numpy as np
import pickle
from collections import defaultdict
import polars as pl

def clan_accuracies(result_path: Path, plot=False):

    clan_top = defaultdict(list)
    clan_top2 = defaultdict(list)

    top_acc = []
    top2_acc = []
    set_acc = []
    set_acc_strict = []

    adjusted_acc = []

    shards = sorted(filter(lambda x: x[:5] == 'shard', os.listdir(result_path)), key=lambda x: int(x[:-4].split('_')[-1]))
    print(f'Data from {len(shards)} shards')

    for shard in shards:

        with open(result_path / shard, 'rb') as f:
            results = pickle.load(f)

        for seq in results.keys():
            first = results[seq]['clan_idx'][:,0] == results[seq]['clan_true']
            second = results[seq]['clan_idx'][:,1] == results[seq]['clan_true']

            top_acc.append(first.mean())
            top2_acc.append((first+second).mean())

            clans = np.unique(results[seq]['clan_true'])
            pred_unique = np.unique(results[seq]['clan_idx'])
            
            common = np.intersect1d(clans, pred_unique)
            set_acc.append(common/clans.shape[0]) # fraction of matching clans
            set_acc_strict.append((clans.shape[0] == pred_unique.shape[0]) and common.shape[0] == clans.shape[0])

            for clan in clans:
                idx = (results[seq]['clan_true'] == clan)
                first = results[seq]['clan_idx'][:,0][idx] == results[seq]['clan_true'][idx]
                second = results[seq]['clan_idx'][:,1][idx] == results[seq]['clan_true'][idx]
                clan_top[clan].append(first.mean())
                clan_top2[clan].append((first+second).mean())
            
            adjusted_score = 0.
            dubious_pos = 0
            # If positions match, then full score
            # If within a 10 residue window (+5, -5), then full score
            # If confused between IDR and NC, ignore position for accuracy calculation
            for i in range(results[seq]['clan_true'].shape[0]):
                if results[seq]['clan_true'][i] == results[seq]['clan_idx'][i,0]:
                    adjusted_score += 1
                elif (results[seq]['clan_true'][i] == results[seq]['clan_idx'][max(0,i-5):i+1,0]).any() or \
                    (results[seq]['clan_true'][i] == results[seq]['clan_idx'][i:min(i+5,results[seq]['clan_true'].shape[0]),0]).any():
                    adjusted_score += 1
                elif results[seq]['clan_true'][i] == 656 and results[seq]['clan_idx'][i,0] == 655 and results[seq]['clan_idx'][i,1] == 656:
                    dubious_pos += 1
                elif results[seq]['clan_true'][i] == 655 and results[seq]['clan_idx'][i,0] == 656 and results[seq]['clan_idx'][i,1] == 655:
                    dubious_pos += 1

            adjusted_acc.append(adjusted_score / (results[seq]['clan_true'].shape[0]-dubious_pos+1e-3))
            if adjusted_score / (results[seq]['clan_true'].shape[0]-dubious_pos + 1e-3) < 0.4:
                print(shard, seq)
         
    to_save = {
        'top': top_acc,
        'top2': top2_acc,
        'clan_top': clan_top,
        'clan_top2': clan_top2,
        'set_acc': set_acc,
        'set_strict': set_acc_strict,
        'adjusted_acc': adjusted_acc
    }

    with open(result_path / f'agg_results.pkl', 'wb') as f:
        pickle.dump(to_save , f)
    
    if plot:
        sns.set_style('ticks')
        
        f, ax = plt.subplots()

        ax.hist(top_acc, range=(0,1), bins=20)
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Proportion of Scores')
        ax.set_title(f'Prediction Accuracy Over Test Sequences: Avg = {np.nanmean(top_acc)}')

        ax.grid(False)
        sns.despine(top=True, right=True)

        f.savefig(result_path / 'top_accuracy.png')
        plt.close(f)

        f, ax = plt.subplots()

        ax.hist(top2_acc, range=(0,1), bins=20)
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Proportion of Scores')
        ax.set_title(f'Prediction Accuracy Over Test Sequences: Avg = {np.nanmean(top2_acc)}')

        ax.grid(False)
        sns.despine(top=True, right=True)

        f.savefig(result_path / 'top2_accuracy.png')
        plt.close(f)

        clan_idx = list(clan_top.keys())
        
        clan_means = np.array([np.nanmean(clan_top[k]) for k in clan_idx])
        clan_std = np.array([np.nanstd(clan_top[k]) for k in clan_idx])

        f, ax = plt.subplots(figsize=(15,8))

        ax.bar(clan_idx, clan_means)
        ax.set_xlabel('Clan')
        ax.set_ylabel('Average accuracy')
        ax.set_title(f'Prediction Accuracy Over Each Clan')

        ax.grid(False)
        sns.despine(top=True, right=True)

        f.savefig(result_path / 'clan_accuracy.png')
        plt.close(f)

    return


def viewSingleClan(shard, seq, results, clan_keys):

    import matplotlib as mpl
    colors = mpl.colormaps.get_cmap('tab20b').resampled(655) # Listed color map vs linear segmented: add .colors at end

    # np.random.seed(1)
    randomizer = np.arange(655)
    # np.random.shuffle(randomizer)
    c_map = {}

    for c_idx, i in enumerate(randomizer):
        c_map[i] = colors(c_idx)

    c_map[655] = 'red'
    c_map[656] = 'black'

    true = results[seq]['clan_true']
    pred_labels = results[seq]['clan_idx']
    pred_vals = results[seq]['clan_vals']
    
    pred1_labels = pred_labels[:,0]
    pred2_labels = pred_labels[:,1]
    pred1_vals = pred_vals[:,0]
    pred2_vals = pred_vals[:,1]

    unique_test_1 = np.unique(pred1_labels)
    unique_test_2 = np.unique(pred2_labels)

    unique_target = np.unique(true)

    first = true == pred1_labels
    second = true == pred2_labels
    non_idr_idx = true != 656
    non_idr_first = true[non_idr_idx] == pred1_labels[non_idr_idx]
    non_idr_second = true[non_idr_idx] == pred2_labels[non_idr_idx]
    
    f, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(15,9))

    for entry in unique_target:
        idx = np.where(true==entry)[0]
        # ax1.fill_between(idx, 0,1, color=c_map[entry])
        ax1.bar(idx, 1, width=1.0, color=c_map[entry], label=clan_keys[entry])

    for entry in unique_test_1:
        idx = np.where(pred1_labels==entry)[0]
        # ax2.fill_between(idx, 0,pred1_vals[idx], color=c_map[entry])
        ax2.bar(idx, pred1_vals[idx], width=1.0, color=c_map[entry], label=clan_keys[entry])

    for entry in unique_test_2:
        idx = np.where(pred2_labels==entry)[0]
        # ax3.fill_between(idx, 0,pred2_vals[idx], color=c_map[entry])
        ax3.bar(idx, pred2_vals[idx], width=1.0, color=c_map[entry], label=clan_keys[entry])

    ax3.set_xlabel('Position')
    ax1.set_title(f'Shard {shard}: Sequence {seq}    Top: {first.mean():.2f}    Top 2: {(first+second).mean():.2f}    Non IDR Top: {non_idr_first.mean():.2f}    Non IDR Top 2: {(non_idr_first+non_idr_second).mean():.2f}')
    ax1.legend(ncol=min(10, unique_target.shape[0]), loc='upper center')
    ax2.legend(ncol=min(10, unique_test_1.shape[0]), loc='upper center')
    ax3.legend(ncol=min(10, unique_test_2.shape[0]), loc='upper center')

    ax1.set_ylabel('Target')
    ax2.set_ylabel('Predicted (Top)')
    ax3.set_ylabel('Predicted (Second)')

    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)

    ax1.set_xlim(0,pred_labels.shape[0])
    ax2.set_xlim(0,pred_labels.shape[0])
    ax3.set_xlim(0,pred_labels.shape[0])

    ax1.set_ylim(0,1.2)
    ax2.set_ylim(0,1.2)
    ax3.set_ylim(0,0.7) # Second probability only at most 0.5, so reduced y lim

    return

def viewSingleFam(shard, seq, results, fam_keys):

    import matplotlib as mpl
    colors = mpl.colormaps.get_cmap('tab20b').resampled(19633) # Listed color map vs linear segmented: add .colors at end

    # np.random.seed(1)
    randomizer = np.arange(19632)
    np.random.shuffle(randomizer)
    c_map = {}

    for c_idx, i in enumerate(randomizer):
        c_map[i] = colors(c_idx)

    c_map[19632] = 'black'

    true = results[seq]['fam_true']
    pred_labels = results[seq]['fam_idx']
    pred_vals = results[seq]['fam_vals']
    true_vals = results[seq]['fam_true_vals']
    
    pred1_labels = pred_labels[:,0]
    pred2_labels = pred_labels[:,1]
    pred1_vals = pred_vals[:,0]
    pred2_vals = pred_vals[:,1]

    unique_test_1 = np.unique(pred1_labels)
    unique_test_2 = np.unique(pred2_labels)

    unique_target = np.unique(true)

    first = true == pred1_labels
    second = true == pred2_labels
    non_idr_idx = true != 19632
    non_idr_first = true[non_idr_idx] == pred1_labels[non_idr_idx]
    non_idr_second = true[non_idr_idx] == pred2_labels[non_idr_idx]
    
    f, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(15,9))

    for entry in unique_target:
        idx = np.where(true==entry)[0]
        # ax1.fill_between(idx, 0,1, color=c_map[entry])
        ax1.bar(idx, true_vals[idx], width=1.0, color=c_map[entry], label=fam_keys[entry])

    for entry in unique_test_1:
        idx = np.where(pred1_labels==entry)[0]
        # ax2.fill_between(idx, 0,pred1_vals[idx], color=c_map[entry])
        ax2.bar(idx, pred1_vals[idx], width=1.0, color=c_map[entry], label=fam_keys[entry])

    for entry in unique_test_2:
        idx = np.where(pred2_labels==entry)[0]
        # ax3.fill_between(idx, 0,pred2_vals[idx], color=c_map[entry])
        ax3.bar(idx, pred2_vals[idx], width=1.0, color=c_map[entry], label=fam_keys[entry])

    ax3.set_xlabel('Position')
    ax1.set_title(f'Shard {shard}: Sequence {seq}    Top: {first.mean():.2f}    Top 2: {(first+second).mean():.2f}    Non IDR Top: {non_idr_first.mean():.2f}    Non IDR Top 2: {(non_idr_first+non_idr_second).mean():.2f}')
    ax1.legend(ncol=min(10, unique_target.shape[0]), loc='upper center')
    ax2.legend(ncol=min(10, unique_test_1.shape[0]), loc='upper center')
    ax3.legend(ncol=min(10, unique_test_2.shape[0]), loc='upper center')

    ax1.set_ylabel('Target')
    ax2.set_ylabel('Predicted (Top)')
    ax3.set_ylabel('Predicted (Second)')

    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)

    ax1.set_xlim(0,pred_labels.shape[0])
    ax2.set_xlim(0,pred_labels.shape[0])
    ax3.set_xlim(0,pred_labels.shape[0])

    ax1.set_ylim(0,1.2)
    ax2.set_ylim(0,1.2)
    ax3.set_ylim(0,0.7) # Second probability only at most 0.5, so reduced y lim

    return

if __name__ == '__main__':

    dpath = Path('../data/results/weighted_lstm_run1/predictions_esm2_t33_650M_UR50D/')
    clan_accuracies(dpath, False)