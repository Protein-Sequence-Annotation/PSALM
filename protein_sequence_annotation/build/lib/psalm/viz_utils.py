import torch
from matplotlib import colormaps
import matplotlib.pyplot as plt

def plot_predictions(fam_preds, clan_preds, maps, threshold, seq_name=None, save_path=None):
    # Get indices of top two values for each position for fam
    fam_top_val, fam_top_index = torch.topk(fam_preds,k=1,dim=1)

    # Get number of fams predicted in top two
    unique_fam_values = torch.unique(fam_top_index)
    num_unique_fam_values = unique_fam_values.numel()

    # Generate fam colors
    seed = 100
    fam_colors = colormaps.get_cmap('tab20b').resampled(num_unique_fam_values)
    torch.manual_seed(seed)
    randomizer = torch.randperm(num_unique_fam_values)
    fam_c_map = {}

    for c_idx, i in enumerate(randomizer.tolist()):  # Convert tensor to list for enumeration
        fam_c_map[int(unique_fam_values[c_idx])] = fam_colors(i)  # Normalize the index to get a color from the colormap

    fam_c_map[19632] = 'black'

    # Get indices of top value for each position for clan
    clan_top_val, clan_top_index = torch.topk(clan_preds, k=1, dim=1)

    # Get number of clans predicted in top one
    unique_clan_values = torch.unique(clan_top_index)
    num_unique_clan_values = unique_clan_values.numel()

    # Generate clan colors
    clan_colors = colormaps.get_cmap('tab20b').resampled(num_unique_clan_values)
    randomizer = torch.randperm(num_unique_clan_values)
    clan_c_map = {}

    for c_idx, i in enumerate(randomizer.tolist()):  # Convert tensor to list for enumeration
        clan_c_map[int(unique_clan_values[c_idx])] = clan_colors(i)

    clan_c_map[655] = 'grey' #used to be red
    clan_c_map[656] = 'black'

    # Fam results
    fam_keys = maps['idx_fam']
    clan_keys = maps['idx_clan']

    fam_pred_labels = fam_top_index
    fam_pred_vals = fam_top_val
    clan_pred_labels = clan_top_index
    clan_pred_vals = clan_top_val

    fam_unique_test = torch.unique(fam_pred_labels)
    clan_unique_test = torch.unique(clan_pred_labels)

    if fam_unique_test[-1] == 19632:
        fam_unique_test = fam_unique_test[:-1]
    if clan_unique_test[-1] == 656:
        clan_unique_test = clan_unique_test[:-1]

    # Plot all
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 4))

    for entry in fam_unique_test:
        idx = (fam_pred_labels == entry) & (fam_pred_vals >= threshold)
        if torch.count_nonzero(idx) != 0:
            ax1.bar(torch.where(idx)[0], fam_pred_vals[idx], width=1.0, color=fam_c_map[entry.item()], label=fam_keys[entry.item()].split(".")[0])

    for entry in clan_unique_test:
        idx = (clan_pred_labels == entry) & (clan_pred_vals >= threshold)
        if torch.count_nonzero(idx) != 0:
            label = clan_keys[entry.item()].split(".")[0]
            if label == "NC0001":
                label = "Non-clan"
            ax2.bar(torch.where(idx)[0], clan_pred_vals[idx], width=1.0, color=clan_c_map[entry.item()], label=label)

    # Axis labels
    ax2.set_xlabel('Position')
    ax1.set_title(f'{seq_name}' if seq_name else 'Sequence')

    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),ncol=2)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5),ncol=2)

    ax1.set_ylabel('Family')
    ax2.set_ylabel('Clan')

    ax1.grid(False)
    ax2.grid(False)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax1.set_xlim(0, fam_pred_labels.shape[0])
    ax2.set_xlim(0, fam_pred_labels.shape[0])

    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])

    ax1.set_ylim(0, 1.2)
    ax2.set_ylim(0, 1.2)
    if save_path:
        plt.tight_layout()
        plt.savefig(f"{save_path}/{seq_name}_PSALM.png")

    plt.tight_layout()
    plt.show()