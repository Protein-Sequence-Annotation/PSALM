import torch
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math  # For ceil function

def plot_predictions(fam_preds, clan_preds, maps, threshold, seq_name=None, save_path=None):
    # Helper functions to calculate legend parameters
    def calculate_fontsize(num_labels):
        if num_labels <= 5:
            return 'medium'
        elif num_labels <= 10:
            return 'small'
        elif num_labels <= 20:
            return 'x-small'
        else:
            return 'xx-small'

    def calculate_figheight(num_labels, ncol):
        base_height = 4
        increment_per_13_labels = 0.5
        return base_height + ((num_labels // 13) + 1) * increment_per_13_labels

    def calculate_figwidth(ncol):
        base_width = 8
        increment_per_col = 2.5
        return base_width + ncol * increment_per_col

    def rearrange_handles_labels(handles, labels, ncol, col_length):
        new_handles = []
        new_labels = []
        for row in range(col_length):
            for col in range(ncol):
                idx = col * col_length + row
                if idx < len(handles):
                    new_handles.append(handles[idx])
                    new_labels.append(labels[idx])
        return new_handles, new_labels

    # Get indices of top value for each position for fam
    fam_top_val, fam_top_index = torch.topk(fam_preds, k=1, dim=1)

    # Get number of unique fams predicted
    unique_fam_values = torch.unique(fam_top_index)
    num_unique_fam_values = unique_fam_values.numel()

    # Generate fam colors
    seed = 100
    fam_colors = colormaps.get_cmap('tab20b').resampled(num_unique_fam_values)
    torch.manual_seed(seed)
    randomizer = torch.randperm(num_unique_fam_values)
    fam_c_map = {}

    for c_idx, i in enumerate(randomizer.tolist()):
        fam_c_map[int(unique_fam_values[c_idx])] = fam_colors(i)

    fam_c_map[19632] = 'black'

    # Get indices of top value for each position for clan
    clan_top_val, clan_top_index = torch.topk(clan_preds, k=1, dim=1)

    # Get number of unique clans predicted
    unique_clan_values = torch.unique(clan_top_index)
    num_unique_clan_values = unique_clan_values.numel()

    # Generate clan colors
    clan_colors = colormaps.get_cmap('tab20b').resampled(num_unique_clan_values)
    torch.manual_seed(seed)
    randomizer = torch.randperm(num_unique_clan_values)
    clan_c_map = {}

    for c_idx, i in enumerate(randomizer.tolist()):
        clan_c_map[int(unique_clan_values[c_idx])] = clan_colors(i)

    clan_c_map[655] = 'grey'  # used to be red
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

    # Calculate legend parameters
    num_entries_fam = len(fam_unique_test)
    ncol_fam = math.ceil(num_entries_fam / 13)
    fontsize_fam = calculate_fontsize(num_entries_fam)
    figheight_fam = calculate_figheight(num_entries_fam, ncol_fam)

    num_entries_clan = len(clan_unique_test)
    ncol_clan = math.ceil(num_entries_clan / 13)
    fontsize_clan = calculate_fontsize(num_entries_clan)
    figheight_clan = calculate_figheight(num_entries_clan, ncol_clan)

    figheight = max(figheight_fam, figheight_clan) + 1.0  # Add extra space for x-label
    figwidth = calculate_figwidth(max(ncol_fam, ncol_clan))

    # Create figure and GridSpec
    f = plt.figure(figsize=(figwidth, figheight))
    gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1], figure=f)

    # Create axes for plots
    ax1 = f.add_subplot(gs[0, 0])
    ax2 = f.add_subplot(gs[1, 0])

    # Create axes for legends
    fam_legend_ax = f.add_subplot(gs[0, 1])
    fam_legend_ax.axis('off')
    clan_legend_ax = f.add_subplot(gs[1, 1])
    clan_legend_ax.axis('off')

    # Plot the data
    for entry in fam_unique_test:
        idx = (fam_pred_labels == entry) & (fam_pred_vals >= threshold)
        if torch.count_nonzero(idx) != 0:
            label = fam_keys[entry.item()].split(".")[0]
            ax1.bar(torch.where(idx)[0], fam_pred_vals[idx], width=1.0, color=fam_c_map[entry.item()], label=label)

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

    ax1.set_ylabel('Family')
    ax2.set_ylabel('Clan')

    # Adjust axes
    ax1.grid(False)
    ax2.grid(False)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax1.set_xlim(0, fam_pred_labels.shape[0])
    ax2.set_xlim(0, fam_pred_labels.shape[0])

    ax1.set_ylim(0, 1.2)
    ax2.set_ylim(0, 1.2)

    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Add legends to the legend axes

    # Get handles and labels
    handles_fam, labels_fam = ax1.get_legend_handles_labels()
    handles_clan, labels_clan = ax2.get_legend_handles_labels()

    # Rearrange handles and labels
    col_length = 13  # Maximum entries per column
    rearranged_handles_fam, rearranged_labels_fam = rearrange_handles_labels(
        handles_fam, labels_fam, ncol_fam, col_length
    )
    rearranged_handles_clan, rearranged_labels_clan = rearrange_handles_labels(
        handles_clan, labels_clan, ncol_clan, col_length
    )

    # Add legends
    fam_legend_ax.legend(
        rearranged_handles_fam,
        rearranged_labels_fam,
        loc='upper left',
        fontsize=fontsize_fam,
        ncol=ncol_fam,
        columnspacing=1.0,
        labelspacing=0.5,
        handlelength=1.5,
    )
    clan_legend_ax.legend(
        rearranged_handles_clan,
        rearranged_labels_clan,
        loc='upper left',
        fontsize=fontsize_clan,
        ncol=ncol_clan,
        columnspacing=1.0,
        labelspacing=0.5,
        handlelength=1.5,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/{seq_name}_PSALM.png", bbox_inches='tight')

    plt.show()