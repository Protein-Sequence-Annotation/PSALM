import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def viewAll(seq, results_clan, results_fam, fam_keys, clan_keys, threshold):

    ##########################################
    # Color generation

    fam_colors = mpl.colormaps.get_cmap('tab20b').resampled(19633) # Listed color map vs linear segmented: add .colors at end

    fam_color_gen = np.random.default_rng(28) #15
    randomizer = np.arange(19632)
    fam_color_gen.shuffle(randomizer)
    # fam_color_gen.shuffle(randomizer)

    fam_c_map = {}
    
    for c_idx, i in enumerate(randomizer):
        fam_c_map[i] = fam_colors(c_idx)

    fam_c_map[19632] = 'black'

    clan_colors = mpl.colormaps.get_cmap('tab20b').resampled(655) # Listed color map vs linear segmented: add .colors at end

    clan_color_gen = np.random.default_rng(42)
    randomizer = np.arange(655)
    clan_color_gen.shuffle(randomizer)

    clan_c_map = {}

    for c_idx, i in enumerate(randomizer):
        clan_c_map[i] = clan_colors(c_idx)

    clan_c_map[655] = 'grey' #used to be red
    clan_c_map[656] = 'black'
    ##############################################

    #############################################
    # Fam results

    fam_pred_labels = results_fam[seq]['fam_idx']
    fam_pred_vals = results_fam[seq]['fam_vals']
    
    fam_pred1_labels = fam_pred_labels[:,0]

    fam_pred1_vals = fam_pred_vals[:,0]

    fam_unique_test_1 = np.unique(fam_pred1_labels)
    if fam_unique_test_1[-1] == 19632:
        fam_unique_test_1 = fam_unique_test_1[:-1]

    ##############################################

    ##############################################
    # Clan results

    clan_pred_labels = results_clan[seq]['clan_idx']
    clan_pred_vals = results_clan[seq]['clan_vals']
    
    clan_pred1_labels = clan_pred_labels[:,0]
    clan_pred1_vals = clan_pred_vals[:,0]

    clan_unique_test_1 = np.unique(clan_pred1_labels)
    if clan_unique_test_1[-1] == 656:
        clan_unique_test_1 = clan_unique_test_1[:-1]
    ###############################################

    ##############################################
    # Plot all
    f, (ax2, ax3) = plt.subplots(2,1, figsize=(15,4))

    for entry in fam_unique_test_1:
        idx = np.where((fam_pred1_labels==entry) & (fam_pred1_vals >= threshold))[0]
        if idx.shape[0] != 0:
            ax2.bar(idx, fam_pred1_vals[idx], width=1.0, color=fam_c_map[entry], label=fam_keys[entry])

    for entry in clan_unique_test_1:
        idx = np.where((clan_pred1_labels==entry) & (clan_pred1_vals >= threshold))[0]
        if idx.shape[0] != 0:
            ax3.bar(idx, clan_pred1_vals[idx], width=1.0, color=clan_c_map[entry], label=clan_keys[entry])
    #############################################

    #############################################
    # Axis labels
    ax3.set_xlabel('Position')
    ax2.set_title(f'Sequence {seq}')

    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax2.set_ylabel('PSALM fam')
    ax3.set_ylabel('PSALM clan')

    ax2.grid(False)
    ax3.grid(False)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    ax2.set_xlim(0,fam_pred_labels.shape[0])
    ax3.set_xlim(0,clan_pred_labels.shape[0])

    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])

    ax2.set_ylim(0,1.2)
    ax3.set_ylim(0,1.2)

    ############################################

    return f