import math
from typing import Dict, List, Tuple, Set, Any
from numba import njit
import numpy as np
import torch

# Background composition frequencies (SwissProt 2025_01, 207.6M residues) in ESM tokenizer order:
# 
BG_COMPOSITION = np.array([
    0.096495625,  # L
    0.082562749,  # A
    0.070735432,  # G
    0.068567626,  # V
    0.066578663,  # S
    0.067174660,  # E
    0.055283010,  # R
    0.053651091,  # T
    0.059082818,  # I
    0.054629495,  # D
    0.047469135,  # P
    0.057994329,  # K
    0.039318590,  # Q
    0.040633456,  # N
    0.038693761,  # F
    0.029245925,  # Y
    0.024122393,  # M
    0.022787971,  # H
    0.011052128,  # W
    0.013878132,  # C
    0.000038748,  # X
    0.000001329,  # B
    0.000001594,  # U
    0.000001199,  # Z
    0.000000140,  # O
], dtype=float)

def compute_prefix_counts(seqs: List[str]) -> np.ndarray:
    """
    Given a list of protein sequences (using the 20 canonical AAs + 5 others), returns a
    NumPy array of shape (N, L_max, 25) where N = len(seqs), L_max = max length.
    prefix_counts[i, t, k] = number of occurrences of amino acid k in seqs[i]
    up to and including position t.  If t >= len(seqs[i]), we just repeat the
    final count (i.e. pad with the last prefix sum).
    """
    # in order of ESM tokenizer (https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/esm/configuration_esm.py)
    AMINO_ACIDS = [
        "L",
        "A",
        "G",
        "V",
        "S",
        "E",
        "R",
        "T",
        "I",
        "D",
        "P",
        "K",
        "Q",
        "N",
        "F",
        "Y",
        "M",
        "H",
        "W",
        "C",
        "X",
        "B",
        "U",
        "Z",
        "O",
    ]
    AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    NUM_AMINO_ACIDS = len(AMINO_ACIDS)

    N = len(seqs)
    L_max = max(len(s) for s in seqs)
    # initialize all zeros
    prefix = np.zeros((N, L_max, NUM_AMINO_ACIDS), dtype=np.int64)

    for i, seq in enumerate(seqs):
        counts = np.zeros(NUM_AMINO_ACIDS, dtype=np.int64)
        for t in range(L_max):
            if t < len(seq):
                aa = seq[t]
                idx = AA_TO_IDX.get(aa)
                if idx is not None:
                    counts[idx] += 1
                # else: non-canonical AA, just skip it (no count increment)
            # record the current counts at position t
            prefix[i, t, :] = counts

    return prefix

@njit(fastmath=True)
def compute_bias_slice_nb(
    prefix_counts: np.ndarray,
    start: int,
    stop: int,
    bg_freq: np.ndarray,
    smoothing: float = 1e-9
) -> float:
    """
    Numba-compiled log-likelihood ratio in bits with pseudocount smoothing.
    prefix_counts: (L_max, 20) cumulative counts
    start, stop: 1-based inclusive bounds
    bg_freq: (20,) background frequencies
    """
    L_max, C = prefix_counts.shape
    # 1) bounds check
    if start < 1 or stop < start or stop > L_max:
        raise IndexError("start/stop out of range")
    # 2) make 0-based
    start0 = start - 1
    stop0  = stop  - 1

    # 3) raw counts in window
    counts = np.empty(C, np.float32)
    if start0 > 0:
        for k in range(C):
            counts[k] = prefix_counts[stop0, k] - prefix_counts[start0-1, k]
    else:
        for k in range(C):
            counts[k] = prefix_counts[stop0, k]

    # 4) add smoothing & adjust length
    for k in range(C):
        counts[k] += smoothing
    window_len = (stop - start + 1) + smoothing * C

    # 5) accumulate LLR
    llr = 0.0
    for k in range(C):
        obs_freq = counts[k] / window_len
        # math.log2 is supported in nopython mode
        llr += counts[k] * (math.log2(obs_freq) - math.log2(bg_freq[k]))

    return llr

@njit(fastmath=True)
def compute_bias_combined_nb(
    prefix_counts: np.ndarray,  # shape (L_max, C)
    starts:        np.ndarray,  # shape (R,), 1-based inclusive
    stops:         np.ndarray,  # shape (R,), 1-based inclusive
    bg_freq:       np.ndarray,  # shape (C,), >0, sums to 1
    smoothing:     float
) -> float:
    """
    Compute LLR in bits over multiple discontiguous regions by
    summing raw counts and window lengths, then applying smoothing.
    """
    L_max, C = prefix_counts.shape
    R = starts.shape[0]

    # 1) accumulate raw counts & lengths
    total_len = 0.0
    total_counts = np.zeros(C, np.float32)
    for i in range(R):
        s = starts[i] - 1  # to 0-based
        e = stops[i]  - 1
        # bounds check
        if s < 0 or e >= L_max or s > e:
            # you could raise, but njit can't raise custom errors,
            # so here we just skip invalid regions
            continue

        # add this window’s counts
        if s > 0:
            for k in range(C):
                total_counts[k] += prefix_counts[e, k] - prefix_counts[s-1, k]
        else:
            for k in range(C):
                total_counts[k] += prefix_counts[e, k]
        total_len += (e - s + 1)

    # 2) smoothing: give every AA a tiny pseudocount
    total_len += smoothing * C
    for k in range(C):
        total_counts[k] += smoothing

    # 3) compute LLR in bits
    inv_ln2 = 1.0 / math.log(2.0)
    llr = 0.0
    for k in range(C):
        # observed frequency
        obs = total_counts[k] / total_len
        # safe because everything > 0 now
        llr += total_counts[k] * (math.log(obs) - math.log(bg_freq[k])) * inv_ln2

    return llr

@njit(fastmath=True)
def compute_bias_slice_nb_JSD(
    prefix_counts: np.ndarray,
    start: int,
    stop: int,
    bg_freq: np.ndarray,
    smoothing: float = 1e-9
) -> float:
    """
    Numba-compiled Jensen–Shannon divergence in bits between the
    observed AA distribution in [start..stop] and bg_freq, with pseudocount smoothing.
    prefix_counts: (L_max, C) cumulative counts
    start, stop: 1-based inclusive bounds
    bg_freq: (C,) background frequencies
    """
    L_max, C = prefix_counts.shape
    # 1) bounds check
    if start < 1 or stop < start or stop > L_max:
        raise IndexError("start/stop out of range")
    # 2) to 0-based
    s0 = start - 1
    e0 = stop  - 1

    # 3) raw counts in window
    counts = np.empty(C, np.float32)
    if s0 > 0:
        for k in range(C):
            counts[k] = prefix_counts[e0, k] - prefix_counts[s0-1, k]
    else:
        for k in range(C):
            counts[k] = prefix_counts[e0, k]

    # 4) add smoothing & compute obs_freq
    total = 0.0
    for k in range(C):
        counts[k] += smoothing
        total += counts[k]
    obs_freq = np.empty(C, np.float32)
    for k in range(C):
        obs_freq[k] = counts[k] / total

    # 5) Jensen–Shannon divergence (base-2 logs → bits)
    jsd = 0.0
    for k in range(C):
        o = obs_freq[k]
        b = bg_freq[k]
        m = 0.5 * (o + b)
        # o>0 always true (smoothing), b>0 by assumption
        jsd += 0.5 * ( o * (math.log2(o) - math.log2(m))
                     + b * (math.log2(b) - math.log2(m)) )
    return jsd

@njit(fastmath=True)
def compute_bias_combined_JSD_nb(
    prefix_counts: np.ndarray,  # shape (L_max, C)
    starts:        np.ndarray,  # shape (R,), 1-based inclusive
    stops:         np.ndarray,  # shape (R,), 1-based inclusive
    bg_freq:       np.ndarray,  # shape (C,), >0, sums to 1
    smoothing:     float
) -> float:
    """
    Numba-compiled Jensen–Shannon divergence in bits over multiple discontiguous
    regions, using pseudocount smoothing exactly as in your original.
    """
    L_max, C = prefix_counts.shape
    R = starts.shape[0]

    # 1) accumulate raw counts & total length
    total_counts = np.zeros(C, np.float32)
    total_len = 0.0
    for i in range(R):
        s0 = starts[i] - 1
        e0 = stops[i]  - 1
        if s0 < 0 or e0 >= L_max or s0 > e0:
            continue
        # add prefix sums
        if s0 > 0:
            for k in range(C):
                total_counts[k] += prefix_counts[e0, k] - prefix_counts[s0-1, k]
        else:
            for k in range(C):
                total_counts[k] += prefix_counts[e0, k]
        total_len += (e0 - s0 + 1)

    # 2) add smoothing pseudocounts and adjust length
    for k in range(C):
        total_counts[k] += smoothing
    total_len += smoothing * C

    # 3) form observed distribution
    obs_freq = np.empty(C, np.float32)
    for k in range(C):
        obs_freq[k] = total_counts[k] / total_len

    # 4) compute JSD in bits
    jsd = 0.0
    for k in range(C):
        o = obs_freq[k]
        b = bg_freq[k]
        m = 0.5 * (o + b)
        # all >0 thanks to smoothing (bg_freq >0 by precondition)
        jsd += 0.5 * ( o * (math.log2(o) - math.log2(m))
                     + b * (math.log2(b) - math.log2(m)) )
    return jsd

@njit(fastmath=True)
def smooth_transitions(
    C_sub:            int,
    from_e:           np.ndarray,  # [E_sub]
    to_e:             np.ndarray,  # [E_sub]
    lp_e:             np.ndarray,  # [E_sub]
    none_sub:         int,
    starts_full:      np.ndarray,  # full-space start indices
    middles_full:     np.ndarray,  # full-space middle indices
    stops_full:       np.ndarray,  # full-space stop indices
    remap_arr:        np.ndarray,  # full→sub mapping (-1 if not kept)
    fill_ms:          float,
    fill_sm:          float,
    fill_ss:          float,
    F:                float,
    pfam_ids_full:    np.ndarray,  # full→pfamID
    role_ids_full:    np.ndarray,  # full→roleID
    from_all:         np.ndarray,  # [E_all] - original edges before filtering
    to_all:           np.ndarray   # [E_all]
):
    """
    Build dense sub-transition matrix, insert missing edges from full-space
    starts/middles, rescale & absorb, then convert back to CSR.
    """
    NEGINF = -1e300
    # build dense log-prob matrix
    mat = np.full((C_sub, C_sub), NEGINF)
    E_sub = from_e.shape[0]
    for e in range(E_sub):
        mat[from_e[e], to_e[e]] = lp_e[e]

    # 1) middle rows (use full starts_full)
    for full_i in middles_full:
        i_sub = remap_arr[full_i]
        if i_sub < 0:
            continue
        fam = pfam_ids_full[full_i]
        # count observed other-family starts in original data
        n_obs = 0
        for e in range(from_all.shape[0]):
            if from_all[e] == full_i and role_ids_full[to_all[e]] == 0 and pfam_ids_full[to_all[e]] != fam:
                n_obs += 1
        n_miss = int(F - 1) - n_obs
        denom = 1.0 + n_miss * fill_ms

        # (a) rescale observed edges
        for j in range(C_sub):
            if mat[i_sub, j] > NEGINF:
                mat[i_sub, j] = math.log(math.exp(mat[i_sub, j]) / denom)

        # (b) insert missing other-family starts
        insert_val = math.log(fill_ms / denom)
        for j_full in starts_full:
            if pfam_ids_full[j_full] == fam:
                continue
            j_sub = remap_arr[j_full]
            if j_sub >= 0 and mat[i_sub, j_sub] <= NEGINF:
                mat[i_sub, j_sub] = insert_val

        # (c) absorb residual into self-loop
        prob = 0.0
        for j in range(C_sub):
            if mat[i_sub, j] > NEGINF:
                prob += math.exp(mat[i_sub, j])
        old_self = math.exp(mat[i_sub, i_sub]) if mat[i_sub, i_sub] > NEGINF else 0.0
        mat[i_sub, i_sub] = math.log(old_self + (1.0 - prob))

    # 2) stop rows (use full middles_full + full starts_full)
    for full_i in stops_full:
        i_sub = remap_arr[full_i]
        if i_sub < 0:
            continue
        fam = pfam_ids_full[full_i]

        # count both observation types in original data
        n_obs_mid = 0
        n_obs_start = 0
        for e in range(from_all.shape[0]):
            if from_all[e] == full_i:
                j_full = to_all[e]
                if role_ids_full[j_full] == 1 and pfam_ids_full[j_full] != fam:
                    n_obs_mid += 1
                elif role_ids_full[j_full] == 0:
                    n_obs_start += 1

        # compute combined denom
        n_miss_mid   = int(F - 1) - n_obs_mid
        n_miss_start = int(F)     - n_obs_start
        denom = 1.0 + n_miss_mid * fill_sm + n_miss_start * fill_ss

        # (a) rescale all observed edges
        for j in range(C_sub):
            if mat[i_sub, j] > NEGINF:
                mat[i_sub, j] = math.log(math.exp(mat[i_sub, j]) / denom)

        # (b1) insert missing other-family middles
        insert_val_mid = math.log(fill_sm / denom)
        for j_full in middles_full:
            if pfam_ids_full[j_full] == fam:
                continue
            j_sub = remap_arr[j_full]
            if j_sub >= 0 and mat[i_sub, j_sub] <= NEGINF:
                mat[i_sub, j_sub] = insert_val_mid

        # (b2) insert missing any-family starts
        insert_val_start = math.log(fill_ss / denom)
        for j_full in starts_full:
            j_sub = remap_arr[j_full]
            if j_sub >= 0 and mat[i_sub, j_sub] <= NEGINF:
                mat[i_sub, j_sub] = insert_val_start

        # (c) absorb residual into none transition
        prob = 0.0
        for j in range(C_sub):
            if mat[i_sub, j] > NEGINF:
                prob += math.exp(mat[i_sub, j])
        old_none = math.exp(mat[i_sub, none_sub]) if mat[i_sub, none_sub] > NEGINF else 0.0
        mat[i_sub, none_sub] = math.log(old_none + (1.0 - prob))

    # 3) convert dense→CSR
    rowptr = np.zeros(C_sub + 1, dtype=np.int64)
    for i in range(C_sub):
        cnt = 0
        for j in range(C_sub):
            if mat[i, j] > NEGINF:
                cnt += 1
        rowptr[i+1] = rowptr[i] + cnt

    E_new = rowptr[C_sub]
    colidx = np.empty(E_new, dtype=np.int64)
    vals   = np.empty(E_new, dtype=np.float64)
    temp   = rowptr.copy()
    for i in range(C_sub):
        base = rowptr[i]
        for j in range(C_sub):
            v = mat[i, j]
            if v > NEGINF:
                idx = temp[i]
                colidx[idx] = j
                vals[idx]   = v
                temp[i] += 1

    return rowptr, colidx, vals

def subset_for_families(
    keep_mask: np.ndarray,
    inverse_label_mapping: Dict[int, Tuple[str, str]],
    role_map,
    from_all: np.ndarray,
    to_all:   np.ndarray,
    lp_all:   np.ndarray,
    none_sub: int,
    F: int,
    p_mid_to_start: float,
    p_stop_to_mid: float,
    p_stop_to_start: float,
    full_pfam_ids,
    full_role_ids
):
    """
    Vectorized & Numba-assisted subset_for_families.
    Returns:
      sub_emits (torch.Tensor [L, C_sub]),
      from_e, to_e, lp_e (np.ndarray),
      starts_sub, middles_sub, stops_sub (np.ndarray),
      sub_inv_lmap (dict), remap_arr (np.ndarray), none_sub (int), C_sub (int)
    """
    # 1) determine keep_mask & remap_arr
    C = keep_mask.shape[0]
    keep = np.nonzero(keep_mask)[0]
    C_sub = keep.shape[0]
    remap_arr = np.full(C, -1, dtype=np.int64)
    remap_arr[keep] = np.arange(C_sub, dtype=np.int64)

    # 3) filter & remap edges
    mask_e = keep_mask[from_all] & keep_mask[to_all]
    from_e = remap_arr[from_all[mask_e]]
    to_e   = remap_arr[to_all[mask_e]]
    lp_e   = lp_all[mask_e]

    starts_full = remap_arr[role_map['start']]    # map all full→sub, yields -1 for dropped
    starts_full = starts_full[starts_full >= 0]     # keep only valid sub-indices

    middles_full = remap_arr[role_map['middle']]
    middles_full = middles_full[middles_full >= 0]

    stops_full = remap_arr[role_map['stop']]
    stops_full = stops_full[stops_full >= 0]

    # 4) prepare sub-role arrays
    starts_sub  = np.array(remap_arr[ role_map['start'][ keep_mask[ role_map['start'] ] ] ])
    # middles_sub = np.array(remap_arr[ role_map['middle'][ keep_mask[ role_map['middle'] ] ] ])
    # stops_sub   = np.array(remap_arr[ role_map['stop'][   keep_mask[ role_map['stop']   ] ] ])

    # 5) smooth transitions via Numba CSR
    fill_ms = p_mid_to_start  / (F * (F-1))
    fill_sm = p_stop_to_mid   / (F * (F-1))
    fill_ss = p_stop_to_start / (F *  F)

    rowptr, colidx, vals = smooth_transitions(
        C_sub, from_e, to_e, lp_e, none_sub,
        starts_full, middles_full, stops_full, remap_arr,
        fill_ms, fill_sm, fill_ss, float(F),
        full_pfam_ids, full_role_ids,
        from_all, to_all
    )

    # 6) rebuild sub_inv_lmap
    sub_inv_lmap = { int(remap_arr[i]): inverse_label_mapping[i] for i in keep }

    return ( (rowptr, colidx, vals), sub_inv_lmap,
        remap_arr, C_sub,
        starts_sub, keep
    )

@njit(fastmath=True)
def _extract_domains_nb(
    best_np: np.ndarray,
    pfam_ids_sub: np.ndarray,
    role_ids_sub: np.ndarray
):
    L = best_np.shape[0]
    # preallocate worst-case
    pfams = np.empty(L, np.int64)
    starts = np.empty(L, np.int64)
    stops = np.empty(L, np.int64)
    statuses = np.empty(L, np.int8)
    count = 0
    if L == 0:
        return pfams, starts, stops, statuses, count
    prev_pfid = -1
    run_start = 0
    for i in range(L):
        sub = best_np[i]
        pfid = pfam_ids_sub[sub]
        role_i = role_ids_sub[sub]
        # split on family change (existing behavior) OR when we observe a new
        # start within the same family while already inside a run
        if pfid != prev_pfid or (prev_pfid >= 0 and pfid == prev_pfid and role_i == 0):
            # close prior run
            if prev_pfid >= 0:
                s = run_start
                e = i - 1
                start_role = role_ids_sub[best_np[s]]
                stop_role  = role_ids_sub[best_np[e]]
                # determine status_id
                if start_role == 0 and stop_role == 2:
                    stid = 0
                elif start_role != 0 and stop_role != 2:
                    stid = 3
                elif start_role != 0:
                    stid = 1
                else:
                    stid = 2
                pfams[count] = prev_pfid
                starts[count] = s + 1
                stops[count] = e + 1
                statuses[count] = stid
                count += 1
            prev_pfid = pfid
            run_start = i
    # last run
    if prev_pfid >= 0:
        s = run_start
        e = L - 1
        start_role = role_ids_sub[best_np[s]]
        stop_role  = role_ids_sub[best_np[e]]
        if start_role == 0 and stop_role == 2:
            stid = 0
        elif start_role != 0 and stop_role != 2:
            stid = 3
        elif start_role != 0:
            stid = 1
        else:
            stid = 2
        pfams[count] = prev_pfid
        starts[count] = s + 1
        stops[count] = e + 1
        statuses[count] = stid
        count += 1
    return pfams, starts, stops, statuses, count

def extract_domains(
    best_path: list,
    sub_states_full: np.ndarray,    # maps sub-index -> full-index
    state_pfam_ids_full: np.ndarray, # full-index -> pfam_id
    state_role_ids_full: np.ndarray, # full-index -> role_id (0,1,2 or -1)
    id_to_pfam: dict                # pfam_id -> pfam string
) -> list:
    """
    Wraps a Numba-accelerated run-length decode to extract domains.

    Uses precomputed full-state maps to avoid dict lookups.
    """
    # convert to numpy array
    best_np = np.array(best_path, dtype=np.int64)
    # build sub-space pfam_ids and role_ids
    pfam_ids_sub = state_pfam_ids_full[sub_states_full]
    role_ids_sub = state_role_ids_full[sub_states_full]

    # call JIT core
    pfams_run, starts, stops, statuses, count = _extract_domains_nb(
        best_np, pfam_ids_sub, role_ids_sub
    )
    # slice to valid runs
    pfams_run = pfams_run[:count]
    starts    = starts[:count]
    stops     = stops[:count]
    statuses  = statuses[:count]
    pfam_ids  = [id_to_pfam[pf] for pf in pfams_run]
    
    return pfams_run, starts, stops, statuses, pfam_ids

def complete_none_transitions_csr(
    rowptr: np.ndarray,           # shape (C_sub+1,)
    colidx: np.ndarray,           # shape (E_sub,)
    vals:   np.ndarray,           # shape (E_sub,)
    sequence_length: int,
    none_sub:          int,
    domain_start_labels: np.ndarray  # shape (M,)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Replace the CSR row for `none_sub` with:
      - a self-loop weight log(p_self)
      - uniform weights log(p_each) to each state in domain_start_labels

    Returns (new_rowptr, new_colidx, new_vals).
    """
    C_sub = rowptr.shape[0] - 1
    if sequence_length < 1 or domain_start_labels.size == 0:
        raise ValueError("Invalid none/transitions setup")

    # 1) compute the new row entries
    p_self = 1.0 - 1.0/sequence_length
    p_each = (1.0/sequence_length) / domain_start_labels.size

    new_cols = np.empty(domain_start_labels.size + 1, dtype=colidx.dtype)
    new_vals = np.empty(domain_start_labels.size + 1, dtype=vals.dtype)

    # self‐loop first
    new_cols[0] = none_sub
    new_vals[0] = math.log(p_self)
    # then one branch per domain start
    for k, s in enumerate(domain_start_labels, start=1):
        new_cols[k] = s
        new_vals[k] = math.log(p_each)

    # 2) figure out where the old none_sub row lives
    old_start = rowptr[none_sub]
    old_end   = rowptr[none_sub+1]
    old_len   = old_end - old_start
    new_len   = new_cols.size

    E_sub = colidx.shape[0]
    E_new = E_sub - old_len + new_len

    # 3) build the new rowptr
    new_rowptr = np.empty_like(rowptr)
    # rows < none_sub are unchanged
    new_rowptr[: none_sub+1] = rowptr[: none_sub+1]
    # shift all later rows by (new_len - old_len)
    shift = new_len - old_len
    new_rowptr[none_sub+1 :] = rowptr[none_sub+1 :] + shift

    # 4) build the new colidx & vals arrays
    new_colidx = np.empty(E_new, dtype=colidx.dtype)
    new_vals2  = np.empty(E_new, dtype=vals.dtype)

    # copy before none_sub row
    pre = old_start
    new_colidx[:pre] = colidx[:pre]
    new_vals2[:pre]  = vals[:pre]

    # insert new none_sub row
    new_colidx[pre : pre+new_len] = new_cols
    new_vals2[pre : pre+new_len]  = new_vals

    # copy after none_sub row
    post_src = old_end
    post_dst = pre + new_len
    new_colidx[post_dst:] = colidx[post_src:]
    new_vals2[post_dst:]  = vals[post_src:]

    return new_rowptr, new_colidx, new_vals2

@njit(fastmath=True)
def _argtopk(arr: np.ndarray, K: int) -> np.ndarray:
    """
    Return the indices of the top-K largest values in arr.
    This uses a full argsort; for large C you may replace with a more efficient selection algorithm.
    """
    # argsort descending
    idx = np.argsort(-arr)
    return idx[:K]

@njit(fastmath=True)
def nb_forward_beam_sumprod(
    emits: np.ndarray,         # shape (L, C)
    from_i: np.ndarray,        # shape (E,)
    to_i:   np.ndarray,        # shape (E,)
    logp:   np.ndarray,        # shape (E,)
    none_sub: int,
    starts: np.ndarray,        # shape (M,), start-state indices
    beam_size: int
) -> tuple:
    """
    Numba-accelerated beam-pruned sum-product forward (log-sum-exp) pass.
    Inputs are all numpy arrays on CPU. Returns (alpha, beam_list, beam_masks).
    alpha: np.ndarray[(L, C)]
    beam_list: np.ndarray[(L, K)] of top-state indices per time
    beam_masks: np.ndarray[(L, C)] of bool mask per time
    """
    L, C = emits.shape
    K = beam_size if beam_size < C else C

    # allocate outputs
    alpha = np.full((L, C), -1e300)
    beam_list = np.full((L, K), -1, np.int64)
    beam_masks = np.zeros((L, C), np.bool_)

    # working buffers
    dp = np.full(C, -1e300)
    M = np.full(C, -1e300)
    S = np.zeros(C)

    # init at t=0
    # this block was uniform over None + all starts (old version)
    # init_states = np.empty(starts.shape[0] + 1, np.int64)
    # init_states[0] = none_sub
    # for i in range(starts.shape[0]):
    #     init_states[i+1] = starts[i]
    # inv_count = init_states.shape[0]
    # init_logp = -np.log(inv_count)
    # for k in range(inv_count):
    #     dp[init_states[k]] = init_logp

    # init at t=0 using the “none→*” row from our CSR transitions
    for j in range(C):
        dp[j] = -1e300
    # scan all edges; any edge from none_sub seeds its to_i with logp
    for e in range(from_i.shape[0]):
        if from_i[e] == none_sub:
            dp[to_i[e]] = logp[e]

    # add emission at position 0
    for j in range(C):
        if dp[j] > -1e299:
            dp[j] += emits[0, j]
    alpha[0, :] = dp

    # initial beam and mask
    beam = _argtopk(dp, K)
    mask = np.zeros(C, np.bool_)
    for k in range(K):
        mask[beam[k]] = True
    mask[none_sub] = True
    # record
    for k in range(K):
        beam_list[0, k] = beam[k]
    beam_masks[0, :] = mask

    # iterate over t
    for t in range(1, L):
        # reset buffers
        for j in range(C):
            M[j] = -1e300
            S[j] = 0.0
        # phase 1: max
        for e in range(from_i.shape[0]):
            i_prev = from_i[e]
            j_next = to_i[e]
            if mask[i_prev]:
                val = dp[i_prev] + logp[e]
                if val > M[j_next]:
                    M[j_next] = val
        # phase 2: sum-exp
        for e in range(from_i.shape[0]):
            i_prev = from_i[e]
            j_next = to_i[e]
            if mask[i_prev] and M[j_next] > -1e299:
                S[j_next] += np.exp(dp[i_prev] + logp[e] - M[j_next])
        # update dp and alpha
        for j in range(C):
            if S[j] > 0.0:
                dp[j] = M[j] + np.log(S[j]) + emits[t, j]
            else:
                dp[j] = -1e300
            alpha[t, j] = dp[j]
        # beam prune
        beam = _argtopk(dp, K)
        for j in range(C):
            mask[j] = False
        for k in range(K):
            mask[beam[k]] = True
        mask[none_sub] = True
        # record
        for k in range(K):
            beam_list[t, k] = beam[k]
        beam_masks[t, :] = mask

    return alpha, beam_list, beam_masks

@njit(fastmath=True)
def nb_backward_beam_sumprod(
    emits: np.ndarray,         # shape (L, C)
    from_i: np.ndarray,        # shape (E,)
    to_i:   np.ndarray,        # shape (E,)
    logp:   np.ndarray,        # shape (E,)
    beam_list: np.ndarray,     # shape (L, K)
    beam_masks: np.ndarray     # shape (L, C), bool
) -> np.ndarray:
    """
    Numba-accelerated beam-pruned sum-product backward (log-sum-exp) pass.
    Returns beta of shape (L, C).
    """
    L, C = emits.shape
    E = from_i.shape[0]
    K = beam_list.shape[1]

    # allocate output and buffers
    beta = np.full((L, C), -1e300)
    M = np.full(C, -1e300)
    S = np.zeros(C)

    # initialize at t = L-1
    last = L - 1
    # set beta[last, j] = 0 for j in the final beam, rest stays -inf
    for k in range(K):
        idx = beam_list[last, k]
        beta[last, idx] = 0.0

    # backward recursion
    for t in range(L-2, -1, -1):
        # reset buffers
        for i in range(C):
            M[i] = -1e300
            S[i] = 0.0
        # phase 1: max over outgoing edges
        for e in range(E):
            i_prev = from_i[e]
            j_next = to_i[e]
            # only consider edges within the beam masks
            if beam_masks[t, i_prev] and beam_masks[t+1, j_next]:
                val = logp[e] + emits[t+1, j_next] + beta[t+1, j_next]
                if val > M[i_prev]:
                    M[i_prev] = val
        # phase 2: sum-exp normalize
        for e in range(E):
            i_prev = from_i[e]
            j_next = to_i[e]
            if beam_masks[t, i_prev] and beam_masks[t+1, j_next] and M[i_prev] > -1e299:
                S[i_prev] += np.exp(logp[e] + emits[t+1, j_next] + beta[t+1, j_next] - M[i_prev])
        # compute beta values and apply mask
        for i in range(C):
            if S[i] > 0.0:
                val = M[i] + np.log(S[i])
            else:
                val = -1e300
            # respect beam mask at time t
            if beam_masks[t, i]:
                beta[t, i] = val
            else:
                beta[t, i] = -1e300

    return beta

@njit(fastmath=True)
def nb_mea_decode(
    gamma: np.ndarray,        # shape (L, C)
    beam_list: np.ndarray,    # shape (L, K)
    adj: np.ndarray,          # shape (C, C), bool adjacency of sub_trans
    none_sub: int,
    starts: np.ndarray        # shape (M,), allowed start-state indices
) -> np.ndarray:
    """
    Numba-accelerated MEA dynamic programming and backtrack.
    Returns best_path (np.int64[L]).
    """
    L, C = gamma.shape
    _, K = beam_list.shape

    # allocate backpointer table
    backptr = np.full((L, C), -1, np.int64)
    # DP arrays
    M_prev = np.full(C, -1e300)
    M_curr = np.full(C, -1e300)

    # initialize at t=0 for allowed starts + none
    M_prev[none_sub] = gamma[0, none_sub]
    for i in range(starts.shape[0]):
        idx = starts[i]
        M_prev[idx] = gamma[0, idx]

    # forward DP
    for t in range(1, L):
        # reset current
        for j in range(C):
            M_curr[j] = -1e300
        # for each beam state j
        for b in range(K):
            j = beam_list[t, b]
            best_val = -1e300
            best_i = -1
            # search among previous beam states
            for a in range(K):
                i_prev = beam_list[t-1, a]
                # only if a transition exists
                if adj[i_prev, j]:
                    val = M_prev[i_prev] + gamma[t, j]
                    if val > best_val:
                        best_val = val
                        best_i = i_prev
            if best_i >= 0:
                M_curr[j] = best_val
                backptr[t, j] = best_i
        # swap M_prev <- M_curr
        for j in range(C):
            M_prev[j] = M_curr[j]

    # backtrack to get best path
    best_path = np.empty(L, np.int64)
    # find max at final time
    best_j = 0
    max_val = -1e300
    for j in range(C):
        if M_prev[j] > max_val:
            max_val = M_prev[j]
            best_j = j
    best_path[L-1] = best_j
    # step backwards
    for t in range(L-1, 0, -1):
        best_path[t-1] = backptr[t, best_path[t]]

    return best_path

@njit(fastmath=True)
def score_domains_nb(
    pfids:        np.ndarray,  # [D] domain PFAM IDs
    s_arr:        np.ndarray,  # [D] start positions (1-based)
    u_arr:        np.ndarray,  # [D] end   positions (1-based)
    gamma:        np.ndarray,  # [L, C_sub]
    prefix_count_b: np.ndarray,  # [L, 25] (this can actually be longer than L, but we only use up to the first L at most)
    none_sub:     int,
    remap_start:  np.ndarray,  # [P]
    remap_mid:    np.ndarray,  # [P]
    remap_stop:   np.ndarray,  # [P]
    exp_len_arr:  np.ndarray,  # [P]
    inv_ln2:      float,       # 1.0 / log(2)
    eps:          float
):
    D = pfids.shape[0]
    L, C = gamma.shape

    # outputs
    scores    = np.empty(D, dtype=np.float64)
    len_ratios = np.empty(D, dtype=np.float64)
    biases = np.empty(D, dtype=np.float64)
    
    for d in range(D):
        # compute bias
        biases[d] = compute_bias_slice_nb(prefix_count_b, 
                                    s_arr[d], 
                                    u_arr[d], 
                                    BG_COMPOSITION, 
                                    1e-9)
        
        pfid = pfids[d]
        s    = s_arr[d] - 1   # convert to 0-based
        u    = u_arr[d]       # exclusive upper bound in Python slicing
        start_idx = remap_start[d]
        mid_idx   = remap_mid[d]
        stop_idx  = remap_stop[d]
        
        # compute bitscore
        lod_sum = 0.0
        for t in range(s, u):
            # domain vs none posterior
            p_dom  = gamma[t, start_idx] + gamma[t, mid_idx] + gamma[t, stop_idx]
            # could have multiple states, but this matches your 3-state bitscore
            # p_none = gamma[t, none_sub]
            p_none = 1.0 - p_dom 
            # clamp
            if p_dom  < eps:  p_dom  = eps
            if p_none < eps:  p_none = eps
            # accumulate log-odds
            lod_sum += math.log(p_dom) - math.log(p_none)
        scores[d] = lod_sum * inv_ln2
        
        # length ratio
        obs_len = u - s
        exp_l   = exp_len_arr[d]
        len_ratios[d] = obs_len / exp_l


    return scores, len_ratios, biases

@njit
def _score_nested(
    region1: np.ndarray,   # shape (R1, C)
    region2: np.ndarray,   # shape (R2, C)
    none1: np.ndarray,     # shape (R1,)
    none2: np.ndarray,     # shape (R2,)
    states: np.ndarray,    # shape (S,), state indices
    none_sub: int,
    exp_len: float,
    eps: float = 1e-9
): # -> (float, float) is not valid in numba
    """
    Compute combined_score and combined_len_ratio for a nested domain pair.
    Returns (combined_score, combined_len_ratio).
    """
    R1, C = region1.shape
    R2, _ = region2.shape
    total = R1 + R2
    lod_sum = 0.0
    # log2 factor
    inv_ln2 = 1.0 / math.log(2.0)
    # accumulate log-odds
    for t in range(total):
        # if t < R1:
        #     # first region
        #     row = region1[t]
        #     p_none = none1[t]
        # else:
        #     idx = t - R1
        #     row = region2[idx]
        #     p_none = none2[idx]
        if t < R1: 
            row = region1[t]
        else:
            row = region2[t - R1]
        # sum domain posteriors
        p_dom = 0.0
        for k in range(states.shape[0]):
            p_dom += row[states[k]]
        # collapse “none” to all other mass
        p_none = 1.0 - p_dom 
        # sum domain posteriors
        p_dom = 0.0
        for k in range(states.shape[0]):
            p_dom += row[states[k]]
        # clamp and log-odds
        if p_dom < eps:
            p_dom = eps
        if p_none < eps:
            p_none = eps
        lod_sum += (math.log(p_dom) - math.log(p_none))
    # bitscore and length ratio
    combined_score = lod_sum * inv_ln2
    combined_obs_len = total
    combined_len_ratio = combined_obs_len / exp_len
    return combined_score, combined_len_ratio


def detect_and_combine_nested_domains(
    domains: list,
    gamma: np.ndarray,           # shape (L, C_sub)
    prefix_count_b: np.ndarray,  # [L, 25] (this can actually be longer than L, but we only use up to the first L at most)
    sub_inv_lmap: dict,
    none_sub: int,
    expected_len_fn
) -> list:
    """
    NumPy + Numba-assisted nested domain detection, now including status-3 runs
    of the same family in between a left-partial and right-partial pair.
    """
    if len(domains) < 2:
        return domains

    # 1) Sort domains by start coordinate
    sorted_domains = sorted(domains, key=lambda x: x[1])
    used = set()
    combined = []

    # 2) Build map: family -> list of sub-state indices
    pfam_states = {}
    for idx, (pf, _) in sub_inv_lmap.items():
        pfam_states.setdefault(pf, []).append(idx)

    # 3) Cache expected lengths
    exp_cache = {}

    # 4) Scan for merges
    for i, (pf1, s1, e1, _, _, _, st1) in enumerate(sorted_domains):
        # Only consider left-partials not yet used
        if i in used or st1 != "partial (no stop)":
            continue

        # Try to find a matching right-partial
        for j in range(i+1, len(sorted_domains)):
            pf2, s2, e2, _, _, _, st2 = sorted_domains[j]
            if j in used or pf2 != pf1 or st2 != "partial (no start)":
                continue

            # Walk through runs between i and j, collecting same-family status-3
            curr_end = e1
            to_include = []  # indices of status-3 runs to include
            ok = True
            for k in range(i+1, j):
                _, sk, ek, _, _, _, stk = sorted_domains[k]
                # Check coverage gap
                if sk > curr_end + 1:
                    ok = False
                    break
                curr_end = ek
                # If this run is status-3 of pf1, include it
                if stk == "partial (no start or stop)" and sorted_domains[k][0] == pf1:
                    to_include.append(k)
            if not ok:
                continue

            # Ensure the final run j also abuts
            if s2 > curr_end + 1:
                continue

            # We have a merge candidate: i, optional to_include, and j
            # Prepare regions for scoring
            all_starts = [s1] + [sorted_domains[k][1] for k in to_include] + [s2]
            all_ends   = [e1] + [sorted_domains[k][2] for k in to_include] + [e2]

            # Expected length for this family
            exp_len = exp_cache.get(pf1)
            if exp_len is None:
                exp_len = expected_len_fn(pf1)
                exp_cache[pf1] = exp_len

            # Combined score: sum component scores from selected segments
            comp_idxs = [i] + to_include + [j]
            score = float(sum(sorted_domains[idx][3] for idx in comp_idxs))
            # Combined length ratio: total length over expected length
            total_len = sum((sorted_domains[idx][2] - sorted_domains[idx][1] + 1) for idx in comp_idxs)
            len_ratio = total_len / exp_len

            # compute combined bias. input starts and stops need to be 1-based and np arrays of all starts and stops respectively
            combined_bias = compute_bias_combined_nb(prefix_count_b, 
                                            np.array(all_starts), 
                                            np.array(all_ends), 
                                            BG_COMPOSITION, 
                                            1e-9)

            # Record merged as a full domain
            combined.append((pf1, s1, e2, score, len_ratio, combined_bias, "full (merged)"))
            used.update({i, j, *to_include})
            break

        # --- Fallback L→R: synthesize right end from last same-family mid ---
        if i not in used:
            curr_end_lr = e1
            mid_idxs_lr = []
            abort_lr = False
            for k in range(i+1, len(sorted_domains)):
                if k in used:
                    continue
                pfk, sk, ek, _, _, _, stk = sorted_domains[k]
                # gap breaks cluster
                if sk > curr_end_lr + 1:
                    break
                # encountering other-family left anchor ends cluster
                if stk == "partial (no stop)" and pfk != pf1:
                    break
                # collect same-family mids
                if pfk == pf1 and stk == "partial (no start or stop)":
                    mid_idxs_lr.append(k)
                # if we see a same-family right anchor here, original path would have handled
                elif pfk == pf1 and stk == "partial (no start)":
                    abort_lr = True
                    break
                # advance
                curr_end_lr = ek

            if (not abort_lr) and len(mid_idxs_lr) > 0:
                comp_idxs = [i] + mid_idxs_lr
                # Expected length
                exp_len = exp_cache.get(pf1)
                if exp_len is None:
                    exp_len = expected_len_fn(pf1)
                    exp_cache[pf1] = exp_len
                # score = sum of component scores
                score = float(sum(sorted_domains[idx][3] for idx in comp_idxs))
                # len ratio over expected length
                total_len = sum((sorted_domains[idx][2] - sorted_domains[idx][1] + 1) for idx in comp_idxs)
                len_ratio = total_len / exp_len
                # bias across all selected windows
                all_starts_fb = [sorted_domains[idx][1] for idx in comp_idxs]
                all_ends_fb   = [sorted_domains[idx][2] for idx in comp_idxs]
                combined_bias = compute_bias_combined_nb(
                    prefix_count_b,
                    np.array(all_starts_fb),
                    np.array(all_ends_fb),
                    BG_COMPOSITION,
                    1e-9
                )
                # right boundary is the end of the last mid
                e_synth = all_ends_fb[-1]
                combined.append((pf1, s1, e_synth, score, len_ratio, combined_bias, "full (merged)"))
                used.update(comp_idxs)

    # --- Fallback R→L: mirror – synthesize left end from first same-family mid ---
    for j in range(len(sorted_domains)-1, -1, -1):
        pfR, sR, eR, _, _, _, stR = sorted_domains[j]
        if j in used or stR != "partial (no start)":
            continue
        curr_start_rl = sR
        mid_idxs_rl = []
        abort_rl = False
        for k in range(j-1, -1, -1):
            if k in used:
                continue
            pfk, sk, ek, _, _, _, stk = sorted_domains[k]
            # gap breaks cluster
            if ek < curr_start_rl - 1:
                break
            # encountering other-family right anchor ends cluster (mirror)
            if stk == "partial (no start)" and pfk != pfR:
                break
            if pfk == pfR and stk == "partial (no start or stop)":
                mid_idxs_rl.append(k)
                curr_start_rl = sk
                continue
            elif pfk == pfR and stk == "partial (no stop)":
                # original path would handle left+right; abort fallback
                abort_rl = True
                break
            curr_start_rl = sk
        if abort_rl or len(mid_idxs_rl) == 0:
            continue
        # use farthest-left mid (last appended in R→L scan)
        comp_idxs = list(reversed(mid_idxs_rl)) + [j]
        # expected length
        exp_len = exp_cache.get(pfR)
        if exp_len is None:
            exp_len = expected_len_fn(pfR)
            exp_cache[pfR] = exp_len
        # combined score & len ratio
        score = float(sum(sorted_domains[idx][3] for idx in comp_idxs))
        total_len = sum((sorted_domains[idx][2] - sorted_domains[idx][1] + 1) for idx in comp_idxs)
        len_ratio = total_len / exp_len
        # bias across windows
        all_starts_fb = [sorted_domains[idx][1] for idx in comp_idxs]
        all_ends_fb   = [sorted_domains[idx][2] for idx in comp_idxs]
        combined_bias = compute_bias_combined_nb(
            prefix_count_b,
            np.array(all_starts_fb),
            np.array(all_ends_fb),
            BG_COMPOSITION,
            1e-9
        )
        s_synth = all_starts_fb[0]
        combined.append((pfR, s_synth, eR, score, len_ratio, combined_bias, "full (merged)"))
        used.update(comp_idxs)

    # 5) Append all untouched runs
    for idx, d in enumerate(sorted_domains):
        if idx not in used:
            combined.append(d)

    # 6) Sort by start and return
    combined.sort(key=lambda x: x[1])
    return combined

@njit(fastmath=True)
def build_4state_hmm(
    from_list:    np.ndarray,
    to_list:      np.ndarray,
    lp_list:      np.ndarray,
    none_full:    int,
    start_full:   int,
    mid_full:     int,
    stop_full:    int,
    seq_length:   int,
    F:            int,
    p_stop_to_start: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 4-state HMM for {None, Start, Middle, Stop} in CSR form.
    Returns (from_i, to_i, logp).
    """
    # allocate the tiny dense matrix
    lin = np.zeros((4, 4), dtype=np.float32)
    # 1) None-row: dwell model
    p_nn = 1.0 - 1.0/seq_length
    p_ns = 1.0/seq_length
    lin[0, 0] = p_nn
    lin[0, 1] = p_ns

    # 2) Start→Middle (copy empirical)
    for idx in range(from_list.shape[0]):
        if from_list[idx] == start_full and to_list[idx] == mid_full:
            lin[1, 2] = math.exp(lp_list[idx])
            break

    # 3) Middle-row: mid→mid & mid→stop, absorb residual into mid→mid
    p_mm = 0.0
    p_ms = 0.0
    for idx in range(from_list.shape[0]):
        src = from_list[idx]; dst = to_list[idx]
        if src == mid_full and dst == mid_full:
            p_mm = math.exp(lp_list[idx])
        elif src == mid_full and dst == stop_full:
            p_ms = math.exp(lp_list[idx])
    # absorb gap
    residual_m = 1.0 - (p_mm + p_ms)
    if residual_m < 0.0:
        residual_m = 0.0
    lin[2, 2] = p_mm + residual_m
    lin[2, 3] = p_ms

    # 4) Stop-row: stop→None + pseudocount stop→Start, renormalize, absorb rounding
    p_sn = 0.0
    for idx in range(from_list.shape[0]):
        if from_list[idx] == stop_full and to_list[idx] == none_full:
            p_sn = math.exp(lp_list[idx])
            break
    fill_ss = p_stop_to_start / (F * F)
    denom_s = p_sn + fill_ss
    if denom_s <= 0.0:
        denom_s = 1e-300
    p_sn_new = p_sn / denom_s
    p_ss_new = fill_ss / denom_s
    residual_s = 1.0 - (p_sn_new + p_ss_new)
    if residual_s < 0.0:
        residual_s = 0.0
    p_sn_new += residual_s
    lin[3, 0] = p_sn_new
    lin[3, 1] = p_ss_new

    # build CSR arrays
    # first count nonzeros
    cnt = 0
    for i in range(4):
        for j in range(4):
            if lin[i, j] > 0.0:
                cnt += 1

    from_i = np.empty(cnt, np.int64)
    to_i   = np.empty(cnt, np.int64)
    logp   = np.empty(cnt, np.float32)
    pos = 0
    for i in range(4):
        for j in range(4):
            val = lin[i, j]
            if val > 0.0:
                from_i[pos] = i
                to_i[pos]   = j
                logp[pos]   = math.log(val)
                pos += 1

    return from_i, to_i, logp

# ------------------------------------------------------------------------------
# Partition-function LLD (PF-LLD) helpers
#   - Uses the existing 4-state family CRF for a given segment [s..e]
#   - Computes logZ_model and logZ_null (domain channels zeroed), returns bits
# ------------------------------------------------------------------------------
@njit(fastmath=True)
def pfld_bits_4state_fast(
    p4: np.ndarray,           # shape (L, 4) probabilities over [None, Start, Mid, Stop]
    from4: np.ndarray,        # 1D transitions of the 4-state HMM
    to4:   np.ndarray,        # 1D
    logp4: np.ndarray         # 1D (log transition probs)
) -> float:
    """
    Optimized exact forward for the fixed 4-state topology with closed-form null.
    Returns PF-LLD in natural logs, divided by ln 2 at the call site.
    """
    L = p4.shape[0]
    if L == 0:
        return 0.0

    NEG = -1e300

    # Extract the used transition logs
    lp_nn = NEG; lp_ns = NEG; lp_sm = NEG; lp_mm = NEG; lp_ms = NEG; lp_sn = NEG; lp_ss = NEG
    for k in range(from4.shape[0]):
        i = from4[k]
        j = to4[k]
        v = float(logp4[k])
        if i == 0 and j == 0:
            lp_nn = v
        elif i == 0 and j == 1:
            lp_ns = v
        elif i == 1 and j == 2:
            lp_sm = v
        elif i == 2 and j == 2:
            lp_mm = v
        elif i == 2 and j == 3:
            lp_ms = v
        elif i == 3 and j == 0:
            lp_sn = v
        elif i == 3 and j == 1:
            lp_ss = v

    # helper: renormalize and return log emissions for a single row
    def renorm_logs(a0, a1, a2, a3):
        s = a0 + a1 + a2 + a3
        if s < 1e-40:
            s = 1e-40
        inv_s = 1.0 / s
        b0 = a0 * inv_s
        b1 = a1 * inv_s
        b2 = a2 * inv_s
        b3 = a3 * inv_s
        if b0 < 1e-40:
            b0 = 1e-40
        if b1 < 1e-40:
            b1 = 1e-40
        if b2 < 1e-40:
            b2 = 1e-40
        if b3 < 1e-40:
            b3 = 1e-40
        return math.log(b0), math.log(b1), math.log(b2), math.log(b3)

    # t=0: seed from None row
    le0, le1, le2, le3 = renorm_logs(p4[0, 0], p4[0, 1], p4[0, 2], p4[0, 3])
    aN = lp_nn + le0
    aS = lp_ns + le1
    aM = NEG
    aT = NEG

    # closed-form null accumulator
    sum_log_e_none = le0

    # t >= 1
    for t in range(1, L):
        le0, le1, le2, le3 = renorm_logs(p4[t, 0], p4[t, 1], p4[t, 2], p4[t, 3])

        # None <- {None, Stop}
        v1 = aN + lp_nn
        v2 = aT + lp_sn
        if v1 > v2:
            m = v1
            s2 = 1.0 + (math.exp(v2 - v1) if v2 > NEG else 0.0)
        else:
            m = v2
            s2 = 1.0 + (math.exp(v1 - v2) if v1 > NEG else 0.0)
        newN = m + math.log(s2) + le0

        # Start <- {None, Stop}
        v1 = aN + lp_ns
        v2 = aT + lp_ss
        if v1 <= NEG and v2 <= NEG:
            newS = NEG
        else:
            if v1 > v2:
                m = v1
                s2 = 1.0 + (math.exp(v2 - v1) if v2 > NEG else 0.0)
            else:
                m = v2
                s2 = 1.0 + (math.exp(v1 - v2) if v1 > NEG else 0.0)
            newS = m + math.log(s2) + le1

        # Mid <- {Start, Mid}
        v1 = aS + lp_sm
        v2 = aM + lp_mm
        if v1 <= NEG and v2 <= NEG:
            newM = NEG
        else:
            if v1 > v2:
                m = v1
                s2 = 1.0 + (math.exp(v2 - v1) if v2 > NEG else 0.0)
            else:
                m = v2
                s2 = 1.0 + (math.exp(v1 - v2) if v1 > NEG else 0.0)
            newM = m + math.log(s2) + le2

        # Stop <- {Mid}
        v1 = aM + lp_ms
        newT = v1 + le3 if v1 > NEG else NEG

        aN = newN
        aS = newS
        aM = newM
        aT = newT
        sum_log_e_none += le0

    # logZ_model = logsumexp(aN, aS, aM, aT)
    m = aN
    if aS > m:
        m = aS
    if aM > m:
        m = aM
    if aT > m:
        m = aT
    tot = 0.0
    if aN > NEG:
        tot += math.exp(aN - m)
    if aS > NEG:
        tot += math.exp(aS - m)
    if aM > NEG:
        tot += math.exp(aM - m)
    if aT > NEG:
        tot += math.exp(aT - m)
    logZ_model = m + math.log(tot)

    # Closed-form null (all-None path only)
    logZ_null = sum_log_e_none + L * lp_nn

    return (logZ_model - logZ_null)
def _pfld_segment_bits(
    p_emit_ref_seg: np.ndarray,   # shape (Lseg, C_sub) probabilities per sub-state
    idx_none: int,
    idx_start: int,
    idx_mid: int,
    idx_stop: int,
    from4: np.ndarray,
    to4: np.ndarray,
    logp4: np.ndarray,
    insert_mask: np.ndarray | None = None,  # shape (Lseg,), True where to force None (zero domain channels)
) -> float:
    """
    Compute PF-LLD bitscore on a segment using an optimized 4-state family CRF.

    Scoring-only emission handling:
      - Renormalize probabilities over the 4 selected states
        {None, Start, Mid, Stop} at each position (NO residual to None).
      - Null model is computed in closed form: only the all-None path contributes.
    """
    Lseg = p_emit_ref_seg.shape[0]
    if Lseg <= 0 or idx_none < 0 or idx_start < 0 or idx_mid < 0 or idx_stop < 0:
        return 0.0
    # Build 4-state emissions for the segment: [None, Start, Mid, Stop]
    p4 = np.empty((Lseg, 4), dtype=np.float32)
    p4[:, 0] = p_emit_ref_seg[:, idx_none]
    p4[:, 1] = p_emit_ref_seg[:, idx_start]
    p4[:, 2] = p_emit_ref_seg[:, idx_mid]
    p4[:, 3] = p_emit_ref_seg[:, idx_stop]

    # Apply optional insert mask: force domain channels to zero before renorm
    if insert_mask is not None and insert_mask.shape[0] == Lseg:
        p4[insert_mask, 1:] = 0.0

    # Call optimized kernel and convert to bits
    diff_log = pfld_bits_4state_fast(p4, from4, to4, logp4)
    return float(diff_log / math.log(2.0))

def _score_domains_pfld(
    domains: List[Tuple[str,int,int,float,float,float,str]],
    label_mapping: Dict[str, Dict[str,int]],
    remap: np.ndarray,
    sub_states_full: np.ndarray,
    none_sub: int,
    p_emit_ref: np.ndarray,              # shape (L, C_sub)
    from_list: np.ndarray,
    to_list: np.ndarray,
    lp_list: np.ndarray,
    prior_stop_to_start: float,
    merged_spans: List[Tuple[str,int,int]] | None = None,
    insert_runs: List[Tuple[str,int,int,str]] | None = None,
) -> List[Tuple[str,int,int,float,float,float,str]]:
    """
    Recompute bitscore for each (pfam, s, e, score, len_ratio, bias, status)
    using PF-LLD on the segment [s..e] with a 4-state CRF.
    Strictly internal inserts (si > s and ei < e, different family) are handled as full domains.
    Merged spans are handled as full domains and their segments are concatenated for scoring.
    Note that slices are 1-based inclusive in domain tuples (to match typical protein database notation)

    """
    if not domains:
        return domains

    full_none = sub_states_full[none_sub]
    idx_none = remap[full_none]

    # small caches (shared within this call)
    trans_cache: Dict[Tuple[str,int], Tuple[np.ndarray,np.ndarray,np.ndarray]] = {}
    idx_cache: Dict[str, Tuple[int,int,int]] = {}
    score_cache: Dict[Tuple[str,int,Tuple[Tuple[int,int],...]], float] = {}

    L_total = p_emit_ref.shape[0]

    # Precompute insert interval lists for merged spans if provided
    inserts_by_merged: Dict[Tuple[str,int,int], List[Tuple[int,int]]] = {}
    if merged_spans is not None and insert_runs is not None and len(merged_spans) > 0 and len(insert_runs) > 0:
        merged_set = {(pfm, so, eo) for (pfm, so, eo) in merged_spans}
        # For each merged owner span, collect strictly-internal insert intervals of other families
        for pfm, so, eo in merged_set:
            intervals: List[Tuple[int,int]] = []
            for (pfi, si, ei, _st) in insert_runs:
                if pfi == pfm:
                    continue
                # strictly inside the merged span
                if si > so and ei < eo:
                    # clamp defensively (though they should already be inside)
                    ls = max(si, so)
                    le = min(ei, eo)
                    if le >= ls:
                        intervals.append((ls, le))
            # sort by start
            if intervals:
                intervals.sort(key=lambda x: x[0])
            inserts_by_merged[(pfm, so, eo)] = intervals

    out_list: List[Tuple[str,int,int,float,float,float,str]] = []
    for pf, s, e, _, lr, bias, st in domains:
        # bounds check
        if s < 1: s = 1
        if e > L_total: e = L_total
        if e < s:
            out_list.append((pf, int(s), int(e), 0.0, float(lr), float(bias), st))
            continue

        # sub-indices for family states (cached)
        idxs = idx_cache.get(pf)
        if idxs is None:
            s_full = label_mapping[pf]["start"]
            m_full = label_mapping[pf]["middle"]
            o_full = label_mapping[pf]["stop"]
            idx_start = remap[s_full]
            idx_mid   = remap[m_full]
            idx_stop  = remap[o_full]
            idx_cache[pf] = (idx_start, idx_mid, idx_stop)
            idxs = (idx_start, idx_mid, idx_stop)

        idx_start, idx_mid, idx_stop = idxs
        # if any mapping invalid, produce zero score
        if idx_none < 0 or idx_start < 0 or idx_mid < 0 or idx_stop < 0:
            out_list.append((pf, int(s), int(e), 0.0, float(lr), float(bias), st))
            continue

        # Determine contiguous owner-only subsegments (concatenate for scoring)
        # Default single segment
        subsegments: List[Tuple[int,int]] = [(int(s), int(e))]
        key_span = (pf, int(s), int(e))
        if key_span in inserts_by_merged:
            ins_intervals = inserts_by_merged[key_span]
            if ins_intervals:
                subsegments = []
                cursor = int(s)
                for (si, ei) in ins_intervals:
                    if cursor <= si - 1:
                        subsegments.append((cursor, si - 1))
                    cursor = ei + 1
                if cursor <= int(e):
                    subsegments.append((cursor, int(e)))
                # if inserts cover whole span, subsegments may be empty

        # Build concatenated emissions by stacking each owner-only subsegment
        if not subsegments:
            out_list.append((pf, int(s), int(e), 0.0, float(lr), float(bias), st))
            continue
        parts = [p_emit_ref[a-1:b, :] for (a, b) in subsegments]
        p_emit_ref_seg = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        Lseg = p_emit_ref_seg.shape[0]
        key = (pf, Lseg)
        trans = trans_cache.get(key)
        if trans is None:
            # build 4-state transitions for this family & segment length
            s_full = label_mapping[pf]["start"]
            m_full = label_mapping[pf]["middle"]
            o_full = label_mapping[pf]["stop"]
            from4, to4, logp4 = build_4state_hmm(
                from_list, to_list, lp_list,
                full_none, s_full, m_full, o_full,
                seq_length=Lseg, F=24076, p_stop_to_start=prior_stop_to_start
            )
            trans = (from4, to4, logp4)
            trans_cache[key] = trans

        from4, to4, logp4 = trans

        # cache key includes the precise subsegment layout
        score_key = (pf, Lseg, tuple(subsegments))
        cached = score_cache.get(score_key)
        if cached is None:
            bits = _pfld_segment_bits(
                p_emit_ref_seg,
                idx_none, idx_start, idx_mid, idx_stop,
                from4, to4, logp4,
            )
            score_cache[score_key] = bits
        else:
            bits = cached

        out_list.append((pf, int(s), int(e), float(bits), float(lr), float(bias), st))

    return out_list

def annotate_domains(
    sub_emits: torch.Tensor,                       # [L, C] these need to be tensors
    keep_mask: torch.Tensor,
    sequence: str,
    inverse_label_mapping: Dict[int, Tuple[str,str]],
    label_mapping: Dict[str, Dict[str,int]],
    from_list,
    to_list,
    lp_list,
    role_map,
    full_pfam_ids,
    full_role_ids,
    id_to_pfam,
    none_sub: int,
    N: int = 1,
    beam_size: int = 128,
    prior_mid_to_start : float = 3.4e-5,
    prior_stop_to_mid  : float = 4.817e-3,
    prior_stop_to_start: float = 5.540e-3,
    refine_extended: bool = True
) -> Tuple[
    List[Tuple[str,int,int,float,float,float,str]],  # (pfam, start, stop, score, len_ratio, bias, status) - original
    List[Tuple[str,int,int,float,float,float,str]],  # (pfam, start, stop, score, len_ratio, bias, status) - with nested combined
    List[int],                          # best_path
    np.ndarray,                      # sub_emits np array[L, C_sub]
    int,                               # num_families_after_prefilter
    Dict[int,Tuple[str,str]]           # sub_inv_lmap
]:
    """
    Annotates domains in a sequence using emission log probs and HMM transitions.
    Performs prefiltering, subsetting, forward/backward (beamed or dense), MEA decoding, domain extraction, and scoring.
    Uses dense path for small C_sub (<= beam_size) for efficiency.
    LLR-based scoring is used for domains and nested combined domains.

    Args:
        emission_log_probs: [L, C] log probs.
        transition_matrix: Sparse log-prob transitions.
        inverse_label_mapping: Index to (pfam, role).
        none_label: "None" index.
        N: Top-N for prefilter.
        beam_size: For beamed path.
        prior_mid_to_start: Smoothing prior.
        prior_stop_to_mid: Smoothing prior.
        prior_stop_to_start: Smoothing prior.
        refine_extended: Run conservative refinement for domains whose len_ratio rounds > 1.

    Returns:
        Tuple of domains list, best_path, sub_emits, num_prefilter, sub_inv_lmap.
    """
    # --- Step 0: prep ---
    L, C_sub = sub_emits.shape
    sub_emits = sub_emits.numpy().astype(np.float32, copy=False)
    keep_mask = keep_mask.contiguous().numpy()
    prefix_count_b = compute_prefix_counts([sequence])[0]
    assert prefix_count_b.shape[0] == L, "prefix_count_b shape mismatch" # to check if sequence length is same as sub_emits length

    # --- Step 1: subset & global forward/backward ---
    (rowptr, colidx, vals), sub_inv_lmap, remap, C_sub, starts_sub, sub_states_full = subset_for_families(
            keep_mask,
            inverse_label_mapping,
            role_map,
            from_list,
            to_list,
            lp_list,
            none_sub,
            F                = 24076,
            p_mid_to_start   = prior_mid_to_start,
            p_stop_to_mid    = prior_stop_to_mid,
            p_stop_to_start  = prior_stop_to_start,
            full_pfam_ids    = full_pfam_ids,
            full_role_ids    = full_role_ids
    )

    rowptr2, to_idxs, trans_logp = complete_none_transitions_csr(
        rowptr, colidx, vals,
        sequence_length=L,
        none_sub=none_sub,
        domain_start_labels=starts_sub
    )

    # 4) forward & backward

    # Build edge arrays

    lengths = rowptr2[1:] - rowptr2[:-1]             # shape (C_sub,), #edges per row
    from_idxs  = np.repeat(np.arange(C_sub, dtype=np.int64), lengths)

    # 3) fill in a single pass

    assert from_idxs.shape == to_idxs.shape == trans_logp.shape

    alpha, beam_list, beam_masks = nb_forward_beam_sumprod(
        sub_emits,
        from_idxs,
        to_idxs,
        trans_logp,
        none_sub,
        starts_sub,
        beam_size
    )
    beta = nb_backward_beam_sumprod(
        sub_emits,
        from_idxs,
        to_idxs,
        trans_logp,
        beam_list,
        beam_masks
    )
    # Compute logZ/gamma (only needed for beamed path; dense already has them)
    logZ = np.logaddexp.reduce(alpha[-1])
    gamma = np.exp(alpha + beta - logZ)

    # initialise with allowed start states (+ None)
    init_states = np.insert(starts_sub, 0, none_sub)

    # Create adjacency matrix for compatibility with numba
    adj = np.zeros((C_sub, C_sub), dtype=bool)

    # vectorized fill
    adj[from_idxs, to_idxs] = True
    
    # Best path extraction
    best_path = nb_mea_decode(gamma, beam_list, adj, none_sub, init_states)
    
    # --- Step 2: extract & initial scoring ---
    pfams_run, starts, stops, status_ids, pfam_accs = extract_domains(
      best_path,
      sub_states_full,
      full_pfam_ids,
      full_role_ids,
      id_to_pfam
    )
    
    # map numeric runs back to strings
    status_map = {
        0: "full",
        1: "partial (no start)",
        2: "partial (no stop)",
        3: "partial (no start or stop)"
    }

    statuses = [status_map[idx] for idx in status_ids]
    
    # score domains
    # get ratio of obs length to expected length
    _exp_len_cache: Dict[str, float] = {}

    def expected_len(pfam: str) -> float:
        """E[length] under the 2-state (middle,stop) sub-chain."""
        if pfam in _exp_len_cache:
            return _exp_len_cache[pfam]

        # locate the full-set indices for this family's middle & stop states
        mid_idx = label_mapping[pfam]["middle"]
        stop_idx = label_mapping[pfam]["stop"]

        mask_mm = (from_list == mid_idx) & (to_list == mid_idx)
        p_mm_raw = np.exp(lp_list[mask_mm]).sum()

        # pick out all “middle→stop” edges
        mask_ms = (from_list == mid_idx) & (to_list == stop_idx)
        p_ms_raw = np.exp(lp_list[mask_ms]).sum()

        # renormalise over the inside-family edges (MM + MS = 1)
        denom = p_mm_raw + p_ms_raw
        p_mm  = p_mm_raw / denom
        p_ms  = p_ms_raw / denom

        exp_mid = p_mm / p_ms            # geometric mean #middles
        exp_len = exp_mid + 2            # + start + stop
        _exp_len_cache[pfam] = exp_len
        return exp_len
    
    # precompute expected lengths per family for len_ratio
    exp_len_arr = np.empty(len(pfam_accs), dtype=np.float32)
    for i,pf in enumerate(pfam_accs):
        exp_len_arr[i] = expected_len(pf)

    # compute len_ratio and composition bias; defer PF-LLD scoring to post-refinement
    lenrat_np = np.empty(len(pfam_accs), dtype=np.float32)
    biases_np = np.empty(len(pfam_accs), dtype=np.float64)
    for i, (s, u, pf) in enumerate(zip(starts, stops, pfam_accs)):
        obs_len = (int(u) - int(s) + 1)
        exp_l = float(exp_len_arr[i]) if exp_len_arr[i] > 0 else 1.0
        lenrat_np[i] = obs_len / exp_l
        biases_np[i] = compute_bias_slice_nb(prefix_count_b, int(s), int(u), BG_COMPOSITION, 1e-9)

    out = [
        (
            pf,
            int(s),
            int(u),
            0.0,                       # placeholder; replaced post-refinement with PF-LLD
            float(length_ratio),
            float(bias),
            st
        )
        for pf, s, u, length_ratio, bias, st
        in zip(pfam_accs, starts, stops, lenrat_np, biases_np, statuses)
    ]
    
    # --- Step 3: nested combine ---
    combined_domains = detect_and_combine_nested_domains(out, gamma, prefix_count_b, sub_inv_lmap, none_sub, expected_len)
    # --- Step 4: gating for refinement based on len_ratio rounding (> 1 only) ---
    # Only run refinement when at least one domain has a len_ratio that rounds > 1.
    def _rounded_stats(dom_list):
        if not dom_list:
            return False, False, False
        lrs = np.array([np.rint(d[4]) for d in dom_list])
        any_ne = np.any(lrs != 1)
        any_gt = np.any(lrs > 1)
        any_lt = np.any(lrs < 1)
        return bool(any_ne), bool(any_gt), bool(any_lt)

    _, any_gt_c, _ = _rounded_stats(combined_domains)
    if not refine_extended or not any_gt_c:
        # For testing: score only final domains; leave original out with placeholder scores
        p_emit_ref_local = np.exp(np.clip(sub_emits, a_min=-690.0, a_max=None))
        final_scored = _score_domains_pfld(combined_domains, label_mapping, remap, sub_states_full, none_sub, p_emit_ref_local, from_list, to_list, lp_list, prior_stop_to_start)
        return out, final_scored, best_path.tolist(), sub_emits, sub_states_full.shape[0], sub_inv_lmap

    # ----------------------------------------------------------------------
    # --- Step 5 : derive refinement candidates from len_ratio only ---------
    # Vector-friendly view of every domain row (Pfam, s, e, score, len, status)
    cd_np = np.array(
        combined_domains,
        dtype=[
            ('pf',    'U16'),
            ('s',     np.int64),
            ('e',     np.int64),
            ('sc',    np.float32),
            ('lr',    np.float32),
            ('bias',  np.float32),      # ← new field
            ('st',    'U24'),
        ]
    )
    is_merge   = (cd_np['st'] == 'full (merged)')
    merge_s    = cd_np['s'][is_merge]
    merge_e    = cd_np['e'][is_merge]
    merge_pf   = cd_np['pf'][is_merge]

    # For every row, whether it sits wholly inside *any* merged span
    # index of first containing span (undefined when outside → 0 but ignored)
    # Guard against the common case "no merged rows"
    if merge_s.size == 0:
        row_inside = np.zeros(cd_np.shape[0], dtype=bool)
        row_owner  = np.array([""] * cd_np.shape[0])      # keeps dtype <U1
    else:
        inside_mat = ((cd_np['s'][:, None] >= merge_s) &
                      (cd_np['e'][:, None] <= merge_e))
        row_inside = inside_mat.any(1)
        owner_idx  = inside_mat.argmax(1)                 # safe: merge_s.size>0
        row_owner  = merge_pf[owner_idx]

    # genuine rows = not synthetic merge row, not swallowed by merge
    genuine = ~is_merge & ~row_inside

    # ----------------------------------------------------------------------

    # 5.0  helpers -------------------------------------------------------------
    pfam2clan: Dict[str, Any] = {
        pf: (info.get("clan_id") if isinstance(info, dict) else None)
        for pf, info in label_mapping.items()
    }

    # Determine extended-refinement rows and family set (rint(len_ratio) > 1)
    rounded_lr = np.rint(cd_np['lr'])
    row_refine_long = (rounded_lr > 1)
    refine_fams_long: set[str] = set(cd_np['pf'][row_refine_long])
    
    # --- Step 6: mask out to preserve ---
    # helper ────────────────────────────────────────────────────────────────────
    def _collect_insert_runs(merged_spans, first_pass_runs):
        """
        merged_spans      : List[(start, stop)]  from merged_full domains
        first_pass_runs   : List of 7-tuples like (pf, s, e, sc, lr, bias, st)

        Returns every run from the first pass that lies completely inside *any*
        merged span.  (Inclusive coordinates, 1-based.)
        """
        inserts = []
        for pf, s, e, sc, lr, bias, st in first_pass_runs:
            for ms, me in merged_spans:
                if ms < s and e < me:
                    idx = np.flatnonzero(merge_s == ms)
                    if idx.size == 0:          # should never happen, but be safe
                        continue
                    owner = merge_pf[idx[0]]  # span’s owner

                    if pf != owner:
                        inserts.append((pf, s, e, sc, lr, bias, st))
                        break                       # no need to test other spans
        return inserts

    # ----------------------------------------------------------------------
    mask_preserve       = np.zeros(L, dtype=bool)
    preserved_runs_list = []

    # 6.1  gather spans of every merged_full domain
    merged_info  = [(pf, s, e)                     # (family, start, stop)
                    for pf, s, e, _, _, _, st in combined_domains
                    if st == "full (merged)"]

    merged_spans = [(s, e) for _, s, e in merged_info]   # keep old list too

    # 6.2  main loop over combined_domains
    for row_idx, (pf, s, e, sc, lr, bias, st) in enumerate(combined_domains):

        if st == "full (merged)":
            # freeze the merged result itself
            mask_preserve[s-1:e] = True
            preserved_runs_list.append((pf, s, e, sc, lr, bias, "full"))
            continue

        # ───────────── inside a merged span? ─────────────
        if row_inside[row_idx]:

            merge_owner = row_owner[row_idx]   # owner of this merged span

            if pf == merge_owner:
                #  internal run of *the same* family (e.g. A-middle) ((this is rare but can happen)) → DROP
                #   It neither gets masked nor contributes to refine_set.
                continue

            # internal run of a DIFFERENT family (B-insert) → keep
            mask_preserve[s-1:e] = True
            preserved_runs_list.append((pf, s, e, sc, lr, bias, st))
            continue                     # done with this run

        # ───────────── outside any merge ─────────────
        if row_refine_long[row_idx]:
            # family will be re-decoded; leave rows unmasked
            continue

        # otherwise preserve & mask
        mask_preserve[s-1:e] = True
        preserved_runs_list.append((pf, s, e, sc, lr, bias, st))

    # 6.3  add “insert-runs” from the first pass that were swallowed by a merge
    insert_runs = _collect_insert_runs(merged_spans, out)
    for tup in insert_runs:
        pf, s, e, sc, lr, bias, st = tup
        # run touches *either* edge of the merged span that contains it
        touch_edge = ((s == merge_s) | (e == merge_e)).any()
        if touch_edge:
            continue
        if (pf, s, e) not in {(x[0], x[1], x[2]) for x in preserved_runs_list}:
            mask_preserve[s-1:e] = True
            preserved_runs_list.append(tup)
    # build masked emissions
    # reminder that sub_emits are log probs over filtered sub space states
    p_emit_ref = np.exp(np.clip(sub_emits, a_min=-690.0, a_max=None))

    # --- Step 7: second-pass per-family MEA for extended refinement (residual → None) ---
    refined_results_long: List[Tuple[str,int,int,float,float,float,str]] = []
    for pfam in sorted(refine_fams_long):
        s_full,m_full,o_full = (label_mapping[pfam][k] for k in ('start','middle','stop'))
        if any(remap[x]<0 for x in (none_sub,s_full,m_full,o_full)):
            continue
        # build HMM
        # — build 4-state family-specific HMM (full-space indices) —
        full_none = sub_states_full[none_sub]
        from4, to4, logp4 = build_4state_hmm(
            from_list, to_list, lp_list,
            full_none, s_full, m_full, o_full,
            seq_length=L, F=24076, p_stop_to_start=prior_stop_to_start
        )

        # — build per-family emissions & log-space —
        p4 = np.stack([
            p_emit_ref[:, remap[full_none]],
            p_emit_ref[:, remap[s_full]],
            p_emit_ref[:, remap[m_full]],
            p_emit_ref[:, remap[o_full]],
        ], axis=1)
        if mask_preserve is not None:
            p4[mask_preserve, 1:] = 0.0   # keep “None” unchanged
        sum4 = p4.sum(axis=1, keepdims=True)
        p4[:, 0] += (1.0 - sum4).ravel()
        gamma4 = np.log(np.clip(p4, 1e-40, None))

        # — forward/backward & MEA backtrack on the 4-state HMM —
        start4 = np.array([1], dtype=np.int64)  # “Start” state index
        a2, b2, m2 = nb_forward_beam_sumprod(
            gamma4, from4, to4, logp4,
            0, start4, 4
        )
        bb2 = nb_backward_beam_sumprod(gamma4, from4, to4, logp4, b2, m2)
        logZ2 = np.logaddexp.reduce(a2[-1])
        g2 = np.exp(a2 + bb2 - logZ2)
        adj4 = np.zeros((4,4), dtype=bool); adj4[from4, to4] = True
        best4 = nb_mea_decode(g2, b2, adj4, 0, start4)

        # — extract & remap back to full PFAMs —
        pfids_run, starts_run, stops_run, status_ids_run, pfam_accs_run = extract_domains(
            best4.tolist(),
            np.array([full_none, s_full, m_full, o_full], dtype=np.int64),
            full_pfam_ids, full_role_ids, id_to_pfam
        )
        sts = [ status_map[i] for i in status_ids_run ]

        # — compute placeholders for PF-LLD (scored later): len_ratio and bias only —
        exp_len_pf = expected_len(pfam)
        for pf0, s0, e0, st0 in zip(pfam_accs_run, starts_run, stops_run, sts):
            obs_len0 = int(e0) - int(s0) + 1
            lr0 = (obs_len0 / exp_len_pf) if exp_len_pf > 0 else 0.0
            bias0 = compute_bias_slice_nb(prefix_count_b, int(s0), int(e0), BG_COMPOSITION, 1e-9)
            refined_results_long.append((pf0, int(s0), int(e0), 0.0, float(lr0), float(bias0), st0))

    # --- Step 8: assemble final ---
    final = preserved_runs_list.copy()
    final.extend(refined_results_long)
    
    # Families that were refined in extended pass
    refined_union = set([pf for pf,_,_,_,_,_,_ in refined_results_long])

    for tup in combined_domains:
        pf, s, e, sc, lr, bias, st = tup

        # already preserved
        if tup in preserved_runs_list:
            continue

        # run lives strictly inside a merged-full span  ->  drop it
        if ((s >= merge_s) & (e <= merge_e)).any():
            continue

        # family scheduled for refinement in extended pass -> drop it
        if pf in refined_union and refine_extended:
            continue

        final.append(tup)

    final.sort(key=lambda x: x[1])

    # PF-LLD scoring for both original and final domain lists (post-refinement)
    # Build merged span/inserts context for efficient, correct scoring
    merged_spans_ctx = [(pf, s, e) for (pf, s, e, sc, lr, bias, st) in final if st == "full (merged)"]
    # inserts are non-owner runs strictly inside merged spans; reuse logic from earlier collection
    # We treat every non-owner run in "final" as a candidate insert; strictly-inside filtering is done in scorer
    insert_runs_ctx = [(pf, s, e, st) for (pf, s, e, sc, lr, bias, st) in final if st != "full (merged)"]

    final_scored = _score_domains_pfld(final, label_mapping, remap, sub_states_full, none_sub, p_emit_ref, from_list, to_list, lp_list, prior_stop_to_start, merged_spans_ctx, insert_runs_ctx)

    return out, final_scored, best_path.tolist(), sub_emits, sub_states_full.shape[0], sub_inv_lmap
