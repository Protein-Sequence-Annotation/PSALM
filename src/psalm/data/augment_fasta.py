"""
augment_fasta.py

Split protein FASTA records into max-length slices that preserve complete PFAM
domains, and optionally emit non-domain-shuffled variants and fully shuffled
negatives. Input domains are a pickle dict {seq_id: [(pfam_id, start, stop), ...]}
with 1-indexed inclusive coordinates. Sequence IDs in the FASTA are normalized
to the middle field of pipe-delimited headers (a|b|c -> b) before lookup.

Rules:
  1. No domain may be partially cut; each slice must contain full domains.
  2. Coordinates are 1-indexed and inclusive.

Slicing strategy:
  - For one domain: try a slice from one end that includes the domain; if not
    possible, use a max-length slice centered on the domain.
  - For multiple domains:
      * If the span (earliest start to latest stop) is <= max_length, treat as
        one group and take one slice.
      * If the span > max_length, break the domains into the minimum number of
        groups (greedy grouping) such that each group's span is within
        max_length. (Slices may overlap if necessary.)
  - Slices are chosen to be as close to max_length as possible using choose_slice.

Output:
  - A new FASTA file is written. New record names have the format:
         <old_name>_<B/M/E>    if there is one slice, or
         <old_name>_<slice_id>_<B/M/E>  if multiple slices are produced.
         "B" indicates a slice that includes the beginning (slice_start == 1),
         "E" indicates that the slice reaches the sequence's end,
         and "M" means neither.
  - A new domain dictionary is generated. Its keys are the new record names, and
    the values are lists of updated domain tuples (coordinates are recalculated
    relative to the slice).
  - Output FASTA records and output dict keys are guaranteed to match.

Variants and negatives:
  - For each emitted base record (original, shuffled, or domain slice), a fully
    shuffled "negative_" sequence may be created.
  - --negative-prob specifies a target fraction of negatives in the final
    dataset (approximate). The per-record emission probability is derived from
    this target.
  - Special case: sequences with a single full-length domain emit a guaranteed
    full negative plus up to two embedded negatives (with optional full-negative
    variants for each embedding).

Large-data mode (optional):
  - Enable with --large-data to control dataset composition at scale.
  - For each base unit (short seq or max-length slice), emit exactly one of
    {original, non-domain-shuffled}, chosen by --p-shuffled.
  - Independently sample fully shuffled negatives to target --negative-prob of
    the final dataset.
  - Optionally sample individual domain slices to target --domain-slice-frac
    using per-class probabilities derived from --domain-counts-tsv (present
    PFAMs only). Domain-slice IDs omit _B/_M/_E in this mode.
  - Fractions are expected/approximate, not guaranteed.

ID normalization:
  - If an input FASTA header looks like "x|y|z", only the middle field ("y") is
    used as the ID, and is used consistently in outputs and the new domain dict.

Usage:
  python scripts/data/augment_fasta.py --fasta input.fa --domain-dict domains.pkl \
      --output-fasta output.fa --output-dict new_domains.pkl [...]
"""

import sys
import pickle
import random
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

_NEG_SUFFIXES = ("", "_B", "_E", "_M")
_EXTRA_FIELD_WARNED = False


def _rand_neg_suffix() -> str:
    """Return a random suffix for synthetic negatives: '', '_B', '_E', or '_M'."""
    return random.choice(_NEG_SUFFIXES)


def _read_pfam_counts_tsv(path):
    """
    Read a TSV with header `family\tcount` and return a dict {pfam_id: count(int)}.
    Lines with malformed content are ignored.
    """
    counts = {}
    try:
        with open(path, "r") as f:
            header_skipped = False
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if not header_skipped:
                    header_skipped = True
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                pfam_id, cnt_s = parts[0], parts[1]
                try:
                    cnt = int(cnt_s)
                except ValueError:
                    continue
                counts[pfam_id] = counts.get(pfam_id, 0) + cnt
    except FileNotFoundError:
        print(
            f"WARNING: domain-counts TSV not found: {path}. "
            "Domain-slice sampling will be disabled."
        )
        return {}
    return counts


def _prune_domain_dict_to_fasta(domain_dict, fasta_id_set):
    """
    Simplify domain_dict keys and keep only entries whose simplified key is in
    fasta_id_set. Merge entries that collapse to the same simplified key and
    filter domains to length > 5 aa. Returns a new dict.
    """
    global _EXTRA_FIELD_WARNED
    pruned = {}
    try:
        items = domain_dict.items()
    except Exception:
        return domain_dict
    for key, doms in items:
        simple_key = simplify_id(key)
        if simple_key not in fasta_id_set:
            continue
        filtered = []
        for entry in doms:
            if entry is None:
                continue
            if len(entry) < 3:
                continue
            if len(entry) > 3 and not _EXTRA_FIELD_WARNED:
                print(
                    "WARNING: domain tuple has extra fields; trimming to first "
                    "three (pfam,start,stop). Further warnings suppressed."
                )
                _EXTRA_FIELD_WARNED = True
            pfam, s, e = entry[:3]
            try:
                if (e - s + 1) > 5:
                    filtered.append((pfam, s, e))
            except Exception:
                continue
        if not filtered:
            continue
        if simple_key in pruned:
            pruned[simple_key].extend(filtered)
        else:
            pruned[simple_key] = list(filtered)
    return pruned


def simplify_id(seq_id):
    """
    If the sequence ID is of the format "something|something|something",
    return only the middle field (i.e. the part between the two pipes).
    Otherwise, return the original seq_id.
    """
    parts = seq_id.split("|")
    if len(parts) >= 3:
        return parts[1]
    return seq_id


def choose_slice(group_min, group_max, L, max_length):
    """
    Computes a slice [slice_start, slice_end] of length max_length that contains
    the entire domain group, defined by group_min and group_max (1-indexed,
    inclusive). The slice is centered as much as possible on the group, and
    boundaries are adjusted so that:
          1 <= slice_start and slice_end <= L.

    Returns (slice_start, slice_end, label) where:
      - slice_end - slice_start + 1 == max_length.
      - label is "B" if slice_start == 1, "E" if slice_end == L, "M" otherwise.
    """
    group_span = group_max - group_min + 1
    available_extra = max_length - group_span
    offset = available_extra // 2
    candidate_start = group_min - offset
    if candidate_start < 1:
        candidate_start = 1
    if candidate_start > L - max_length + 1:
        candidate_start = L - max_length + 1
    slice_start = candidate_start
    slice_end = candidate_start + max_length - 1
    if slice_start == 1:
        label = "B"
    elif slice_end == L:
        label = "E"
    else:
        label = "M"
    return slice_start, slice_end, label


def process_sequence(seq_record, domains, max_length):
    """
    Given a SeqRecord and its list of domain tuples [(pfam, start, stop), ...],
    splits the sequence into slices if its length > max_length.

    Domains are expected to be pre-sorted by start.
    Coordinates are 1-indexed and inclusive.

    Slicing strategy:
      1. Single domain: if the domain fits in a slice starting at 1 or ending at
         L, take that slice; otherwise, center on the domain.
      2. Multiple domains:
         - If overall span (min start to max stop) <= max_length, use one slice.
         - Else, group domains greedily into groups where each group's span
           <= max_length (overlap allowed).

    Returns a list of slices. Each slice is a tuple:
         (new_slice_name, slice_start, slice_end, slice_seq, updated_domains)
    The new_slice_name is built as:
         <old_name>_<B/M/E>   if one slice, or
         <old_name>_<slice_id>_<B/M/E>   for multiple slices.
    """
    L = len(seq_record.seq)
    sorted_domains = domains  # assumed pre-sorted by start

    # Check that no domain's stop coordinate exceeds the sequence length.
    max_domain_stop = max(d[2] for d in sorted_domains)
    if max_domain_stop > L:
        raise ValueError(
            f"Sequence {seq_record.id} has a domain with stop coordinate "
            f"{max_domain_stop} which exceeds the sequence length {L}."
        )

    slices_info = []

    def update_domains(domain_list, s_start):
        updated = []
        for pfam, dstart, dstop in domain_list:
            updated.append((pfam, dstart - s_start + 1, dstop - s_start + 1))
        return updated

    if len(sorted_domains) == 1:
        pfam, dstart, dstop = sorted_domains[0]
        if dstop <= max_length:
            s_start, s_end, label = 1, max_length, "B"
        elif dstart >= L - max_length + 1:
            s_start, s_end, label = L - max_length + 1, L, "E"
        else:
            s_start, s_end, label = choose_slice(dstart, dstop, L, max_length)
        if not (s_start <= dstart and dstop <= s_end):
            raise ValueError(
                f"Domain {(pfam, dstart, dstop)} not fully in slice "
                f"{s_start}-{s_end}."
            )
        slices_info.append(
            (None, s_start, s_end, seq_record.seq[s_start - 1:s_end],
             [(pfam, dstart, dstop)], label)
        )
    else:
        overall_min = min(d[1] for d in sorted_domains)
        overall_max = max(d[2] for d in sorted_domains)
        if overall_max - overall_min + 1 <= max_length:
            s_start, s_end, label = choose_slice(overall_min, overall_max, L, max_length)
            for pfam, dstart, dstop in sorted_domains:
                if not (s_start <= dstart and dstop <= s_end):
                    raise ValueError(
                        f"Domain {(pfam, dstart, dstop)} not in slice "
                        f"{s_start}-{s_end}."
                    )
            slices_info.append(
                (None, s_start, s_end, seq_record.seq[s_start - 1:s_end],
                 sorted_domains, label)
            )
        else:
            groups = []
            current_group = [sorted_domains[0]]
            for d in sorted_domains[1:]:
                g_min = min(g[1] for g in current_group)
                g_max = max(g[2] for g in current_group)
                new_min = min(g_min, d[1])
                new_max = max(g_max, d[2])
                if new_max - new_min + 1 <= max_length:
                    current_group.append(d)
                else:
                    groups.append(current_group)
                    current_group = [d]
            groups.append(current_group)
            slice_index = 1
            for i, group in enumerate(groups):
                g_min = min(d[1] for d in group)
                g_max = max(d[2] for d in group)
                s_start, s_end, _ = choose_slice(g_min, g_max, L, max_length)
                if i == 0 and s_start == 1:
                    label = "B"
                elif i == len(groups) - 1 and s_end == L:
                    label = "E"
                else:
                    label = "M"
                slices_info.append(
                    (slice_index, s_start, s_end, seq_record.seq[s_start - 1:s_end],
                     group, label)
                )
                slice_index += 1

    new_slices = []
    if len(slices_info) == 1:
        _, s_start, s_end, s_seq, doms, label = slices_info[0]
        new_name = f"{seq_record.id}_{label}"
        new_slices.append((new_name, s_start, s_end, s_seq, update_domains(doms, s_start)))
    else:
        for slice_id, s_start, s_end, s_seq, doms, label in slices_info:
            new_name = f"{seq_record.id}_{slice_id}_{label}"
            new_slices.append((new_name, s_start, s_end, s_seq, update_domains(doms, s_start)))
    return new_slices


def shuffle_non_domain(seq_str, updated_domains):
    """
    Given a sequence (as a string) and its list of domain tuples (with
    coordinates relative to the sequence, 1-indexed), returns a new sequence
    string where any region not covered by a domain is randomly shuffled.
    Each non-domain segment is shuffled independently, while domain regions
    remain intact. If no domains are provided, the entire sequence is shuffled.
    """
    if not updated_domains:
        return shuffle_entire(seq_str)
    seq_list = list(seq_str)
    domains = sorted(updated_domains, key=lambda x: x[1])
    non_domain_regions = []
    current = 0
    for _, dstart, dstop in domains:
        if dstart - 1 > current:
            non_domain_regions.append((current, dstart - 1))
        current = dstop
    if current < len(seq_list):
        non_domain_regions.append((current, len(seq_list)))
    for start_idx, end_idx in non_domain_regions:
        region = seq_list[start_idx:end_idx]
        random.shuffle(region)
        seq_list[start_idx:end_idx] = region
    return "".join(seq_list)


def shuffle_entire(seq_str):
    """Returns a fully shuffled version of the input sequence string."""
    seq_list = list(seq_str)
    random.shuffle(seq_list)
    return "".join(seq_list)


def _neg_emit_probability(target_frac: float) -> float:
    """
    Convert a target negative fraction into a per-record emission probability.
    For base count B, emitting with probability p gives expected fraction
    p / (1 + p). Solving for p yields p = target / (1 - target).
    """
    if target_frac < 0:
        raise ValueError("negative-prob must be >= 0.")
    if target_frac == 0:
        return 0.0
    if target_frac >= 1:
        raise ValueError("negative-prob must be < 1.")
    p_emit = target_frac / (1.0 - target_frac)
    if p_emit > 1.0:
        print(
            "WARNING: negative-prob >= 0.5; capping per-record emission "
            "at 1.0 to keep probability valid."
        )
        return 1.0
    return p_emit


def _make_record(seq_str, rec_id):
    return SeqRecord(Seq(seq_str), id=rec_id, name=rec_id, description="")


def _emit_record(output_records, new_domain_dict, record, domains):
    rec_id = record.id
    if rec_id in new_domain_dict:
        raise ValueError(f"Duplicate output record id: {rec_id}")
    record.name = rec_id
    output_records.append(record)
    new_domain_dict[rec_id] = domains


def _emit_negative(output_records, new_domain_dict, neg_id, seq_str):
    neg_record = _make_record(seq_str, neg_id)
    _emit_record(output_records, new_domain_dict, neg_record, [("None", 1, len(seq_str))])


def _maybe_emit_negative(output_records, new_domain_dict, base_id, base_seq_str, p_emit, vprint):
    if p_emit <= 0:
        return
    if random.random() >= p_emit:
        return
    negative_seq = shuffle_entire(base_seq_str)
    negative_name = f"negative_{base_id}{_rand_neg_suffix()}"
    _emit_negative(output_records, new_domain_dict, negative_name, negative_seq)
    vprint(f"DEBUG: Created negative version: {negative_name}")


def _maybe_emit_domain_slice_negative(
    output_records,
    new_domain_dict,
    rec_id,
    dstart,
    dstop,
    slice_label,
    dom_seq_str,
    p_emit,
    vprint,
):
    if p_emit <= 0:
        return
    if random.random() >= p_emit:
        return
    neg_name = f"negative_{rec_id}_{dstart}_{dstop}{slice_label}"
    neg_seq = shuffle_entire(dom_seq_str)
    _emit_negative(output_records, new_domain_dict, neg_name, neg_seq)
    vprint(f"DEBUG: Created negative for domain slice: {neg_name}")


def _emit_full_length_domain_special_case(
    rec,
    doms,
    max_length,
    fasta_records,
    output_records,
    new_domain_dict,
    p_neg_emit,
    vprint,
    shuffle_only,
):
    """Emit special-case records for single full-length domain sequences."""
    seq_str = str(rec.seq)
    pfam, dstart, dstop = doms[0]
    L = len(rec.seq)

    if not shuffle_only:
        _emit_record(output_records, new_domain_dict, rec, doms)

    # Guaranteed full-sequence negative
    neg_seq = shuffle_entire(seq_str)
    neg_name = f"negative_{rec.id}{_rand_neg_suffix()}"
    _emit_negative(output_records, new_domain_dict, neg_name, neg_seq)
    vprint(f"DEBUG: Created full-negative version: {neg_name}")

    # Two embedded negatives
    bg_pool = [r for r in fasta_records if L <= len(r.seq) <= max_length and r.id != rec.id]
    if not bg_pool:
        return
    sampled = random.sample(bg_pool, min(2, len(bg_pool)))
    for idx, bg in enumerate(sampled, start=1):
        bg_shuf = shuffle_entire(str(bg.seq))
        max_pos = len(bg_shuf) - L
        insert_pos = 0 if max_pos == 0 else random.randint(0, max_pos)
        embedded_seq = (
            bg_shuf[:insert_pos]
            + seq_str
            + bg_shuf[insert_pos + L:]
        )
        emb_name = f"shuffled_{rec.id}_embed{idx}"
        new_rec_emb = _make_record(embedded_seq, emb_name)
        _emit_record(
            output_records,
            new_domain_dict,
            new_rec_emb,
            [(pfam, insert_pos + 1, insert_pos + L)],
        )
        vprint(f"DEBUG: Created embedded-negative version: {emb_name}")

        if random.random() < p_neg_emit:
            emb_neg = shuffle_entire(embedded_seq)
            emb_neg_name = f"negative_{emb_name}{_rand_neg_suffix()}"
            _emit_negative(output_records, new_domain_dict, emb_neg_name, emb_neg)
            vprint(f"DEBUG: Created embedded full-negative: {emb_neg_name}")


def _ensure_unique_ids(records, label):
    seen = set()
    dupes = set()
    for rec in records:
        if rec.id in seen:
            dupes.add(rec.id)
        else:
            seen.add(rec.id)
    if dupes:
        sample = ", ".join(sorted(list(dupes))[:5])
        suffix = "" if len(dupes) <= 5 else " ..."
        raise ValueError(f"{label} has duplicate IDs after normalization: {sample}{suffix}")


def _validate_output_alignment(output_records, new_domain_dict):
    output_ids = [r.id for r in output_records]
    if len(output_ids) != len(set(output_ids)):
        raise ValueError("Output FASTA contains duplicate IDs.")
    output_set = set(output_ids)
    dict_set = set(new_domain_dict.keys())
    if output_set != dict_set:
        missing = output_set - dict_set
        extra = dict_set - output_set
        details = []
        if missing:
            details.append(f"missing dict entries: {sorted(list(missing))[:5]}")
        if extra:
            details.append(f"extra dict entries: {sorted(list(extra))[:5]}")
        raise ValueError("Output FASTA and domain dict keys differ: " + "; ".join(details))


def _write_outputs(output_fasta, output_dict, output_records, new_domain_dict):
    _validate_output_alignment(output_records, new_domain_dict)
    with open(output_fasta, "w") as out_f:
        SeqIO.write(output_records, out_f, "fasta")
    with open(output_dict, "wb") as out_d:
        pickle.dump(new_domain_dict, out_d)


def _print_length_distributions(output_records):
    bins = [
        ("<=25", lambda L: L <= 25),
        ("26-50", lambda L: 25 < L <= 50),
        ("51-100", lambda L: 50 < L <= 100),
        ("101-400", lambda L: 100 < L <= 400),
        ("401-1600", lambda L: 400 < L <= 1600),
        (">1600", lambda L: L > 1600),
    ]
    nonneg = [len(r.seq) for r in output_records if not r.id.startswith("negative_")]
    neg = [len(r.seq) for r in output_records if r.id.startswith("negative_")]

    def print_dist(name, lengths):
        total = len(lengths)
        print(f"\n{name}  (n={total}):")
        if total == 0:
            print("  (none)")
            return
        for label, test in bins:
            cnt = sum(1 for L in lengths if test(L))
            pct = cnt / total * 100
            print(f"  {label:8s} : {cnt:5d} sequences  ({pct:5.1f}%)")

    print_dist("Non-negatives", nonneg)
    print_dist("Negatives    ", neg)


def _print_domain_slice_stats(total_domain_slices, mid_domain_slices):
    if total_domain_slices:
        frac_mid = mid_domain_slices / total_domain_slices * 100.0
        print(
            "\nDomain slice stats: "
            f"{mid_domain_slices}/{total_domain_slices} slices are pure middles "
            f"(_M)  ({frac_mid:.1f}%)."
        )
    else:
        print("\nDomain slice stats: no domain slices were generated.")


def _run_large_data_mode(args, fasta_records, domain_dict, max_length, vprint):
    """
    Large-data mode:
      - For each base unit (short seq or max-length slice), emit exactly one of
        {original, non-domain-shuffled} with probability controlled by
        --p-shuffled.
      - Independently emit a full negative from base units with probability
        chosen to target --negative-prob of the final dataset.
      - Optionally emit domain slices by sampling each domain instance with
        per-class probability p_c = alpha*(1-f_c), where f_c are normalized
        frequencies from --domain-counts-tsv restricted to PFAMs present in
        this FASTA. Domain slice IDs omit _B/_M/_E in this mode.
    Returns (output_records, new_domain_dict).
    """
    # Basic debug: show first few FASTA IDs and first few pickle keys
    try:
        first5_fa = [r.id for r in fasta_records[:5]]
    except Exception:
        first5_fa = []
    try:
        first5_keys = list(domain_dict.keys())[:5]
    except Exception:
        first5_keys = []
    vprint(f"LARGEDATA DEBUG: FASTA total={len(fasta_records)} first5={first5_fa}")
    vprint(f"LARGEDATA DEBUG: domain_dict total={len(domain_dict)} first5={first5_keys}")

    # Pre-pass: filter domains, compute base count B and collect domain instances
    B = 0
    domain_instances = []  # list of (rec.id, pfam, dstart, dstop)
    present_pfams = set()

    for rec in tqdm(fasta_records, desc="Prepass"):
        if rec.id not in domain_dict:
            # skip records without domains
            continue
        # filter domains > 5 aa (1-indexed inclusive)
        doms = [
            (pfam, s, e)
            for pfam, s, e in domain_dict[rec.id]
            if (e - s + 1) > 5
        ]
        if not doms:
            continue
        doms = sorted(doms, key=lambda x: x[1])
        domain_dict[rec.id] = doms
        L = len(rec.seq)
        if L > max_length:
            try:
                slices = process_sequence(rec, doms, max_length)
            except Exception as e:
                print(f"Error during prepass slicing {rec.id}: {e}", file=sys.stderr)
                continue
            B += len(slices)
        else:
            B += 1
        for pfam, s, e in doms:
            if pfam == "None":
                continue
            domain_instances.append((rec.id, pfam, s, e))
            present_pfams.add(pfam)

    # Compute sampling targets
    r_d = getattr(args, "domain_slice_frac", 0.05)
    r_n = getattr(args, "negative_prob", 0.05)
    if not getattr(args, "domain_counts_tsv", None) and r_d > 0:
        print(
            "WARNING: --domain-slice-frac set but --domain-counts-tsv not "
            "provided; domain-slice sampling disabled."
        )
        r_d = 0.0
    if r_d < 0 or r_n < 0 or r_d + r_n >= 1:
        print(
            "WARNING: target fractions invalid or sum >= 1. Disabling "
            "domain-slice and negative quotas for safety."
        )
        r_d = 0.0
        r_n = 0.0

    S_expected = B / (1 - (r_d + r_n)) if (r_d + r_n) < 1 else float(B)
    M_domain_target = r_d * S_expected
    K_neg_target = r_n * S_expected

    # Domain-slice probabilities p_c
    p_c_map = None
    N = len(domain_instances)
    if getattr(args, "domain_counts_tsv", None):
        counts_map = _read_pfam_counts_tsv(args.domain_counts_tsv)
        # Restrict to PFAMs present in this FASTA; treat missing as rare (f=0)
        if present_pfams:
            present_counts = {pf: counts_map.get(pf, 0) for pf in present_pfams}
            total_present = sum(present_counts.values())
            f_map = {}
            if total_present > 0:
                for pf in present_pfams:
                    cnt = present_counts.get(pf, 0)
                    f_map[pf] = cnt / total_present if total_present > 0 else 0.0
            else:
                # Fallback: uniform over present PFAMs
                uniform = 1.0 / len(present_pfams)
                for pf in present_pfams:
                    f_map[pf] = uniform
            G = 1.0 - sum((f_map[pf] ** 2 for pf in present_pfams))
            if N == 0 or G <= 0:
                alpha = 0.0
            else:
                alpha = M_domain_target / (N * G)
                alpha = max(0.0, min(1.0, alpha))
            p_c_map = {
                pf: max(0.0, min(1.0, alpha * (1.0 - f_map.get(pf, 0.0))))
                for pf in present_pfams
            }
        else:
            p_c_map = {}
    else:
        p_c_map = None  # domain-slice sampling disabled

    # Base negative probability
    p_neg_base = 0.0
    if B > 0:
        p_neg_base = max(0.0, min(1.0, K_neg_target / B))

    # Emission pass
    output_records = []
    new_domain_dict = {}

    p_shuf = getattr(args, "p_shuffled", 0.5)

    for rec in tqdm(fasta_records, desc="Emission pass"):
        if rec.id not in domain_dict:
            continue
        doms = domain_dict[rec.id]
        L = len(rec.seq)
        if L > max_length:
            try:
                slices = process_sequence(rec, doms, max_length)
            except Exception as e:
                print(f"Error processing sequence {rec.id}: {e}", file=sys.stderr)
                continue
            for new_name, s_start, s_end, s_seq, updated_domains in slices:
                emit_shuf = (random.random() < p_shuf)
                if emit_shuf:
                    shuffled_seq = shuffle_non_domain(str(s_seq), updated_domains)
                    shuffled_name = f"shuffled_{new_name}"
                    new_rec_shuffled = _make_record(shuffled_seq, shuffled_name)
                    _emit_record(output_records, new_domain_dict, new_rec_shuffled, updated_domains)
                    base_id = shuffled_name
                    base_seq_str = shuffled_seq
                else:
                    new_rec = _make_record(str(s_seq), new_name)
                    _emit_record(output_records, new_domain_dict, new_rec, updated_domains)
                    base_id = new_name
                    base_seq_str = str(s_seq)

                if random.random() < p_neg_base:
                    negative_seq = shuffle_entire(base_seq_str)
                    negative_name = f"negative_{base_id}{_rand_neg_suffix()}"
                    _emit_negative(output_records, new_domain_dict, negative_name, negative_seq)
        else:
            emit_shuf = (random.random() < p_shuf)
            if emit_shuf:
                shuffled_seq = shuffle_non_domain(str(rec.seq), doms)
                shuffled_name = f"shuffled_{rec.id}"
                new_rec_shuffled = _make_record(shuffled_seq, shuffled_name)
                _emit_record(output_records, new_domain_dict, new_rec_shuffled, doms)
                base_id = shuffled_name
                base_seq_str = shuffled_seq
            else:
                _emit_record(output_records, new_domain_dict, rec, doms)
                base_id = rec.id
                base_seq_str = str(rec.seq)

            if random.random() < p_neg_base:
                negative_seq = shuffle_entire(base_seq_str)
                negative_name = f"negative_{base_id}{_rand_neg_suffix()}"
                _emit_negative(output_records, new_domain_dict, negative_name, negative_seq)

        # Domain slices (no B/M/E suffix) sampled via p_c_map
        if p_c_map is not None:
            for pfam, dstart, dstop in domain_dict[rec.id]:
                if pfam == "None":
                    continue
                p_c = p_c_map.get(pfam)
                if p_c is None:
                    if len(p_c_map) == 0:
                        continue
                    avg_pc = sum(p_c_map.values()) / max(1, len(p_c_map))
                    p_c = avg_pc
                if random.random() < p_c:
                    dom_seq = rec.seq[dstart - 1:dstop]
                    dom_name = f"{rec.id}/{dstart}-{dstop}"
                    new_rec_dom = _make_record(str(dom_seq), dom_name)
                    _emit_record(
                        output_records,
                        new_domain_dict,
                        new_rec_dom,
                        [(pfam, 1, dstop - dstart + 1)],
                    )

    return output_records, new_domain_dict


def run(args):
    """
    Execute augmentation using a parsed args object that supplies:
      fasta, domain_dict, output_fasta, output_dict, max_length,
      negative_prob, include_domain_slices, shuffle_only, no_shuffle,
      domain_slices_only, large_data, p_shuffled, domain_counts_tsv,
      domain_slice_frac, seed, verbose.
    """
    seed = getattr(args, "seed", 100)
    if seed is not None:
        try:
            random.seed(int(seed))
        except Exception:
            print(f"WARNING: could not set random seed from {seed}")

    vprint = print if getattr(args, "verbose", False) else (lambda *a, **k: None)
    max_length = args.max_length

    # Load FASTA file.
    print(f"Loading FASTA file: {args.fasta}", flush=True)
    fasta_records = list(SeqIO.parse(args.fasta, "fasta"))
    loaded_fasta_count = len(fasta_records)
    print(f"Loaded {loaded_fasta_count} sequences from FASTA file", flush=True)

    # Simplify record IDs if they match the pattern with two pipes.
    print("Simplifying record IDs", flush=True)
    for rec in fasta_records:
        simple_id = simplify_id(rec.id)
        rec.id = simple_id
        rec.name = simple_id
    _ensure_unique_ids(fasta_records, "FASTA")
    print(f"Simplified {len(fasta_records)} record IDs", flush=True)

    # Load domain dictionary.
    print(f"Loading domain dictionary: {args.domain_dict}", flush=True)
    with open(args.domain_dict, "rb") as f:
        domain_dict = pickle.load(f)
    # Accept wrapper dicts with top-level key 'domain_dict' (for compatibility)
    if (
        isinstance(domain_dict, dict)
        and "domain_dict" in domain_dict
        and isinstance(domain_dict["domain_dict"], dict)
    ):
        domain_dict = domain_dict["domain_dict"]

    # Prune domain_dict keys to those present in FASTA (after simplify) and drop short domains
    fasta_id_set = {rec.id for rec in fasta_records}
    domain_dict = _prune_domain_dict_to_fasta(domain_dict, fasta_id_set)
    pruned_domain_count = len(domain_dict)
    # Sort domains once by start to avoid per-record sorting later
    domain_dict = {k: sorted(v, key=lambda x: x[1]) for k, v in domain_dict.items()}
    print(f"Loaded & pruned domain dictionary entries: {pruned_domain_count}", flush=True)

    # Filter FASTA records to those still present in the pruned domain dict
    fasta_records = [rec for rec in fasta_records if rec.id in domain_dict]
    _ensure_unique_ids(fasta_records, "Filtered FASTA")
    filtered_fasta_count = len(fasta_records)
    print(
        f"Filtered FASTA to {filtered_fasta_count} sequences present in domain dict "
        f"(from {loaded_fasta_count}); domain dict entries: {pruned_domain_count}",
        flush=True,
    )

    # Large-data mode branch
    if args.large_data:
        output_records, new_domain_dict = _run_large_data_mode(
            args, fasta_records, domain_dict, max_length, vprint
        )
        _write_outputs(args.output_fasta, args.output_dict, output_records, new_domain_dict)
        print(f"Processed {len(output_records)} sequences written to {args.output_fasta}")
        print(f"New domain dictionary written to {args.output_dict}")

        _print_length_distributions(output_records)
        dslices = sum(
            1 for r in output_records
            if "/" in r.id and not r.id.startswith("negative_")
        )
        print(f"\nLarge-data: emitted {dslices} domain slices (B/M/E suppressed).")
        return

    output_records = []
    new_domain_dict = {}

    # Counters for domain slice statistics
    total_domain_slices = 0
    mid_domain_slices = 0

    p_neg_emit = _neg_emit_probability(args.negative_prob)

    # Wrap sequence iterator with tqdm for a top-level progress bar
    iter_records = tqdm(
        fasta_records,
        desc="Sequences",
        unit="seq",
        mininterval=5,
        miniters=1000,
    )

    for rec in iter_records:
        # 1) Ensure domains exist (already pruned globally)
        if rec.id not in domain_dict:
            vprint(f"DEBUG: Sequence {rec.id} not found in domain dictionary, skipping.")
            continue
        if not domain_dict[rec.id]:
            vprint(f"DEBUG: Sequence {rec.id} has no domains left after filtering, skipping.")
            continue
        doms = domain_dict[rec.id]
        L = len(rec.seq)
        seq_str = str(rec.seq)

        # Special case: single full-length domain with embedded negatives
        if (
            len(doms) == 1
            and doms[0][1] == 1
            and doms[0][2] == L
            and not args.domain_slices_only
        ):
            _emit_full_length_domain_special_case(
                rec=rec,
                doms=doms,
                max_length=max_length,
                fasta_records=fasta_records,
                output_records=output_records,
                new_domain_dict=new_domain_dict,
                p_neg_emit=p_neg_emit,
                vprint=vprint,
                shuffle_only=args.shuffle_only,
            )
            continue

        if not args.domain_slices_only:
            if L > max_length:
                vprint(f"DEBUG: Sequence {rec.id} original domain entry: {doms}")
                try:
                    slices = process_sequence(rec, doms, max_length)
                except Exception as e:
                    print(f"Error processing sequence {rec.id}: {e}", file=sys.stderr)
                    continue
                for new_name, s_start, s_end, s_seq, updated_domains in slices:
                    s_seq_str = str(s_seq)
                    vprint(f"DEBUG: Slice {new_name}: coordinates {s_start}-{s_end}")
                    vprint(f"DEBUG: Updated domain coordinates for {new_name}: {updated_domains}")

                    if not args.shuffle_only:
                        new_rec = _make_record(s_seq_str, new_name)
                        _emit_record(output_records, new_domain_dict, new_rec, updated_domains)
                        _maybe_emit_negative(
                            output_records, new_domain_dict, new_name, s_seq_str, p_neg_emit, vprint
                        )

                    if not args.no_shuffle:
                        shuffled_seq = shuffle_non_domain(s_seq_str, updated_domains)
                        shuffled_name = f"shuffled_{new_name}"
                        new_rec_shuffled = _make_record(shuffled_seq, shuffled_name)
                        _emit_record(output_records, new_domain_dict, new_rec_shuffled, updated_domains)
                        vprint(f"DEBUG: Created shuffled non-domain version: {shuffled_name}")
                        _maybe_emit_negative(
                            output_records, new_domain_dict, shuffled_name, shuffled_seq, p_neg_emit, vprint
                        )
            else:
                if not args.shuffle_only:
                    _emit_record(output_records, new_domain_dict, rec, doms)
                    _maybe_emit_negative(
                        output_records, new_domain_dict, rec.id, seq_str, p_neg_emit, vprint
                    )

                if not args.no_shuffle:
                    shuffled_seq = shuffle_non_domain(seq_str, doms)
                    shuffled_name = f"shuffled_{rec.id}"
                    new_rec_shuffled = _make_record(shuffled_seq, shuffled_name)
                    _emit_record(output_records, new_domain_dict, new_rec_shuffled, doms)
                    vprint(f"DEBUG: Created shuffled non-domain version: {shuffled_name}")
                    _maybe_emit_negative(
                        output_records, new_domain_dict, shuffled_name, shuffled_seq, p_neg_emit, vprint
                    )

        # Include individual domains if requested
        if (args.include_domain_slices or args.domain_slices_only) and rec.id in domain_dict:
            for pfam, dstart, dstop in domain_dict[rec.id]:
                if pfam == "None":
                    continue
                dom_seq_str = str(rec.seq[dstart - 1:dstop])
                # Determine slice label based on domain position within the parent sequence
                if dstart == 1 and dstop == L:
                    slice_label = ""
                elif dstart == 1:
                    slice_label = "_B"
                elif dstop == L:
                    slice_label = "_E"
                else:
                    slice_label = "_M"

                dom_name = f"{rec.id}/{dstart}-{dstop}{slice_label}"

                total_domain_slices += 1
                if slice_label == "_M":
                    mid_domain_slices += 1

                new_rec_dom = _make_record(dom_seq_str, dom_name)
                _emit_record(
                    output_records,
                    new_domain_dict,
                    new_rec_dom,
                    [(pfam, 1, dstop - dstart + 1)],
                )
                vprint(f"DEBUG: Included domain slice: {dom_name}")
                _maybe_emit_domain_slice_negative(
                    output_records,
                    new_domain_dict,
                    rec.id,
                    dstart,
                    dstop,
                    slice_label,
                    dom_seq_str,
                    p_neg_emit,
                    vprint,
                )

    _write_outputs(args.output_fasta, args.output_dict, output_records, new_domain_dict)
    print(f"Processed {len(output_records)} sequences written to {args.output_fasta}")
    print(f"New domain dictionary written to {args.output_dict}")
    _print_length_distributions(output_records)
    _print_domain_slice_stats(total_domain_slices, mid_domain_slices)
