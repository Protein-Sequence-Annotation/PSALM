"""
data_processing.py

Process an input FASTA file and convert it to tokenized, padded, and batched
Hugging Face Dataset format. In addition to tokenization, it generates
ground-truth labels based on domain coordinates and a label mapping dictionary.
"""

import os
import pickle
import random
import shutil
from collections import defaultdict

from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
from tqdm import tqdm


def generate_labels(sequence, token_output, domain_info, label_mapping, seq_id, ignore_label):
    """
    Generates ground truth labels for a tokenized sequence.

    Args:
      sequence      : Raw sequence string.
      token_output  : Dictionary with keys "input_ids" and "attention_mask" from the tokenizer.
      domain_info   : List of domain tuples [(pfam, start, stop), ...] (1-indexed, inclusive).
      label_mapping : Dictionary mapping domain IDs to label dictionaries, with "None" mapped to an int.
      seq_id        : The sequence ID.

    Returns:
      A list of integer labels.
    """
    n = len(token_output["input_ids"])
    labels = [ignore_label] * n

    assert len(token_output["input_ids"]) == len(token_output["attention_mask"]), (
        f"Token output length mismatch for {seq_id}: {len(token_output['input_ids'])} "
        f"vs {len(token_output['attention_mask'])}"
    )

    if seq_id.startswith("negative_"):
        none_val = label_mapping["None"]
        labels = [none_val] * n
        labels[0] = ignore_label
        labels[-1] = ignore_label
    else:
        is_shuffled = seq_id.startswith("shuffled_")
        if is_shuffled:
            none_val = label_mapping["None"]
            labels = [none_val] * n
            labels[0] = ignore_label
            labels[-1] = ignore_label

        if domain_info is None or len(domain_info) == 0:
            raise ValueError(f"Sequence {seq_id} not found in domain dictionary.")

        for pfam, dstart, dstop in sorted(domain_info, key=lambda x: x[1]):
            start_idx = dstart
            stop_idx = dstop

            if dstart < 1 or dstop > len(sequence):
                raise ValueError(
                    f"Domain {(pfam, dstart, dstop)} out of range for sequence "
                    f"{seq_id} of length {len(sequence)}."
                )

            if dstart == dstop:
                labels[start_idx] = label_mapping.get(pfam, {}).get("start", label_mapping["None"])
            else:
                labels[start_idx] = label_mapping.get(pfam, {}).get("start", label_mapping["None"])
                if stop_idx >= n:
                    raise IndexError(
                        f"Calculated stop index {stop_idx} out of range for sequence {seq_id} "
                        f"(token length {n}).\n token_output: {token_output}"
                    )
                labels[stop_idx] = label_mapping.get(pfam, {}).get("stop", label_mapping["None"])
                for i in range(start_idx + 1, stop_idx):
                    labels[i] = label_mapping.get(pfam, {}).get("middle", label_mapping["None"])

        assert any(label != ignore_label for label in labels), (
            f"After processing, all labels for sequence {seq_id} remain ignore_label. "
            f"Domain info: {domain_info}"
        )

    if seq_id.endswith("_B"):
        labels = labels[:-1]
        token_output["input_ids"] = token_output["input_ids"][:-1]
        token_output["attention_mask"] = token_output["attention_mask"][:-1]
    elif seq_id.endswith("_M"):
        labels = labels[1:-1]
        token_output["input_ids"] = token_output["input_ids"][1:-1]
        token_output["attention_mask"] = token_output["attention_mask"][1:-1]
    elif seq_id.endswith("_E"):
        labels = labels[1:]
        token_output["input_ids"] = token_output["input_ids"][1:]
        token_output["attention_mask"] = token_output["attention_mask"][1:]

    assert len(token_output["input_ids"]) == len(token_output["attention_mask"]) == len(labels), (
        f"Final length mismatch for {seq_id}: input_ids {len(token_output['input_ids'])}, "
        f"attention_mask {len(token_output['attention_mask'])}, labels {len(labels)}"
    )
    assert any(label != ignore_label for label in labels), (
        f"Final label check failed for {seq_id}: all labels are ignore_label."
    )

    return labels


def tokenize_and_label(seq_id, sequence, domain_info, label_mapping, tokenizer, max_length, ignore_label):
    """
    Tokenizes the sequence and generates corresponding ground-truth labels.

    Returns:
      A dictionary with keys "input_ids", "attention_mask", "length", and "labels".
    """
    tokens = tokenizer(
        sequence,
        truncation=True,
        max_length=max_length + 2,  # +2 for [CLS] and [SEP]
        padding=False,
        return_attention_mask=True,
    )
    tokens["length"] = len(tokens["input_ids"])
    tokens["labels"] = generate_labels(
        sequence, tokens, domain_info, label_mapping, seq_id, ignore_label
    )

    assert len(tokens["input_ids"]) == len(tokens["attention_mask"]) == len(tokens["labels"]), (
        f"After tokenization, sequence {seq_id} has mismatched lengths: "
        f"input_ids {len(tokens['input_ids'])}, attention_mask {len(tokens['attention_mask'])}, "
        f"labels {len(tokens['labels'])}"
    )
    return tokens


def pad_batch(batch, max_length, pad_token_id, ignore_label):
    """
    Pads each example in a batch to the target max_length.
    """
    for example in batch:
        padding_length = max_length - len(example["input_ids"])
        example["input_ids"] += [pad_token_id] * padding_length
        example["attention_mask"] += [0] * padding_length
        example["labels"] += [ignore_label] * padding_length

        assert len(example["input_ids"]) == max_length, (
            f"Padding error in input_ids: {len(example['input_ids'])} != {max_length}"
        )
        assert len(example["attention_mask"]) == max_length, (
            f"Padding error in attention_mask: {len(example['attention_mask'])} != {max_length}"
        )
        assert len(example["labels"]) == max_length, (
            f"Padding error in labels: {len(example['labels'])} != {max_length}"
        )
    return batch


def save_chunk_to_disk(chunk, chunk_idx, output_dir="data/tmp_chunks"):
    """
    Saves a chunk of processed data to disk as a Hugging Face Dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    chunk_file = os.path.join(output_dir, f"chunk_{chunk_idx}.json")
    Dataset.from_list(chunk).save_to_disk(chunk_file)
    print(f"Saved chunk {chunk_idx} to {chunk_file}")


def refine_fasta(input_fasta, output_fasta, max_length=4096):
    """
    Reads a FASTA file, retains only sequences not exceeding max_length,
    and writes them to a new FASTA file.
    """
    print("Reading FASTA file...")
    refined_sequences = {}
    excluded_count = 0
    duplicate_count = 0
    sequences = {}

    with open(input_fasta, "r") as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    if current_id not in sequences:
                        sequences[current_id] = "".join(current_seq)
                    else:
                        duplicate_count += 1
                current_id = line
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            if current_id not in sequences:
                sequences[current_id] = "".join(current_seq)
            else:
                duplicate_count += 1

    for seq_id, seq in sequences.items():
        if len(seq) <= max_length:
            refined_sequences[seq_id] = seq
        else:
            print(
                f"Excluding sequence {seq_id} of length {len(seq)} "
                f"(longer than {max_length} characters)."
            )
            excluded_count += 1

    print(f"Excluded {excluded_count} sequences longer than {max_length} characters.")
    print(f"Refined FASTA contains {len(refined_sequences)} sequences.")
    print(f"Skipped {duplicate_count} duplicate IDs.")

    with open(output_fasta, "w") as out_f:
        for seq_id, seq in refined_sequences.items():
            out_f.write(f"{seq_id}\n")
            out_f.write(f"{seq}\n")

    print(f"Refined FASTA written to {output_fasta}.")
    return refined_sequences


def preprocess_fasta(
    file_path,
    tokenizer,
    max_length,
    max_tokens_per_batch,
    domain_dict,
    label_mapping,
    ignore_label,
    chunk_size=1000000,
    batch_dir="data/tmp_chunks",
):
    """
    Preprocesses a FASTA file into tokenized, labeled batches and saves chunks to disk.
    """

    def tokenize_and_label_wrapper(seq_id, sequence):
        simplified_seq_id = seq_id.split()[0][1:]
        if simplified_seq_id not in domain_dict:
            raise ValueError(f"Sequence {simplified_seq_id} not found in domain dictionary.")
        domain_info = domain_dict[simplified_seq_id]
        return tokenize_and_label(
            simplified_seq_id,
            sequence,
            domain_info,
            label_mapping,
            tokenizer,
            max_length,
            ignore_label,
        )

    print("Refining FASTA file...")
    raw_dir = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    name_no_ext = os.path.splitext(base_name)[0]
    refined_fasta_path = os.path.join(raw_dir, f"{name_no_ext}_refined.fasta")
    refined_sequences = refine_fasta(file_path, refined_fasta_path, max_length=max_length)

    sorted_sequences = sorted(refined_sequences.items(), key=lambda x: len(x[1]))
    print("Processing sequences in chunks...")
    chunked_data = []
    chunk_count = 0
    batch_id = 0
    current_batch = []
    max_padded_length = 0

    debug_examples = []

    for idx, (seq_id, seq) in enumerate(tqdm(sorted_sequences, desc="Tokenizing Sequences")):
        tokens = tokenize_and_label_wrapper(seq_id, seq)
        sequence_length = len(tokens["input_ids"])
        max_padded_length = max(max_padded_length, sequence_length)
        padded_tokens = max_padded_length * (len(current_batch) + 1)

        if padded_tokens > max_tokens_per_batch:
            current_batch = pad_batch(
                current_batch, max_padded_length, tokenizer.pad_token_id, ignore_label
            )
            for example in current_batch:
                example["batch_id"] = batch_id
            chunked_data.extend(current_batch)
            batch_id += 1
            current_batch = []
            max_padded_length = sequence_length

        current_batch.append(
            {
                "id": seq_id,
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "labels": tokens["labels"],
                "sequence_length": sequence_length,
            }
        )

        if idx < 5:
            debug_examples.append(
                {
                    "id": seq_id,
                    "input_ids": tokens["input_ids"],
                    "labels": tokens["labels"],
                }
            )

        if len(chunked_data) >= chunk_size:
            save_chunk_to_disk(chunked_data, chunk_count, batch_dir)
            chunked_data = []
            chunk_count += 1

    if current_batch:
        current_batch = pad_batch(
            current_batch, max_padded_length, tokenizer.pad_token_id, ignore_label
        )
        for example in current_batch:
            example["batch_id"] = batch_id
        chunked_data.extend(current_batch)

    if chunked_data:
        save_chunk_to_disk(chunked_data, chunk_count, batch_dir)

    print(f"Processed and saved {chunk_count + 1} chunks.")
    print("Debug sample of tokenized sequences with labels:")
    for ex in debug_examples:
        print(f"Sequence ID: {ex['id']}")
        print(f"token ids: {ex['input_ids']}")
        print(f"First 20 labels: {ex['labels']}")


def merge_and_shuffle_batches(batch_dir, output_dir, shard_size=25000, seed=100):
    """
    Merges and shuffles processed chunks from disk and saves them as dataset shards.
    """
    print("Merging and shuffling batches...")
    chunk_files = [os.path.join(batch_dir, f) for f in os.listdir(batch_dir) if f.startswith("chunk_")]
    datasets = [Dataset.load_from_disk(chunk_file) for chunk_file in chunk_files]
    merged_dataset = concatenate_datasets(datasets)
    print(f"Merged {len(datasets)} chunks into one dataset with {len(merged_dataset)} examples.")

    print("Grouping dataset by batch_id...")
    batch_map = defaultdict(list)
    for example in merged_dataset:
        batch_map[example["batch_id"]].append(example)

    all_batches = list(batch_map.values())
    print(f"Total batches: {len(all_batches)}")
    print("Shuffling batches...")
    rng = random.Random(seed)
    rng.shuffle(all_batches)

    print("Saving shards...")
    os.makedirs(output_dir, exist_ok=True)
    shard_count = 0
    for i in range(0, len(all_batches), shard_size):
        shard_batches = all_batches[i:i + shard_size]
        shard_examples = [ex for batch in shard_batches for ex in batch]
        shard_dataset = Dataset.from_list(shard_examples)
        shard_dir = os.path.join(output_dir, f"shard-{shard_count:05d}")
        shard_dataset.save_to_disk(shard_dir)
        shard_count += 1

    print(f"Dataset saved with {shard_count} shards.")


def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    preprocess_fasta(
        args.fasta,
        tokenizer,
        args.max_length,
        args.max_tokens_per_batch,
        domain_dict=pickle.load(open(args.domain_dict, "rb")),
        label_mapping=pickle.load(open(args.label_mapping_dict, "rb")),
        ignore_label=args.ignore_label,
        chunk_size=args.chunk_size,
        batch_dir=args.tmp_dir,
    )

    print("Merging, shuffling, and saving datasets...")
    merge_and_shuffle_batches(
        args.tmp_dir,
        args.output_dir,
        shard_size=args.shard_size,
        seed=args.seed,
    )
    if getattr(args, "keep_tmp", False):
        print(f"Keeping temporary directory {args.tmp_dir}...")
    else:
        print(f"Removing temporary directory {args.tmp_dir}...")
        shutil.rmtree(args.tmp_dir, ignore_errors=True)
    print("Processing complete.")
    print("================================================")
