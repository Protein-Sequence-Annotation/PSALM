#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


DEFAULT_CHUNK_SIZE = 5_000_000
DEFAULT_TMP_DIR = "data/tmp_chunks"
DEFAULT_SHARD_SIZE = 250_000


def _add_src_to_path():
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    sys.path.insert(0, str(src_root))


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess a FASTA file by tokenizing and generating ground truth "
            "labels based on domain coordinates and a label mapping. Sequences "
            "are batched and saved to disk in chunks."
        )
    )
    parser.add_argument("--fasta", "-f", required=True, help="Path to the input FASTA file")
    parser.add_argument(
        "--domain-dict",
        "-D",
        required=True,
        help="Path to the domain dictionary pickle file: {seq_id: [(pfam_id, start, stop), ...]}",
    )
    parser.add_argument(
        "--label-mapping-dict",
        "-L",
        required=True,
        help="Path to the label mapping dictionary pickle file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory to save the processed dataset shards",
    )
    parser.add_argument(
        "--tmp-dir",
        default=DEFAULT_TMP_DIR,
        help="Temporary directory for chunked outputs",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of sequences per chunk",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="Number of batches per dataset shard",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Tokenizer model name for AutoTokenizer.from_pretrained",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        required=True,
        help="Maximum sequence length (excluding BOS/EOS tokens).",
    )
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        required=True,
        help="Maximum total tokens per batch for pre-batching",
    )
    parser.add_argument(
        "--ignore-label",
        type=int,
        required=True,
        help="Label value used for ignored tokens (e.g., -100).",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep temporary chunk directory instead of deleting it",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Random seed for batch shuffling",
    )
    return parser


def main():
    _add_src_to_path()
    from psalm.data import data_processing

    parser = build_parser()
    args = parser.parse_args()

    data_processing.run(args)


if __name__ == "__main__":
    main()
