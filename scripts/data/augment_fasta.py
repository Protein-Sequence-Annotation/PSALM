#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


def _add_src_to_path():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Split sequences into domain-preserving slices and emit shuffled "
            "and negative variants. Negative outputs target an approximate "
            "fraction of the final dataset."
        )
    )
    parser.add_argument("--fasta", "-f", required=True, help="Input FASTA file")
    parser.add_argument(
        "--domain-dict",
        "-d",
        required=True,
        help="Pickle file containing domain dictionary: {seq_id: [(pfam_id, start, stop), ...]}",
    )
    parser.add_argument(
        "--output-fasta",
        "-o",
        required=True,
        help="Output FASTA file for slices and additional versions",
    )
    parser.add_argument(
        "--output-dict",
        "-O",
        required=True,
        help="Output pickle file for new domain dictionary (keys are new record names)",
    )
    parser.add_argument(
        "--max-length",
        "-L",
        type=int,
        default=4096,
        help="Maximum length of a sequence to be sliced",
    )
    parser.add_argument(
        "--negative-prob",
        "-p",
        type=float,
        default=0.05,
        help="Target fraction of negatives in the final dataset (approximate)",
    )
    parser.add_argument(
        "--include-domain-slices",
        action="store_true",
        help="Also output each individual annotated domain as its own FASTA record and dict entry",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-record debug information"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--shuffle-only",
        action="store_true",
        help="Only write shuffled and negative sequences; skip the original sequences/slices.",
    )
    group.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not write shuffled sequences; only original and negative versions.",
    )
    group.add_argument(
        "--domain-slices-only",
        action="store_true",
        help="Only write domain slices and their negatives; skip original and shuffled sequences.",
    )
    group.add_argument(
        "--large-data",
        action="store_true",
        help=(
            "Large-data mode: single-copy base records (original OR shuffled), "
            "domain-slice and negative quotas by fraction of final size."
        ),
    )

    # Large-data mode knobs
    parser.add_argument(
        "--p-shuffled",
        type=float,
        default=0.5,
        help="Probability that a base record is emitted as non-domain-shuffled rather than original in --large-data mode.",
    )
    parser.add_argument(
        "--domain-counts-tsv",
        dest="domain_counts_tsv",
        default=None,
        help="TSV with header 'family\\tcount' to compute PFAM frequencies for domain-slice sampling in --large-data mode.",
    )
    parser.add_argument(
        "--domain-slice-frac",
        type=float,
        default=0.05,
        help="Target fraction of final dataset to be domain slices in --large-data mode (expected).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Random seed for all modes.",
    )
    return parser


def main():
    _add_src_to_path()
    from psalm.data import augment_fasta

    parser = build_parser()
    args = parser.parse_args()
    augment_fasta.run(args)


if __name__ == "__main__":
    main()
