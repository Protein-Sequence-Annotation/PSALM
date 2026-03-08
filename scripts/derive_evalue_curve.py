from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


DEFAULT_NEG_SCORES = Path(
    "/n/eddy_lab/Lab/protein_annotation_dl/PSALM-2/results/evalue_estimation_as_scored/neg_scores.pkl"
)
DEFAULT_OUT = Path("psalm/inference/evalue_curve.json")

NORM_FACTOR = 24_076.0
REF_DATASET_SIZE = 30_000_000.0


def build_curve(neg_scores_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with neg_scores_path.open("rb") as handle:
        scores = np.asarray(pickle.load(handle), dtype=float)
    if scores.size == 0:
        raise ValueError("neg_scores.pkl is empty; cannot build E-value curve.")

    # Unique score thresholds (descending) and empirical FP tail counts.
    score_thresholds = np.unique(np.sort(scores)[::-1])
    fp_counts = np.asarray([(scores >= t).sum() for t in score_thresholds], dtype=float)

    # Store score ascending for score->fp-count interpolation.
    order = np.argsort(score_thresholds)
    score_sorted = score_thresholds[order]
    fp_counts_sorted = fp_counts[order]
    return score_sorted, fp_counts_sorted


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive packaged E-value interpolation curve JSON.")
    parser.add_argument("--neg-scores", type=Path, default=DEFAULT_NEG_SCORES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    score_sorted, fp_counts_sorted = build_curve(args.neg_scores)
    payload = {
        "norm_factor": NORM_FACTOR,
        "ref_dataset_size": REF_DATASET_SIZE,
        "score_sorted_asc": score_sorted.tolist(),
        "fp_counts_sorted_asc": fp_counts_sorted.tolist(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=True)

    print(f"Wrote curve JSON to {args.out} with {len(score_sorted)} score points.")


if __name__ == "__main__":
    main()
