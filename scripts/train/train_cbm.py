#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from psalm.inference.cbm_score import (  # noqa: E402
    _build_feature_matrix,
    compute_adjusted_lengths,
)

DomainTuple = Tuple[str, int, int, float, float, float, str]


def _cast_row(row: Sequence[object]) -> DomainTuple:
    if len(row) == 7:
        return (
            str(row[0]),
            int(row[1]),
            int(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
            str(row[6]),
        )
    if len(row) == 8:
        return (
            str(row[0]),
            int(row[1]),
            int(row[2]),
            float(row[4]),
            float(row[5]),
            float(row[6]),
            str(row[7]),
        )
    raise ValueError(f"Expected 7 or 8 fields, got {len(row)}")


def _flatten_records(raw: Iterable[object]) -> List[DomainTuple]:
    rows: List[DomainTuple] = []
    for i, rec in enumerate(raw):
        if isinstance(rec, (list, tuple)) and len(rec) in (7, 8) and not any(
            isinstance(x, (list, tuple)) for x in rec
        ):
            rows.append(_cast_row(rec))
            continue
        if (
            isinstance(rec, (list, tuple))
            and len(rec) > 0
            and all(isinstance(x, (list, tuple)) and len(x) in (7, 8) for x in rec)
        ):
            rows.extend(_cast_row(x) for x in rec)
            continue
        raise ValueError(
            f"Element {i} is not a 7/8-tuple or a container of 7/8-tuples. "
            f"type={type(rec).__name__}, value sample={repr(rec)[:200]}"
        )
    return rows


def load_records(path: Path) -> List[DomainTuple]:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            raw = pickle.load(handle)
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    else:
        raise ValueError("Expected a .pkl/.pickle or .json file for inputs.")
    if isinstance(raw, dict):
        merged: List[object] = []
        for _, runs in raw.items():
            merged.append(runs)
        return _flatten_records(merged)
    if not isinstance(raw, (list, tuple)):
        raise ValueError("Input must be a list/dict of 7/8-tuples or a list of lists.")
    return _flatten_records(raw)


def build_features(records: Sequence[DomainTuple]) -> np.ndarray:
    adjusted = compute_adjusted_lengths(records)
    return _build_feature_matrix(records, adjusted)


def build_model(args: argparse.Namespace) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        iterations=args.iterations,
        random_seed=args.seed,
        verbose=args.verbose,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Train CatBoost CBM scorer using scan-style tuples. Accepts "
            "7-tuples (pfam, start, stop, bit_score, len_ratio, bias, status) "
            "or 8-tuples from scan() (pfam, start, stop, cbm_score, bit_score, "
            "len_ratio, bias, status), or a dict {seq_id: [tuples]}."
        )
    )
    ap.add_argument(
        "--pos",
        required=True,
        help="Path to positive .pkl/.json list or dict of scan tuples.",
    )
    ap.add_argument(
        "--neg",
        required=True,
        help="Path to negative .pkl/.json list or dict of scan tuples.",
    )
    ap.add_argument("--outdir", default="cbm_outputs", help="Output directory.")
    ap.add_argument("--model-out", default="score.cbm", help="Model filename (within outdir).")
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--l2-leaf-reg", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pos_records = load_records(Path(args.pos))
    neg_records = load_records(Path(args.neg))

    records = pos_records + neg_records
    y = np.array([1] * len(pos_records) + [0] * len(neg_records), dtype=int)

    X = build_features(records)
    cat_features = [6]

    weights = None
    if not args.no_class_weights:
        classes = np.array([0, 1])
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        weight_map = {int(c): float(w) for c, w in zip(classes, class_weights)}
        weights = np.array([weight_map[int(label)] for label in y], dtype=float)

    pool = Pool(X, label=y, cat_features=cat_features, weight=weights)
    model = build_model(args)
    model.fit(pool, verbose=args.verbose)

    probs = model.predict_proba(pool)[:, 1]
    metrics = {
        "n_samples": int(len(y)),
        "roc_auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y, probs)) if len(np.unique(y)) == 2 else float("nan"),
        "params": {
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "depth": args.depth,
            "l2_leaf_reg": args.l2_leaf_reg,
            "seed": args.seed,
            "class_weights": None if args.no_class_weights else weight_map,
        },
    }
    with (outdir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    model_path = outdir / args.model_out
    model.save_model(str(model_path))
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
