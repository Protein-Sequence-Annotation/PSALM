from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class EValueCurve:
    norm_factor: float
    ref_dataset_size: float
    score_sorted_asc: np.ndarray
    fp_counts_sorted_asc: np.ndarray

    @classmethod
    def from_json(cls, path: Path) -> "EValueCurve":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        score_sorted = np.asarray(raw["score_sorted_asc"], dtype=float)
        fp_counts_sorted = np.asarray(raw["fp_counts_sorted_asc"], dtype=float)
        norm_factor = float(raw.get("norm_factor", 24076.0))
        ref_dataset_size = float(raw["ref_dataset_size"])
        if score_sorted.size == 0 or fp_counts_sorted.size == 0:
            raise ValueError("E-value curve JSON is empty.")
        if score_sorted.size != fp_counts_sorted.size:
            raise ValueError("E-value curve JSON has mismatched score/fp-count lengths.")
        if not np.all(np.isfinite(score_sorted)):
            raise ValueError("E-value curve JSON contains non-finite scores.")
        if not np.all(np.isfinite(fp_counts_sorted)):
            raise ValueError("E-value curve JSON contains non-finite FP counts.")
        if norm_factor <= 0:
            raise ValueError("E-value curve JSON must define a positive norm_factor.")
        if ref_dataset_size <= 0:
            raise ValueError("E-value curve JSON must define a positive ref_dataset_size.")
        if np.any(np.diff(score_sorted) <= 0):
            raise ValueError("E-value curve scores must be strictly increasing.")
        if np.any(fp_counts_sorted < 0):
            raise ValueError("E-value curve FP counts must be non-negative.")
        if np.any(np.diff(fp_counts_sorted) > 0):
            raise ValueError(
                "E-value curve FP counts must be non-increasing as the score threshold increases."
            )
        return cls(
            norm_factor=norm_factor,
            ref_dataset_size=ref_dataset_size,
            score_sorted_asc=score_sorted,
            fp_counts_sorted_asc=fp_counts_sorted,
        )

    def evalue_from_score(self, score: float, dataset_size: float) -> float:
        z = max(float(dataset_size), 1e-12)
        fp = float(np.interp(score, self.score_sorted_asc, self.fp_counts_sorted_asc))
        # E-value is per family, so normalize expected FP count by number of families.
        return (fp / self.norm_factor) * (z / self.ref_dataset_size)

    def score_from_evalue(self, evalue: float, dataset_size: float) -> float:
        z = max(float(dataset_size), 1e-12)
        target_fp = float(evalue) * self.norm_factor * (self.ref_dataset_size / z)

        # fp_counts_sorted_asc decreases as score grows; reverse for np.interp.
        x_asc = self.fp_counts_sorted_asc[::-1]
        score_for_x = self.score_sorted_asc[::-1]
        return float(np.interp(target_fp, x_asc, score_for_x))


def load_default_curve(base_dir: Optional[Path] = None) -> EValueCurve:
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    curve_path = base_dir / "evalue_curve.json"
    return EValueCurve.from_json(curve_path)
