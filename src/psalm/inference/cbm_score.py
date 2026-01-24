from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:
    from catboost import CatBoostClassifier, Pool
except ImportError:  # pragma: no cover - handled at runtime
    CatBoostClassifier = None
    Pool = None

DomainTuple = Tuple[str, int, int, float, float, float, str]
ScoredDomainTuple = Tuple[str, int, int, float, float, float, float, str]


def _require_catboost() -> None:
    if CatBoostClassifier is None or Pool is None:
        raise ImportError(
            "CatBoost is required for CBM scoring. Install catboost to use scan()."
        )


def compute_adjusted_lengths(runs: Sequence[DomainTuple]) -> List[int]:
    """Compute adjusted lengths for containment-corrected CBM features."""
    if not runs:
        return []
    starts = np.array([int(r[1]) for r in runs], dtype=np.int64)
    stops = np.array([int(r[2]) for r in runs], dtype=np.int64)
    lens = (stops - starts + 1).astype(np.int64)
    n = len(runs)

    order = np.lexsort((stops, starts))
    s_sorted = starts[order]
    e_sorted = stops[order]
    has_containment = False
    max_e = -1
    last_s = None
    last_e = None
    for si, ei in zip(s_sorted, e_sorted):
        if si == last_s and ei == last_e:
            pass
        else:
            if max_e >= ei and max_e != ei:
                has_containment = True
                break
        if ei > max_e:
            max_e = ei
        last_s = si
        last_e = ei
    if not has_containment:
        return lens.tolist()

    adj = lens.copy()
    for i in range(n):
        s_i = starts[i]
        e_i = stops[i]
        same_span = (starts == s_i) & (stops == e_i)
        contained = (starts >= s_i) & (stops <= e_i) & (~same_span)
        sub_sum = int(lens[contained].sum())
        a = int(lens[i]) - sub_sum
        if a < 0:
            a = 0
        adj[i] = a
    return adj.tolist()


def _build_feature_matrix(
    runs: Sequence[DomainTuple],
    adjusted_lengths: Sequence[int],
) -> np.ndarray:
    n = len(runs)
    X = np.empty((n, 7), dtype=object)
    ratio = np.empty(n, dtype=np.float64)
    bias = np.empty(n, dtype=np.float64)
    length = np.empty(n, dtype=np.float64)
    score = np.empty(n, dtype=np.float64)
    status = np.empty(n, dtype=object)

    for i, r in enumerate(runs):
        score[i] = float(r[3])
        ratio[i] = float(r[4])
        bias[i] = float(r[5])
        length[i] = float(max(0, int(adjusted_lengths[i])))
        status[i] = str(r[6])

    abs_len_err = np.abs(1.0 - ratio)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_len = np.where(ratio != 0.0, length / ratio, np.nan)
        adj_score = np.where(
            (expected_len != 0.0) & (~np.isnan(expected_len)),
            score / expected_len,
            np.nan,
        )

    X[:, 0] = ratio
    X[:, 1] = bias
    X[:, 2] = length
    X[:, 3] = abs_len_err
    X[:, 4] = expected_len
    X[:, 5] = adj_score
    X[:, 6] = status
    return X


def score_domains(
    runs: Sequence[DomainTuple],
    model: CatBoostClassifier,
) -> List[float]:
    _require_catboost()
    if not runs:
        return []
    adjusted = compute_adjusted_lengths(runs)
    X = _build_feature_matrix(runs, adjusted)
    pool = Pool(X, label=None, cat_features=[6])
    probs = model.predict_proba(pool)[:, 1]
    return [float(p) for p in probs]


def add_cbm_scores(
    runs: Sequence[DomainTuple],
    model: CatBoostClassifier,
) -> List[ScoredDomainTuple]:
    _require_catboost()
    scores = score_domains(runs, model)
    return [
        (
            pfam,
            int(start),
            int(stop),
            float(scores[i]),
            float(bit_score),
            float(len_ratio),
            float(bias),
            str(status),
        )
        for i, (pfam, start, stop, bit_score, len_ratio, bias, status) in enumerate(runs)
    ]

