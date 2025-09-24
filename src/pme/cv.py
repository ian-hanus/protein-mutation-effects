"""Cross-validation that holds out whole sequence positions."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def position_folds(positions: np.ndarray, n_folds: int = 5, seed: int = 0):
    """Yield (train_idx, test_idx) row indices, K folds, split by *position*.

    Every mutation at a position lands in the same fold, so a model is never
    tested on a site it has seen other substitutions of. A random split over
    mutations would leak that site-level information and overstate performance.
    """
    positions = np.asarray(positions)
    unique = np.unique(positions)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    fold_of = {p: i % n_folds for i, p in enumerate(unique)}
    folds = np.array([fold_of[p] for p in positions])
    for k in range(n_folds):
        yield np.where(folds != k)[0], np.where(folds == k)[0]


def spearman(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 3 or np.std(pred) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(y, pred).statistic)
