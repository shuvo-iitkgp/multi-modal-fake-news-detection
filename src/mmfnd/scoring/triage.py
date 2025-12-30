from __future__ import annotations
import numpy as np

def rank_for_verification(cmpu_scores: np.ndarray) -> np.ndarray:
    """
    returns indices sorted descending by CMPU.
    """
    return np.argsort(-cmpu_scores)

def select_top_fraction(ranking: np.ndarray, frac: float) -> np.ndarray:
    k = int(round(len(ranking) * frac))
    k = max(0, min(len(ranking), k))
    return ranking[:k]
