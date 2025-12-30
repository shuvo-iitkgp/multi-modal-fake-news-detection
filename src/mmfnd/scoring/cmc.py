from __future__ import annotations
import numpy as np

def cmc(prob_real: np.ndarray, dtf: np.ndarray, w_model: float = 0.7) -> np.ndarray:
    """
    Combined Metric of Classification proxy:
    combine classifier confidence with domain trust.
    """
    return w_model * prob_real + (1 - w_model) * (0.5 + 0.5 * dtf)
