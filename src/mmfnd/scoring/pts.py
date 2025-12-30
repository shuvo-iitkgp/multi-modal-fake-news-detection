from __future__ import annotations
import numpy as np

def pts(prob_real: np.ndarray, dtf: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Preliminary Trustworthiness Score proxy:
    blend model probability and dtf.
    """
    return alpha * prob_real + (1 - alpha) * (0.5 + 0.5 * dtf)
