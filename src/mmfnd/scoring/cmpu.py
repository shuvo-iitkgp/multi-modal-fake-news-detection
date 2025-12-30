from __future__ import annotations
import numpy as np

def entropy(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

def cmpu(prob_real: np.ndarray, dtf: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """
    Combined Metric of Prediction Uncertainty proxy:
    higher means more uncertain or risky, should be triaged for verification.
    """
    u = entropy(prob_real)
    # if dtf is low or negative, increase uncertainty signal
    trust_penalty = 1.0 - (0.5 + 0.5 * dtf)  # in [0,1] roughly
    return u + beta * trust_penalty
