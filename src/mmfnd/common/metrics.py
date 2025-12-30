from dataclasses import dataclass
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

@dataclass
class Metrics:
    accuracy: float
    macro_f1: float

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
    )
