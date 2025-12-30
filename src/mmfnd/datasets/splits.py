import numpy as np
import pandas as pd

def event_disjoint_split(df: pd.DataFrame, event_col: str, seed: int, train_frac=0.8, val_frac=0.1):
    rng = np.random.default_rng(seed)
    events = df[event_col].fillna("NO_EVENT").unique().tolist()
    rng.shuffle(events)
    n = len(events)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_events = set(events[:n_train])
    val_events = set(events[n_train:n_train+n_val])
    test_events = set(events[n_train+n_val:])

    def tag(e):
        if e in train_events: return "train"
        if e in val_events: return "val"
        return "test"

    split = df[event_col].fillna("NO_EVENT").map(tag)
    return split
