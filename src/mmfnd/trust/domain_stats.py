from __future__ import annotations
from collections import Counter, defaultdict
from typing import Dict, Any, Iterable

def build_domain_stats(samples: Iterable[dict]) -> dict:
    """
    samples: iterable of sample dicts with keys: domains (list[str]) and label (int)
    For each domain: count true vs false (label=1/0 or your mapping).
    We assume label 1 = real, 0 = fake by default. If opposite, flip outside.
    """
    stats = defaultdict(lambda: Counter())
    for s in samples:
        y = s.get("label", None)
        if y is None:
            continue
        for d in s.get("domains", []) or []:
            if d:
                stats[d]["n"] += 1
                if int(y) == 1:
                    stats[d]["real"] += 1
                else:
                    stats[d]["fake"] += 1
    return {k: dict(v) for k, v in stats.items()}
