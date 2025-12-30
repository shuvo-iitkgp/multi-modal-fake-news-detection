from __future__ import annotations
from typing import Dict
import math

def dcf(real: int, fake: int) -> float:
    """
    Domain Confidence Factor: bounded, smooth.
    This is a sane proxy if paper-specific formula differs.
    """
    n = real + fake
    if n == 0:
        return 0.0
    p = real / n
    # confidence grows with n, penalize tiny n
    conf = 1.0 - math.exp(-n / 20.0)
    # map p in [0,1] to [-1,1]
    val = (2.0 * p - 1.0) * conf
    return float(val)

def dtf_for_sample(domains: list[str], domain_stats: Dict[str, Dict[str, int]]) -> float:
    if not domains:
        return 0.0
    vals = []
    for d in domains:
        st = domain_stats.get(d, None)
        if not st:
            continue
        vals.append(dcf(int(st.get("real", 0)), int(st.get("fake", 0))))
    if not vals:
        return 0.0
    # mean trust signal
    return float(sum(vals) / len(vals))
