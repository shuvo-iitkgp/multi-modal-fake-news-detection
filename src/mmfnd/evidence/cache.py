from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

def load_evidence_jsonl(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    out: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out[str(obj["id"])] = obj
    return out
