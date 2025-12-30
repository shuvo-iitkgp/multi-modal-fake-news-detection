from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml

def load_config(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())

def deep_get(d: Dict[str, Any], key: str, default=None):
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
