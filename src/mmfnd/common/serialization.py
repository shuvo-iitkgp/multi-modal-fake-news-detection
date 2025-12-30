import json
from pathlib import Path
from typing import Any

def dump_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())
