from typing import Callable, Dict, Any

_REGISTRY: Dict[str, Callable[..., Any]] = {}

def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco

def get(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"Component not registered: {name}")
    return _REGISTRY[name]

def available():
    return sorted(_REGISTRY.keys())
