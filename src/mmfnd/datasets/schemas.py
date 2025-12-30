from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class Sample:
    id: str
    text: str
    image_path: Optional[str]
    label: Optional[int]  # 0/1 or multiclass
    lang: Optional[str]
    urls: List[str]
    domains: List[str]
    event_id: Optional[str]
    topic_id: Optional[str]
    meta: Dict[str, Any]
