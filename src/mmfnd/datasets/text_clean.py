import re
from urllib.parse import urlparse

_URL_RE = re.compile(r"https?://\S+|www\.\S+")

def extract_urls(text: str) -> list[str]:
    return _URL_RE.findall(text or "")

def domains_from_urls(urls: list[str]) -> list[str]:
    out = []
    for u in urls:
        if u.startswith("www."):
            u = "http://" + u
        try:
            host = urlparse(u).netloc.lower()
            host = host.split(":")[0]
            if host:
                out.append(host)
        except Exception:
            continue
    return out

def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text
