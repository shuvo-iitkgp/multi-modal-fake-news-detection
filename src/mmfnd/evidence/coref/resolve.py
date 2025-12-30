from __future__ import annotations
import re

_PRONOUN_RE = re.compile(r"\b(he|she|they|his|her|their|him|them|it)\b", re.IGNORECASE)

def simple_coref_resolve(text: str) -> str:
    """
    This is not real coreference resolution.
    It is a safe placeholder that reduces pronoun noise for retrieval by:
      - removing standalone pronouns (keeps sentence structure mostly intact)
    Real coref needs heavy models; add later if you want.
    """
    if not text:
        return text
    text = _PRONOUN_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
