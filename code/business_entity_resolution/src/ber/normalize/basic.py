"""Baseline normalizer: country-agnostic text cleanup.

Adds: name_norm, addr_norm, name_core (legal-form tokens removed), addr_digits (digit tokens).
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from ber.interfaces import Normalizer
from ber.registry import register

_SPACES = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")


def clean(text: str) -> str:
    """Lowercase, strip Latin accents (é->e) but keep non-Latin scripts intact, & -> and, punctuation -> space."""
    text = unicodedata.normalize("NFKD", str(text))
    out, prev_ascii = [], False
    for ch in text:
        if unicodedata.combining(ch) and prev_ascii:
            continue                       # accent on a Latin letter: drop it
        out.append(ch)
        prev_ascii = ch.isascii()
    text = unicodedata.normalize("NFC", "".join(out)).lower().replace("&", " and ").replace("'", "")
    # keep letters/digits of ANY script + combining marks (Devanagari vowel signs); everything else -> space
    text = "".join(ch if (ch.isalnum() or unicodedata.category(ch)[0] == "M") else " " for ch in text)
    return _SPACES.sub(" ", text).strip()


@register("normalizer", "basic")
class BasicNormalizer(Normalizer):
    def __init__(self, legal_tokens: list[str] | None = None):
        self.legal = set(legal_tokens or [])

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        out = records.copy()
        out["name_norm"] = out["business_name"].map(clean)
        out["addr_norm"] = out["business_address"].map(clean)
        legal = self.legal
        out["name_core"] = out["name_norm"].map(
            lambda s: " ".join(t for t in s.split() if t not in legal) or s)
        out["addr_digits"] = out["addr_norm"].map(lambda s: " ".join(_DIGITS.findall(s)))
        return out
