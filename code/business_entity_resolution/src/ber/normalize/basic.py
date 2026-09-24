"""Baseline normalizer: country-agnostic text cleanup.

Adds: name_norm, addr_norm, name_core (legal-form tokens removed), addr_digits (digit tokens).
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from ber.interfaces import Normalizer
from ber.registry import register

_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_DIGITS = re.compile(r"\d+")


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = text.replace("&", " and ").replace("'", "")
    return _NON_ALNUM.sub(" ", text).strip()


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
