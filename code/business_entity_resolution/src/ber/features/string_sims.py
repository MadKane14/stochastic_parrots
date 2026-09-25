"""Baseline pairwise features: rapidfuzz string similarities on normalized name/address,
token Jaccard, digit-token agreement, length ratios, same-country flag, candidate source."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from ber.data import Dataset
from ber.interfaces import FeatureSet
from ber.registry import register


def _threads() -> int:
    """Per-process thread budget (BER_THREADS, default 8) — never grab every core on a shared server."""
    return max(1, int(os.environ.get("BER_THREADS", "8")))


_SCORERS = {
    "ratio": fuzz.ratio,
    "partial": fuzz.partial_ratio,
    "tsort": fuzz.token_sort_ratio,
    "tset": fuzz.token_set_ratio,
    "jw": JaroWinkler.normalized_similarity,
}


def _jaccard(a: pd.Series, b: pd.Series) -> np.ndarray:
    out = np.zeros(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        if sx or sy:
            out[i] = len(sx & sy) / len(sx | sy)
    return out


@register("features", "string_sims")
class StringSims(FeatureSet):
    def __init__(self, fields: list[str] | None = None):
        self.fields = fields or ["name_norm", "name_core", "addr_norm"]

    def transform(self, ds: Dataset, pairs: pd.DataFrame) -> pd.DataFrame:
        cols = sorted(set(self.fields) | {"addr_digits", "country"})
        L = ds.lookup(pairs["s1_id"], cols)
        R = ds.lookup(pairs["cand_id"], cols)
        f = {}
        for fld in self.fields:
            a, b = L[fld].tolist(), R[fld].tolist()
            for sname, scorer in _SCORERS.items():
                f[f"f_str_{fld}_{sname}"] = process.cpdist(a, b, scorer=scorer, workers=_threads()).astype(np.float32)
            f[f"f_str_{fld}_jacc"] = _jaccard(L[fld], R[fld])
            la, lb = L[fld].str.len().to_numpy(), R[fld].str.len().to_numpy()
            f[f"f_str_{fld}_lenratio"] = (np.minimum(la, lb) / np.maximum(1, np.maximum(la, lb))).astype(np.float32)
            f[f"f_str_{fld}_empty"] = ((la == 0) | (lb == 0)).astype(np.float32)
        da, db = L["addr_digits"], R["addr_digits"]
        f["f_str_digits_jacc"] = _jaccard(da, db)
        f["f_str_digits_any_missing"] = ((da.str.len() == 0) | (db.str.len() == 0)).to_numpy().astype(np.float32)
        f["f_str_same_country"] = (L["country"].to_numpy() == R["country"].to_numpy()).astype(np.float32)
        f["f_str_cand_is_s3"] = pairs["cand_id"].str.startswith("S3-").to_numpy().astype(np.float32)
        return pd.DataFrame(f, index=pairs.index)
