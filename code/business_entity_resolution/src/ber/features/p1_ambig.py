"""Name-ambiguity features (p1, T21): how identifiable is an S2/S3 record from its name alone?

46% of the remaining dense-CV loss sits on S1 entities with an empty-address S2/S3 record; for those records the name
is the only evidence, and 24% of them have a core name that is identical for their own S1 AND other S1 entities of the
same country (coin flips — under F0.5 better left unassigned). These features let the matcher see that directly.
Counts are computed per group of equal `country` string on the processed records (no labels → transductive):
  r_core_s1cnt / r_full_s1cnt   # S1 entities whose core name / full normalized name equals the record's
  s_core_s1cnt                  # S1 entities sharing the S1's own core name (siblings with identical names)
  s_core_rcnt                   # S2/S3 records whose core name equals the S1's core name (supply of copies)
  core_eq / full_eq             record's core / full name equals the S1's
  core_unique_owner             core_eq and this S1 is the only S1 with that core name
  r_addr_ntok / l_addr_ntok     address token counts (0 = name-only record)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ber.data import Dataset
from ber.interfaces import FeatureSet
from ber.registry import register


@register("features", "p1_ambig")
class NameAmbiguity(FeatureSet):
    """Name-ambiguity counts; see module docstring."""

    transductive = True

    def __init__(self, group_by: str = "country"):
        self.group_by = group_by

    def transform(self, ds: Dataset, pairs: pd.DataFrame) -> pd.DataFrame:
        recs = ds.records
        idx = pd.Index(recs["entity_id"].to_numpy())
        li, ri = idx.get_indexer(pairs["s1_id"]), idx.get_indexer(pairs["cand_id"])
        grp = recs[self.group_by].astype(str).to_numpy()
        is_s1 = (recs["source"] == 1).to_numpy()
        out = {}
        codes = {}
        for fld in ("name_core", "name_norm"):
            key, _ = pd.factorize(pd.Series(grp) + "\x1f" + recs[fld].astype(str).to_numpy())
            codes[fld] = key
        c_core, c_full = codes["name_core"], codes["name_norm"]
        s1_core = np.bincount(c_core[is_s1], minlength=c_core.max() + 1)
        s1_full = np.bincount(c_full[is_s1], minlength=c_full.max() + 1)
        r_core = np.bincount(c_core[~is_s1], minlength=c_core.max() + 1)
        core_eq = c_core[li] == c_core[ri]
        out["r_core_s1cnt"] = s1_core[c_core[ri]]
        out["r_full_s1cnt"] = s1_full[c_full[ri]]
        out["s_core_s1cnt"] = s1_core[c_core[li]]
        out["s_core_rcnt"] = r_core[c_core[li]]
        out["core_eq"] = core_eq
        out["full_eq"] = c_full[li] == c_full[ri]
        out["core_unique_owner"] = core_eq & (s1_core[c_core[ri]] == 1)
        ntok = recs["addr_norm"].astype(str).str.split().str.len().fillna(0).to_numpy()
        out["r_addr_ntok"] = ntok[ri]
        out["l_addr_ntok"] = ntok[li]
        cols = {}
        for k, v in out.items():
            v = np.asarray(v, dtype=np.float32)
            if k.endswith("cnt"):
                v = np.log1p(v)
            cols[f"f_p1a_{k}"] = v.astype(np.float16)
        return pd.DataFrame(cols, index=pairs.index)
