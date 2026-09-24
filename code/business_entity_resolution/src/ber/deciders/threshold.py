"""Baseline decision rule: keep pairs with score >= threshold.

Options:
  max_per_s1:     keep at most this many matches per Source 1 entity (highest scores)
  exclusive_right: an S2/S3 record is kept only for its best-scoring S1 entity
                   (Source 1 is deduplicated, so a record should belong to one S1 entity at most)
"""
from __future__ import annotations

import pandas as pd

from ber.interfaces import Decider
from ber.registry import register


@register("decider", "threshold")
class ThresholdDecider(Decider):
    def __init__(self, threshold: float = 0.5, max_per_s1: int | None = None, exclusive_right: bool = False):
        self.threshold, self.max_per_s1, self.exclusive_right = threshold, max_per_s1, exclusive_right

    def decide(self, ds, pairs: pd.DataFrame) -> pd.DataFrame:
        p = pairs.loc[pairs["score"] >= self.threshold, ["s1_id", "cand_id", "score"]]
        p = p.sort_values("score", ascending=False)
        if self.exclusive_right:
            p = p.drop_duplicates("cand_id", keep="first")
        if self.max_per_s1:
            p = p.groupby("s1_id").head(int(self.max_per_s1))
        return p[["s1_id", "cand_id"]]
