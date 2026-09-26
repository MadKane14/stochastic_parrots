"""Decision rule (p1, T03): one owner per S2/S3 record, then a score threshold, plus an optional best-guess rule.

  1. exclusivity: each S2/S3 record goes only to its best-scoring S1 entity (S1 is deduplicated; verified: no
     S2/S3 record is in two ground-truth lists)
  2. keep pairs with score >= threshold
  3. top1_min (optional): an S1 entity left empty keeps its single best remaining candidate if score >= top1_min
     (an empty prediction scores 0 on any non-singleton entity, so a confident-enough best guess pays off)
"""
from __future__ import annotations

import pandas as pd

from ber.interfaces import Decider
from ber.registry import register


@register("decider", "p1_assign")
class AssignDecider(Decider):
    def __init__(self, threshold: float = 0.5, top1_min: float | None = None, exclusive_right: bool = True):
        self.threshold, self.top1_min, self.exclusive_right = float(threshold), top1_min, bool(exclusive_right)

    def decide(self, ds, pairs: pd.DataFrame) -> pd.DataFrame:
        p = pairs[["s1_id", "cand_id", "score"]].sort_values("score", ascending=False, kind="stable")
        if self.exclusive_right:
            p = p.drop_duplicates("cand_id", keep="first")
        keep = p[p["score"] >= self.threshold]
        if self.top1_min is not None:
            rest = p[(p["score"] >= float(self.top1_min)) & ~p["s1_id"].isin(set(keep["s1_id"]))]
            keep = pd.concat([keep, rest.drop_duplicates("s1_id", keep="first")])
        return keep[["s1_id", "cand_id"]]
