"""Stage interfaces — THE contract between independent work directions.

Pipeline:  Dataset --normalize--> Dataset(+norm cols) --block--> pairs
           --features--> pairs(+f_* cols) --score--> pairs(+score) --decide--> matches

Tables (pandas DataFrames) passed between stages:

  records   one row per record of ALL sources. Required columns:
              entity_id (str, 'S1-…'/'S2-…'/'S3-…'), source (int 1/2/3),
              business_name, business_address, country (str; open set, never hard-code values)
            Normalizers ADD columns (must include `name_norm`, `addr_norm`) and never drop rows.

  pairs     one row per (Source 1 entity, candidate) pair:
              s1_id (str, S1-…), cand_id (str, S2-…/S3-…), unique per (s1_id, cand_id).
            The pipeline adds blocker provenance columns `blk_<i>_score`, `blk_<i>_rank`
            (NaN where blocker i did not propose the pair), `label` (train only),
            features add `f_<featureset>_<name>` columns, the scorer adds `score` in [0, 1].

  matches   subset of pairs: columns s1_id, cand_id. Written to matching_results.tsv.

Rules for implementations:
  * Stateless w.r.t. which split is being predicted: fit() only sees the training Dataset.
    Unsupervised statistics may be computed on the target Dataset's records inside
    candidates()/transform() (e.g. TF-IDF vocab) — flag it with `transductive = True`.
  * Never use `ds.gt` outside fit().
  * Never filter or special-case country values. Country is a string feature at most.
  * Put new implementations in a NEW file named `<handle>_<desc>.py` and register them
    with a name prefixed by your handle, e.g. @register("blocker", "p3_addr_tfidf").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ber.data import Dataset


class Component:
    spec: dict = {}
    registry_name: str = ""
    transductive: bool = False  # True if it fits unsupervised stats on the target dataset


class Normalizer(Component):
    def fit(self, train: Dataset) -> "Normalizer":
        return self

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `records` with extra columns (must add name_norm, addr_norm)."""
        raise NotImplementedError


class Blocker(Component):
    def fit(self, train: Dataset) -> "Blocker":
        return self

    def candidates(self, ds: Dataset) -> pd.DataFrame:
        """Return DataFrame[s1_id, cand_id, score] — score: higher = more similar.
        Only S1 ids on the left, only S2/S3 ids on the right, no duplicate pairs."""
        raise NotImplementedError


class FeatureSet(Component):
    def fit(self, train: Dataset, pairs: pd.DataFrame) -> "FeatureSet":
        return self

    def transform(self, ds: Dataset, pairs: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with the SAME index as `pairs`, columns named f_<name>_<feature>."""
        raise NotImplementedError


class Scorer(Component):
    def fit(self, train: Dataset, pairs: pd.DataFrame) -> "Scorer":
        """`pairs` has s1_id, cand_id, label, blk_* and f_* columns."""
        raise NotImplementedError

    def score(self, ds: Dataset, pairs: pd.DataFrame) -> np.ndarray:
        """Return match probabilities in [0, 1], aligned with `pairs` rows."""
        raise NotImplementedError


class Decider(Component):
    def fit(self, train: Dataset, pairs: pd.DataFrame) -> "Decider":
        return self

    def decide(self, ds: Dataset, pairs: pd.DataFrame) -> pd.DataFrame:
        """`pairs` has a `score` column. Return DataFrame[s1_id, cand_id] (subset of pairs)."""
        raise NotImplementedError


def feature_columns(pairs: pd.DataFrame) -> list[str]:
    """Columns a scorer may use as features."""
    return [c for c in pairs.columns if c.startswith("f_") or c.startswith("blk_")]
