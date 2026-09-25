"""Gradient-boosted tree scorers over the pair feature columns (f_*, blk_*)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ber.data import Dataset
from ber.interfaces import Scorer, feature_columns
from ber.registry import register


class _TabularScorer(Scorer):
    def _make(self):
        raise NotImplementedError

    def fit(self, train: Dataset, pairs: pd.DataFrame):
        self.cols = feature_columns(pairs)
        y = pairs["label"].to_numpy()
        if len(np.unique(y)) < 2:
            raise ValueError("training pairs contain a single class — blocking found no positives (or no negatives)")
        self.model = self._make()
        self.model.fit(pairs[self.cols].astype(np.float32), y)
        return self

    def score(self, ds: Dataset, pairs: pd.DataFrame) -> np.ndarray:
        X = pairs.reindex(columns=self.cols).astype(np.float32)
        return self.model.predict_proba(X)[:, 1]


@register("scorer", "hgb")
class HGBScorer(_TabularScorer):
    """sklearn HistGradientBoostingClassifier (no extra dependency, handles NaN)."""

    def __init__(self, **params):
        self.params = {"max_iter": 300, "learning_rate": 0.08, "random_state": 0, **params}

    def _make(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(**self.params)


@register("scorer", "lgbm")
class LGBMScorer(_TabularScorer):
    def __init__(self, **params):
        self.params = {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63, "random_state": 0,
                       "verbose": -1, "n_jobs": int(os.environ.get("BER_THREADS", "8")), **params}

    def _make(self):
        import lightgbm as lgb
        return lgb.LGBMClassifier(**self.params)
