"""LightGBM pair scorer (p1, T03): fits on a sample of S1 groups (all candidates of a sampled S1 entity kept, so
the within-group features keep their test-time meaning), early-stops on held-out S1 groups, no class re-weighting
(`is_unbalance` inflates probabilities, and the decision rule needs calibrated-ish scores)."""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from ber.data import Dataset, unit_hash
from ber.interfaces import Scorer, feature_columns
from ber.registry import register

log = logging.getLogger(__name__)


@register("scorer", "p1_lgbm")
class P1LGBMScorer(Scorer):
    def __init__(self, max_rows: int = 20_000_000, valid_frac: float = 0.05, seed: int = 0,
                 num_boost_round: int = 2000, early_stopping: int = 50, **params):
        self.max_rows, self.valid_frac, self.seed = int(max_rows), float(valid_frac), int(seed)
        self.num_boost_round, self.early_stopping = int(num_boost_round), int(early_stopping)
        self.params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 100,
                       "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
                       "verbose": -1, "seed": seed, "num_threads": int(os.environ.get("BER_THREADS", "8")), **params}

    def fit(self, train: Dataset, pairs: pd.DataFrame):
        import lightgbm as lgb
        self.cols = feature_columns(pairs)
        s1 = pairs["s1_id"].to_numpy()
        u = pd.Series(pd.unique(s1))
        h = u.map(lambda s: unit_hash(s, 1000 + self.seed)).to_numpy()
        frac = min(1.0, self.max_rows / max(1, len(pairs)))
        keep_ids = set(u[h < frac])
        valid_ids = set(u[h < frac * self.valid_frac])
        in_keep = pairs["s1_id"].isin(keep_ids).to_numpy()
        in_valid = pairs["s1_id"].isin(valid_ids).to_numpy()
        tr, va = in_keep & ~in_valid, in_valid
        y = pairs["label"].to_numpy()
        if len(np.unique(y[tr])) < 2:
            raise ValueError("training pairs contain a single class")
        # select rows first, so only the sampled rows are ever copied
        dtr = lgb.Dataset(pairs.loc[tr, self.cols].to_numpy(np.float32), y[tr], feature_name=self.cols, free_raw_data=True)
        dva = lgb.Dataset(pairs.loc[va, self.cols].to_numpy(np.float32), y[va], reference=dtr)
        self.model = lgb.train(self.params, dtr, self.num_boost_round, valid_sets=[dva],
                               callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)])
        self.best_iteration = self.model.best_iteration
        log.info("p1_lgbm: %d train rows (%.0f%% of pairs), %d valid rows, best_iteration=%d",
                 tr.sum(), 100 * frac, va.sum(), self.best_iteration)
        return self

    def score(self, ds: Dataset, pairs: pd.DataFrame, chunk: int = 2_000_000) -> np.ndarray:
        out = np.empty(len(pairs), np.float64)
        for s in range(0, len(pairs), chunk):   # chunked: never copy the full feature matrix
            X = pairs.iloc[s:s + chunk].reindex(columns=self.cols).to_numpy(np.float32)
            out[s:s + len(X)] = self.model.predict(X, num_iteration=self.best_iteration)
        return np.clip(out, 0.0, 1.0)

    def importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importance("gain"), index=self.cols).sort_values(ascending=False)
