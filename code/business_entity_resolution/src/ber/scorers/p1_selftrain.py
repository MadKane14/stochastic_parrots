"""Self-training LightGBM scorer (p1, T24): adapts the matcher to the records it scores — e.g. a country that never
appears in training (France) — using its own confident decisions as pseudo-labels. Transductive: it uses the
features of the pairs being scored (no labels, no external data, no hand labelling).

Why: at test density the model is as good on US / India test pairs as in cross-validation, but an unseen country's
pairs sit in other regions of feature space (other address formats, abbreviations, admin-area names); trained on
US / India only, the model's decision surface there is extrapolated. Confident pairs of the new data anchor it.

fit(train, pairs): exactly p1_lgbm; additionally keeps a fixed random sample of its training rows (fp16) and its
  validation rows, for the refit.
score(ds, pairs): p0 = base model scores; pseudo-positives = pairs with p0 >= hi that are the best-scoring S1 of
  their S2/S3 record (the one-owner rule); pseudo-negatives = pairs with p0 <= lo; at most max_pos positives and
  neg_per_pos negatives per positive (uniform samples); then a fresh model is trained on the kept train rows + the
  pseudo rows (weight pseudo_weight), early-stopped on the train validation rows, and its scores are returned.
adapt(): the same step for callers that score in slices (exploration/p1/dense_cv.py).
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from ber.data import Dataset
from ber.registry import register
from ber.scorers.p1_lgbm import P1LGBMScorer

log = logging.getLogger(__name__)


@register("scorer", "p1_selftrain")
class P1SelfTrainScorer(P1LGBMScorer):
    """p1_lgbm + one round of self-training on the scored pairs; see module docstring."""

    transductive = True

    def __init__(self, hi: float = 0.95, lo: float = 0.02, max_pos: int = 2_000_000, neg_per_pos: float = 1.5,
                 pseudo_weight: float = 1.0, keep_rows: int = 8_000_000, **kw):
        super().__init__(**kw)
        self.hi, self.lo, self.max_pos, self.neg_per_pos = float(hi), float(lo), int(max_pos), float(neg_per_pos)
        self.pseudo_weight, self.keep_rows = float(pseudo_weight), int(keep_rows)
        self.adapted = False

    def fit(self, train: Dataset, pairs: pd.DataFrame):
        import lightgbm as lgb
        from ber.data import unit_hash
        import re
        self.cols = [c for c in pairs.columns if (c.startswith("f_") or c.startswith("blk_"))
                     and not any(re.search(r, c) for r in self.drop_features)]
        s1 = pairs["s1_id"].to_numpy()
        u = pd.Series(pd.unique(s1))
        h = u.map(lambda s: unit_hash(s, 1000 + self.seed)).to_numpy()
        frac = min(1.0, self.max_rows / max(1, len(pairs)))
        in_keep = pairs["s1_id"].isin(set(u[h < frac])).to_numpy()
        in_valid = pairs["s1_id"].isin(set(u[h < frac * self.valid_frac])).to_numpy()
        tr, va = np.flatnonzero(in_keep & ~in_valid), np.flatnonzero(in_valid)
        y = pairs["label"].to_numpy()
        if len(np.unique(y[tr])) < 2:
            raise ValueError("training pairs contain a single class")
        self._yv = y[va].astype(np.float32)
        Xva = pairs.iloc[va].reindex(columns=self.cols).to_numpy(np.float32)
        dtr = lgb.Dataset(pairs.iloc[tr].reindex(columns=self.cols).to_numpy(np.float32), y[tr],
                          feature_name=self.cols, free_raw_data=True)
        dva = lgb.Dataset(Xva, self._yv, reference=dtr)
        self.model = lgb.train(self.params, dtr, self.num_boost_round, valid_sets=[dva],
                               callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)])
        self.best_iteration = self.model.best_iteration
        del dtr, dva
        self._Xv = Xva.astype(np.float16)                 # refit copies taken after training (memory peak)
        del Xva
        rng = np.random.default_rng(self.seed)
        kept = np.sort(rng.choice(tr, min(self.keep_rows, len(tr)), replace=False))
        self._Xk = pairs.iloc[kept].reindex(columns=self.cols).to_numpy(np.float16)
        self._yk = y[kept].astype(np.float32)
        log.info("p1_selftrain base: %d train rows, %d valid rows, kept %d for refit, best_iteration=%d",
                 len(tr), len(va), len(kept), self.best_iteration)
        self.adapted = False
        return self

    def base_score(self, pairs: pd.DataFrame) -> np.ndarray:
        """Scores with the current model (base before adapt(), refit after)."""
        return P1LGBMScorer.score(self, None, pairs)

    def adapt(self, p0: np.ndarray, rec_key: np.ndarray, take: Callable[[np.ndarray], pd.DataFrame]):
        """One self-training step: pseudo-label from p0 (aligned with rec_key = S2/S3 record id per pair), fetch the
        selected pairs' rows with take(idx), refit on kept train rows + pseudo rows."""
        import lightgbm as lgb
        order = np.lexsort((-p0, rec_key))
        own = order[np.r_[True, rec_key[order][1:] != rec_key[order][:-1]]]
        pos = own[p0[own] >= self.hi]
        neg = np.flatnonzero(p0 <= self.lo)
        rng = np.random.default_rng(self.seed + 1)
        if len(pos) > self.max_pos:
            pos = rng.choice(pos, self.max_pos, replace=False)
        n_neg = min(len(neg), int(self.neg_per_pos * len(pos)))
        neg = rng.choice(neg, n_neg, replace=False) if n_neg < len(neg) else neg
        idx = np.sort(np.r_[pos, neg])
        yp = np.isin(idx, pos).astype(np.float32)
        Xp = take(idx).reindex(columns=self.cols).to_numpy(np.float32)
        X = np.vstack([self._Xk.astype(np.float32), Xp])
        del Xp
        yy = np.r_[self._yk, yp]
        w = np.r_[np.ones(len(self._yk), np.float32), np.full(len(yp), self.pseudo_weight, np.float32)]
        dtr = lgb.Dataset(X, yy, weight=w, feature_name=self.cols, free_raw_data=True)
        dva = lgb.Dataset(self._Xv.astype(np.float32), self._yv, reference=dtr)
        self.model = lgb.train(self.params, dtr, self.num_boost_round, valid_sets=[dva],
                               callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)])
        del X, dtr
        self.best_iteration = self.model.best_iteration
        self.adapted = True
        self.pseudo_stats = {"pos": int(len(pos)), "neg": int(len(neg)), "scored": int(len(p0)),
                             "best_iteration": int(self.best_iteration)}
        log.info("p1_selftrain refit: %s", self.pseudo_stats)

    def score(self, ds: Dataset, pairs: pd.DataFrame, chunk: int = 2_000_000) -> np.ndarray:
        if len(pairs) == 0:
            return np.zeros(0)
        p0 = self.base_score(pairs)
        key = pairs["cand_id"].to_numpy() if "cand_id" in pairs.columns else pairs["ri"].to_numpy()
        self.adapt(p0, key, lambda idx: pairs.iloc[idx])
        return self.base_score(pairs)
