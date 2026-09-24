"""Baseline blocker: character n-gram TF-IDF, top-k cosine neighbours of each S1 record
among all S2/S3 records (no country filter)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ber.data import Dataset
from ber.interfaces import Blocker
from ber.registry import register


def topk_cosine(A, B, k: int, min_sim: float = 0.0, budget: float = 2e7):
    """A (n_a x d), B (n_b x d) L2-normalised sparse. Returns (rows, cols, sims) of the top-k per row of A."""
    n_a, n_b = A.shape[0], B.shape[0]
    k = min(k, n_b)
    chunk = max(1, int(budget // max(1, n_b)))
    Bt = B.T.tocsc()
    rows, cols, sims = [], [], []
    for start in range(0, n_a, chunk):
        S = (A[start:start + chunk] @ Bt).toarray().astype(np.float32)
        idx = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
        val = np.take_along_axis(S, idx, axis=1)
        r = np.repeat(np.arange(start, start + S.shape[0]), k)
        keep = val.ravel() > min_sim
        rows.append(r[keep]); cols.append(idx.ravel()[keep]); sims.append(val.ravel()[keep])
    if not rows:
        return np.array([], int), np.array([], int), np.array([], np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(sims)


@register("blocker", "tfidf_knn")
class TfidfKnnBlocker(Blocker):
    transductive = True  # vocabulary/IDF fitted on the target dataset's own records

    def __init__(self, field: str = "name_norm", k: int = 20, ngram: tuple = (2, 4), analyzer: str = "char_wb",
                 min_sim: float = 0.05, fit_on: str = "target", max_df: float = 1.0):
        self.field, self.k, self.ngram, self.analyzer, self.min_sim, self.fit_on = field, k, tuple(ngram), analyzer, min_sim, fit_on
        self.max_df = max_df  # drop n-grams present in > max_df of records (speed; very common n-grams carry little signal)
        self.vec = None
        self.transductive = fit_on == "target"

    def _vectorizer(self):
        return TfidfVectorizer(analyzer=self.analyzer, ngram_range=self.ngram, sublinear_tf=True, min_df=1, max_df=self.max_df,
                               dtype=np.float32)

    def fit(self, train: Dataset):
        if self.fit_on == "train":
            self.vec = self._vectorizer().fit(train.records[self.field])
        return self

    def candidates(self, ds: Dataset) -> pd.DataFrame:
        s1, right = ds.s1, ds.right
        vec = self.vec if self.fit_on == "train" else self._vectorizer().fit(ds.records[self.field])
        A, B = vec.transform(s1[self.field]), vec.transform(right[self.field])
        r, c, s = topk_cosine(A, B, self.k, self.min_sim)
        return pd.DataFrame({"s1_id": s1["entity_id"].to_numpy()[r], "cand_id": right["entity_id"].to_numpy()[c],
                             "score": s.astype(float)})
