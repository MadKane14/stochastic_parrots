"""Matcher v2 extra features (p1, T07), aimed at the dense-CV errors (exploration/p1/dense_analysis.md):

  skel     script-agnostic name/address comparison: consonant skeletons of the (romanized) fields
           (p1_romanize) → rapidfuzz ratio / token_set, IDF-weighted skeleton-token overlap, left/right-only mass.
           Separates "रेड कंसल्टेंट्स" (= Red Consultants) from a different company at the same address.
  numdist  unmatched distractors are near-copies with a house number shifted by a few units (229 → 232), while
           true-pair noise is mostly a one-digit typo or a dropped digit: Levenshtein distance and log absolute
           difference of the first address numbers, best (min) distance over the first 3 numbers on each side,
           token_set of the number strings, share of the shorter number list found in the longer one.
  replace  name tokens only on the left / only on the right (counts) and a "one token replaced" flag.
IDF from the processed dataset's own records (no labels) → transductive, like p1_match.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from ber.blocking.p1_compound import tokenize
from ber.data import Dataset
from ber.features.p1_match import _csr, _rowdot, group_idf
from ber.features.p1_romanize import name_skeleton
from ber.interfaces import FeatureSet
from ber.registry import register

N_NUM = 3


def _threads() -> int:
    return max(1, int(os.environ.get("BER_THREADS", "8")))


def _idf(B):
    df = np.asarray(B.sum(0)).ravel()
    return np.where(df > 0, np.log((B.shape[0] + 1) / (df + 1)) + 1.0, 0.0).astype(np.float32)


class _ExtraTables:
    def __init__(self, recs: pd.DataFrame, idf_group: str | None = None):
        n = len(recs)
        codes = pd.factorize(recs[idf_group].to_numpy())[0] if idf_group else None
        self.index = pd.Index(recs["entity_id"].to_numpy())
        # skeletons (romanize only non-ASCII values; cache per distinct string)
        self.skel_n = recs["name_norm"].map(name_skeleton).to_numpy(dtype=object)
        self.skel_a = recs["addr_norm"].map(name_skeleton).to_numpy(dtype=object)
        self.tables = {}
        for tag, vals in (("n", self.skel_n), ("a", self.skel_a)):
            B = _csr(tokenize(pd.Series(vals)), n)
            if codes is not None:
                W = group_idf(B, codes)[0]
            else:
                W = B.multiply(_idf(B)).tocsr().astype(np.float32)
            self.tables[tag] = (B, W, np.asarray(W.sum(1)).ravel())
        # plain name tokens for replacement counts
        Bn = _csr(tokenize(recs["name_norm"]), n)
        self.Bn, self.cnt_n = Bn, np.asarray(Bn.sum(1)).ravel()
        # address number strings: first N_NUM number tokens per record (zero-stripped by tokenize)
        off, ids, is_num, _, vocab = tokenize(recs["addr_norm"])
        vocab = vocab.to_numpy(zero_copy_only=False)
        num_pos = np.flatnonzero(is_num[ids])
        row = np.searchsorted(off, num_pos, side="right") - 1
        first = np.r_[True, row[1:] != row[:-1]] if len(row) else np.zeros(0, bool)
        start = np.maximum.accumulate(np.where(first, np.arange(len(row)), 0)) if len(row) else row
        rank = np.arange(len(row)) - start
        self.nums = np.full((n, N_NUM), "", dtype=object)
        sel = rank < N_NUM
        self.nums[row[sel], rank[sel]] = vocab[ids[num_pos[sel]]]
        allnum = pd.Series(vocab[ids[num_pos]]).groupby(row).agg(" ".join)
        self.numstr = np.full(n, "", dtype=object)
        self.numstr[allnum.index.to_numpy()] = allnum.to_numpy()
        self.n_num = np.bincount(row, minlength=n)
        v = pd.to_numeric(pd.Series(self.nums[:, 0]).str[-12:], errors="coerce").to_numpy()
        self.first_val = np.where(np.isnan(v), -1.0, v)


@register("features", "p1_extra")
class ExtraFeatures(FeatureSet):
    """Matcher v2 extra features; see module docstring."""

    transductive = True

    def __init__(self, chunk: int = 2_000_000, dtype: str = "float16", idf_group: str | None = None):
        self.chunk, self.dtype, self.idf_group = int(chunk), np.dtype(dtype), idf_group
        self._key, self._t = None, None

    def _tables(self, ds: Dataset) -> _ExtraTables:
        key = (id(ds.records), len(ds.records))
        if self._key != key:
            self._t, self._key = _ExtraTables(ds.records, self.idf_group), key
        return self._t

    def _chunk(self, T: _ExtraTables, li, ri) -> dict:
        w = _threads()
        f = {}
        a, b = T.skel_n[li].tolist(), T.skel_n[ri].tolist()
        f["skel_n_ratio"] = process.cpdist(a, b, scorer=fuzz.ratio, workers=w)
        f["skel_n_tset"] = process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=w)
        a2, b2 = T.skel_a[li].tolist(), T.skel_a[ri].tolist()
        f["skel_a_tset"] = process.cpdist(a2, b2, scorer=fuzz.token_set_ratio, workers=w)
        for tag in ("n", "a"):
            B, W, mass = T.tables[tag]
            shared = _rowdot(W, B, li, ri)
            ml, mr = mass[li], mass[ri]
            f[f"skel_{tag}_idf_jacc"] = shared / np.maximum(ml + mr - shared, 1e-6)
            f[f"skel_{tag}_idf_left_only"] = ml - shared
            f[f"skel_{tag}_idf_right_only"] = mr - shared
        # numbers
        x0, y0 = T.nums[li, 0].tolist(), T.nums[ri, 0].tolist()
        both = (T.n_num[li] > 0) & (T.n_num[ri] > 0)
        f["num1_lev"] = np.where(both, process.cpdist(x0, y0, scorer=Levenshtein.distance, workers=w), np.nan)
        xv, yv = T.first_val[li], T.first_val[ri]
        f["num1_logdiff"] = np.where((xv >= 0) & (yv >= 0), np.log1p(np.abs(xv - yv)), np.nan)
        best = np.full(len(li), 99.0)
        for i in range(N_NUM):
            xi = T.nums[li, i].tolist()
            for j in range(N_NUM):
                yj = T.nums[ri, j].tolist()
                d = process.cpdist(xi, yj, scorer=Levenshtein.distance, workers=w).astype(np.float64)
                ok = (T.nums[li, i] != "") & (T.nums[ri, j] != "")
                best = np.where(ok, np.minimum(best, d), best)
        f["num_best_lev"] = np.where(both, best, np.nan)
        f["num_tset"] = np.where(both, process.cpdist(T.numstr[li].tolist(), T.numstr[ri].tolist(),
                                                      scorer=fuzz.token_set_ratio, workers=w), np.nan)
        # name token replacement
        shared_n = _rowdot(T.Bn, T.Bn, li, ri)
        lo, ro = T.cnt_n[li] - shared_n, T.cnt_n[ri] - shared_n
        f["name_left_only_cnt"] = lo
        f["name_right_only_cnt"] = ro
        f["name_one_replaced"] = ((lo == 1) & (ro == 1)).astype(np.float32)
        return {k: np.asarray(v, dtype=np.float32) for k, v in f.items()}

    def transform(self, ds: Dataset, pairs: pd.DataFrame) -> pd.DataFrame:
        T = self._tables(ds)
        li = T.index.get_indexer(pairs["s1_id"])
        ri = T.index.get_indexer(pairs["cand_id"])
        arr, names = None, None
        for s in range(0, len(pairs), self.chunk):
            f = self._chunk(T, li[s:s + self.chunk], ri[s:s + self.chunk])
            if arr is None:
                names = [f"f_p1x_{k}" for k in f]
                arr = np.empty((len(pairs), len(names)), dtype=self.dtype)
            for j, v in enumerate(f.values()):
                arr[s:s + len(v), j] = np.minimum(v, 6e4)   # float16-safe
        self._t, self._key = None, None   # free the token tables (rebuilt per dataset anyway)
        if arr is None:
            return pd.DataFrame(index=pairs.index)
        return pd.DataFrame(arr, index=pairs.index, columns=names, copy=False)
