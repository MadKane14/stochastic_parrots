"""Learned noise-token features (p1, T17): which unmatched words are "explained" as source noise?

The generator adds words to copies ("Center", "Services", "Group", "[Inc]", "#8036") and drops words from them, and it
also creates sibling entities whose names differ from a real entity by exactly such a word. Only the statistics of
which words are noise can separate "Lucky Coffee" + noise "Center" from a sibling entity "Lucky Coffee Center" —
the largest remaining error class at density (exploration/p1/dense_analysis.md, T10 run: name-only S2/S3 records
lost to another S1 or below threshold).

fit(train): over all TRAIN true pairs (ground truth), per token of name_norm (and of address words):
    add_rate[t]  = P(t is only on the S2/S3 side | t occurs in the S2/S3 record of a true pair)
    drop_rate[t] = P(t is only on the S1 side    | t occurs in the S1 record of a true pair)
  shrunk to the global rate with a pseudo-count (tokens seen in < min_support true pairs get the prior), which also
  keeps a pair's own contribution to its token rates negligible in dense CV.
transform: per pair, over tokens only on the right (resp. left) side: sum and max of (1 - add_rate) (resp.
  (1 - drop_rate)) = "unexplained" additions / drops, their count, plus the rank and margin of the total unexplained
  mass among the S1 entities competing for the same S2/S3 record (who explains the record best).
Rates are keyed by token string, so they transfer to any dataset; unseen tokens get the prior.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ber.blocking.p1_compound import tokenize
from ber.data import Dataset
from ber.features.p1_match import _csr, _rowdot, _split_numbers
from ber.interfaces import FeatureSet
from ber.registry import register


def _field_matrix(recs: pd.DataFrame, field: str):
    """Binary token CSR of a field (address: words only, numbers removed) + vocab strings."""
    tok = tokenize(recs[field])
    B = _csr(tok, len(recs))
    if field == "addr_norm":
        B, _ = _split_numbers(B, tok[2])
    return B, np.asarray(tok[4].to_numpy(zero_copy_only=False), dtype=object)


@register("features", "p1_noise")
class NoiseTokenFeatures(FeatureSet):
    """Unexplained added / dropped tokens under learned noise rates; see module docstring."""

    def __init__(self, fields: list[str] | None = None, min_support: int = 30, pseudo: float = 20.0,
                 chunk: int = 2_000_000):
        self.fields = fields or ["name_norm", "addr_norm"]
        self.min_support, self.pseudo, self.chunk = int(min_support), float(pseudo), int(chunk)
        self.rates: dict = {}

    def fit(self, train: Dataset, pairs: pd.DataFrame | None = None):
        recs = train.records
        idx = pd.Index(recs["entity_id"].to_numpy())
        g_s = [s for s in train.gt for _ in train.gt[s]]
        g_c = [c for s in train.gt for c in train.gt[s]]
        L, R = idx.get_indexer(g_s), idx.get_indexer(g_c)
        ok = (L >= 0) & (R >= 0)
        L, R = L[ok], R[ok]
        for field in self.fields:
            B, vocab = _field_matrix(recs, field)
            BL, BR = B[L], B[R]
            both = np.asarray(BL.multiply(BR).sum(0)).ravel()
            in_r, in_l = np.asarray(BR.sum(0)).ravel(), np.asarray(BL.sum(0)).ravel()
            out = {}
            for name, only, tot in (("add", in_r - both, in_r), ("drop", in_l - both, in_l)):
                prior = only.sum() / max(tot.sum(), 1)
                rate = (only + self.pseudo * prior) / (tot + self.pseudo)
                rate[tot < self.min_support] = prior
                keep = tot > 0
                out[name] = (dict(zip(vocab[keep].tolist(), rate[keep].astype(np.float32).tolist())), float(prior))
            self.rates[field] = out
        return self

    def _weights(self, vocab: np.ndarray, field: str, kind: str) -> np.ndarray:
        table, prior = self.rates[field][kind]
        return (1.0 - pd.Series(vocab).map(table).fillna(prior).to_numpy(np.float32)).astype(np.float32)

    def transform(self, ds: Dataset, pairs: pd.DataFrame) -> pd.DataFrame:
        if not self.rates:
            raise RuntimeError("p1_noise: fit() must run on a labelled training set before transform()")
        recs = ds.records
        index = pd.Index(recs["entity_id"].to_numpy())
        li, ri = index.get_indexer(pairs["s1_id"]), index.get_indexer(pairs["cand_id"])
        out = {}
        total = np.zeros(len(pairs), np.float32)
        for field in self.fields:
            tag = "n" if field == "name_norm" else "a"
            B, vocab = _field_matrix(recs, field)
            for kind, own, other in (("add", ri, li), ("drop", li, ri)):
                w = self._weights(vocab, field, kind)
                Bw = B.multiply(w).tocsr().astype(np.float32)
                s_sum = np.empty(len(pairs), np.float32)
                s_max = np.empty(len(pairs), np.float32)
                cnt = np.empty(len(pairs), np.float32)
                for s in range(0, len(pairs), self.chunk):
                    o, t = own[s:s + self.chunk], other[s:s + self.chunk]
                    Wo = Bw[o]
                    only = Wo - Wo.multiply(B[t])            # own-side tokens absent on the other side, weighted
                    only = sp.csr_matrix(only)
                    only.eliminate_zeros()
                    s_sum[s:s + len(o)] = np.asarray(only.sum(1)).ravel()
                    s_max[s:s + len(o)] = only.max(axis=1).toarray().ravel()
                    cnt[s:s + len(o)] = np.diff(only.indptr)
                side = "ro" if kind == "add" else "lo"
                out[f"f_p1n_{tag}_{side}_unexpl_sum"] = s_sum
                out[f"f_p1n_{tag}_{side}_unexpl_max"] = s_max
                out[f"f_p1n_{tag}_{side}_cnt"] = cnt
                total += s_sum
        # who explains this S2/S3 record best among its competing S1 entities (lower unexplained mass = better)
        order = np.lexsort((total, ri))
        g, v = ri[order], total[order]
        first = np.r_[True, g[1:] != g[:-1]]
        start = np.maximum.accumulate(np.where(first, np.arange(len(g)), 0))
        nxt = np.minimum(start + 1, len(v) - 1)
        second = np.where((g[nxt] == g) & (nxt != start), v[nxt], np.nan)
        other_best = np.where(np.arange(len(v)) == start, second, v[start])
        rank = np.empty(len(v), np.float32); rank[order] = np.arange(len(g)) - start + 1
        margin = np.empty(len(v), np.float32); margin[order] = other_best - v    # > 0: this S1 explains it better
        out["f_p1n_total_unexpl"] = total
        out["f_p1n_rec_rank"] = rank
        out["f_p1n_rec_margin"] = margin
        return pd.DataFrame({k: v.astype(np.float16) for k, v in out.items()}, index=pairs.index)
