"""Mined noise-word features (p1, T26): which differing words are source noise, learned WITHOUT labels inside each
group of equal `country` string — so the statistics exist for a country never seen in training (France).

Why: p1_noise learns per-word add/drop rates from train true pairs, keyed by word; every word of an unseen country
gets the prior. France names come from a small vocabulary (club, amicale, ecole, comite …: 56 words > 1% of S1 names,
INSIGHTS I-27) and France has 5-8x more same-address differently named candidates than US/India (I-28): a sibling
"DD Musique" at the address of "DD Comite" differs by a genuine word, while a copy differs by noise (SARL/EURL/SAS,
"(France)", "Holding"). Only per-country word statistics can tell the two apart.

transform (transductive, no labels): mine likely copies among the candidate pairs of the dataset being processed —
  same non-empty house-number set, name token_set_ratio >= name_min, number-free address token_set_ratio >= addr_min —
  then per group and word: add_rate = P(word only on the S2/S3 side | word in the S2/S3 name of a mined pair),
  drop_rate likewise for the S1 side, shrunk to the group's rate (pseudo-count, < min_support mined pairs → prior).
  Per candidate pair, over words on one side only: sum / max of (1 - rate) ("unexplained" = stable words that differ),
  their count, and the rank / margin of the total among the S1 entities competing for the same S2/S3 record.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz, process

from ber.data import Dataset
from ber.features.p1_noise import _field_matrix
from ber.interfaces import FeatureSet
from ber.registry import register

_DIG = re.compile(r"\d+")


@register("features", "p1_mnoise")
class MinedNoiseFeatures(FeatureSet):
    """Per-country noise-word rates mined from likely copies (label-free); see module docstring."""

    transductive = True

    def __init__(self, fields: list[str] | None = None, name_min: float = 85.0, addr_min: float = 75.0,
                 min_support: int = 20, pseudo: float = 10.0, chunk: int = 2_000_000, group_by: str = "country"):
        self.fields = fields or ["name_norm", "addr_norm"]
        self.name_min, self.addr_min = float(name_min), float(addr_min)
        self.min_support, self.pseudo, self.chunk, self.group_by = int(min_support), float(pseudo), int(chunk), group_by

    def _mine(self, recs: pd.DataFrame, li: np.ndarray, ri: np.ndarray) -> np.ndarray:
        """Indices of candidate pairs that are near-certain copies (label-free rules above)."""
        w = int(os.environ.get("BER_THREADS", "8"))
        addr = recs["addr_norm"].astype(str)
        num = addr.map(lambda s: "|".join(sorted(set(_DIG.findall(s)))))
        code, _ = pd.factorize(num)
        has = num.ne("").to_numpy()
        c = np.flatnonzero(has[li] & (code[li] == code[ri]))
        name = recs["name_norm"].astype(str).to_numpy(object)
        ns = process.cpdist(name[li[c]], name[ri[c]], scorer=fuzz.token_set_ratio, workers=w)
        c = c[ns >= self.name_min]
        street = addr.str.replace(r"\d+", " ", regex=True).to_numpy(object)
        st = process.cpdist(street[li[c]], street[ri[c]], scorer=fuzz.token_set_ratio, workers=w)
        return c[st >= self.addr_min]

    def _rates(self, B: sp.csr_matrix, L: np.ndarray, R: np.ndarray, grp_pair: np.ndarray, G: int) -> dict:
        """(G, V) add / drop rates from mined pairs (rows L = S1 side, R = S2/S3 side)."""
        BL, BR = B[L], B[R]
        both = BL.multiply(BR).tocsr()
        Gm = sp.csr_matrix((np.ones(len(L), np.float32), (grp_pair, np.arange(len(L)))), shape=(G, len(L)))
        in_r, in_l, bb = (np.asarray((Gm @ M).todense(), dtype=np.float32) for M in (BR, BL, both))
        out = {}
        for kind, only, tot in (("add", in_r - bb, in_r), ("drop", in_l - bb, in_l)):
            prior = only.sum(1, keepdims=True) / np.maximum(tot.sum(1, keepdims=True), 1)
            rate = (only + self.pseudo * prior) / (tot + self.pseudo)
            out[kind] = np.where(tot < self.min_support, prior, rate).astype(np.float32)
        return out

    def transform(self, ds: Dataset, pairs: pd.DataFrame) -> pd.DataFrame:
        recs = ds.records
        index = pd.Index(recs["entity_id"].to_numpy())
        li, ri = index.get_indexer(pairs["s1_id"]), index.get_indexer(pairs["cand_id"])
        codes = pd.factorize(recs[self.group_by].astype(str).to_numpy())[0]
        G = int(codes.max()) + 1
        m = self._mine(recs, li, ri)
        L, R = li[m], ri[m]
        out = {"f_p1m_mined_rec": np.zeros(len(pairs), np.float32)}
        mined_r = np.zeros(len(recs), bool); mined_r[R] = True
        out["f_p1m_mined_rec"] = mined_r[ri].astype(np.float32)          # record has a likely-copy partner
        total = np.zeros(len(pairs), np.float32)
        for field in self.fields:
            tag = "n" if field == "name_norm" else "a"
            B, _ = _field_matrix(recs, field)
            B = B.tocsr()
            rates = self._rates(B, L, R, codes[L], G)
            row = np.repeat(np.arange(B.shape[0]), np.diff(B.indptr))
            for kind, own, other in (("add", ri, li), ("drop", li, ri)):
                w = 1.0 - rates[kind][codes[row], B.indices]              # per-entry weight in the record's group
                Bw = sp.csr_matrix((w.astype(np.float32) + 1e-3, B.indices.copy(), B.indptr.copy()), shape=B.shape)
                s_sum = np.empty(len(pairs), np.float32)
                s_max = np.empty(len(pairs), np.float32)
                cnt = np.empty(len(pairs), np.float32)
                for s in range(0, len(pairs), self.chunk):
                    o, t = own[s:s + self.chunk], other[s:s + self.chunk]
                    Wo = Bw[o]
                    only = sp.csr_matrix(Wo - Wo.multiply(B[t]))
                    only.eliminate_zeros()
                    s_sum[s:s + len(o)] = np.asarray(only.sum(1)).ravel()
                    s_max[s:s + len(o)] = only.max(axis=1).toarray().ravel()
                    cnt[s:s + len(o)] = np.diff(only.indptr)
                side = "ro" if kind == "add" else "lo"
                out[f"f_p1m_{tag}_{side}_unexpl_sum"] = s_sum
                out[f"f_p1m_{tag}_{side}_unexpl_max"] = s_max
                out[f"f_p1m_{tag}_{side}_cnt"] = cnt
                total += s_sum
        order = np.lexsort((total, ri))
        g, v = ri[order], total[order]
        first = np.r_[True, g[1:] != g[:-1]]
        start = np.maximum.accumulate(np.where(first, np.arange(len(g)), 0))
        nxt = np.minimum(start + 1, len(v) - 1)
        second = np.where((g[nxt] == g) & (nxt != start), v[nxt], np.nan)
        other_best = np.where(np.arange(len(v)) == start, second, v[start])
        rank = np.empty(len(v), np.float32); rank[order] = np.arange(len(g)) - start + 1
        margin = np.empty(len(v), np.float32); margin[order] = other_best - v
        out["f_p1m_total_unexpl"] = total
        out["f_p1m_rec_rank"] = rank
        out["f_p1m_rec_margin"] = margin
        return pd.DataFrame({k: np.asarray(v).astype(np.float16) for k, v in out.items()}, index=pairs.index)
