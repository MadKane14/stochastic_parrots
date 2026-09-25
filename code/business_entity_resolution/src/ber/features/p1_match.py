"""Matcher v1 features (p1, T03): memory-lean, chunked, vectorised — scales to ~75M candidate pairs.

Per pair (S1 record l, S2/S3 record r), all computed from integer row positions (no per-pair DataFrame lookups):
  string   rapidfuzz ratio / token_set / token_sort / partial / Jaro-Winkler on names, ratio on de-spaced names
           (web-style names), token_set + partial on addresses
  idf      IDF-weighted overlap of name tokens and of address words: shared IDF mass, IDF-Jaccard, IDF mass of
           tokens only on the left / only on the right (the "Acme Bakery" vs "Acme Robotics" signal), rarest shared
           token, shared-token count
  numbers  address numbers (digits split from letters, leading zeros stripped): shared count, Jaccard, first-number
           equality, first numbers equal after dropping one leading/trailing digit, conflict (both have numbers,
           none shared), missing on one side
  flags    empty address / non-ASCII name on either side, candidate is S3
  group    for key signals: rank and gap to the best value within the S1 entity's candidates AND within the
           S2/S3 record's candidate S1 entities (each S2/S3 record belongs to at most one S1), group sizes
IDF is computed on the records of the dataset being processed (no labels) → transductive, like the blocker.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from ber.blocking.p1_compound import tokenize
from ber.data import Dataset
from ber.interfaces import FeatureSet
from ber.registry import register


def _threads() -> int:
    return max(1, int(os.environ.get("BER_THREADS", "8")))


def _csr(tok, n):
    offsets, ids = tok[0], tok[1]
    m = sp.csr_matrix((np.ones(len(ids), np.float32), ids, offsets), shape=(n, int(ids.max()) + 1 if len(ids) else 1))
    m.sum_duplicates()
    m.data[:] = 1.0
    return m


def _split_numbers(B: sp.csr_matrix, is_num: np.ndarray):
    """Split a binary token matrix into (words, numbers) by token id."""
    num_cols = np.zeros(B.shape[1], bool)
    num_cols[: len(is_num)] = is_num
    mask = num_cols[B.indices]
    W = B.copy(); W.data = np.where(mask, 0, W.data).astype(np.float32); W.eliminate_zeros()
    N = B.copy(); N.data = np.where(mask, N.data, 0).astype(np.float32); N.eliminate_zeros()
    return W, N


def _first_number(tok, vocab_is_num, vocab_int):
    """Integer value of each record's first address number (-1 if none; capped at 9 digits)."""
    offsets, ids = tok[0], tok[1]
    n = len(offsets) - 1
    out = np.full(n, -1, np.int64)
    num_pos = np.flatnonzero(vocab_is_num[ids])
    if len(num_pos):
        row = np.searchsorted(offsets, num_pos, side="right") - 1
        first = np.r_[True, row[1:] != row[:-1]]
        out[row[first]] = vocab_int[ids[num_pos[first]]]
    return out


def _digits_len(x):
    return np.where(x > 0, np.floor(np.log10(np.maximum(x, 1))).astype(np.int64) + 1, 1)


class _Tables:
    """Per-dataset token matrices / IDF, built once per Dataset."""

    def __init__(self, recs: pd.DataFrame):
        n = len(recs)
        self.index = pd.Index(recs["entity_id"].to_numpy())
        ntok = tokenize(recs["name_norm"])
        atok = tokenize(recs["addr_norm"])
        Bn = _csr(ntok, n)
        Ba = _csr(atok, n)
        Aw, An = _split_numbers(Ba, atok[2])
        self.Bn, self.Bw, self.Bnum = Bn, Aw, An
        self.idf_n = self._idf(Bn)
        self.idf_w = self._idf(Aw)
        self.Wn = Bn.multiply(self.idf_n).tocsr().astype(np.float32)
        self.Ww = Aw.multiply(self.idf_w).tocsr().astype(np.float32)
        self.mass_n = np.asarray(self.Wn.sum(1)).ravel()
        self.mass_w = np.asarray(self.Ww.sum(1)).ravel()
        self.cnt_num = np.asarray(An.sum(1)).ravel()
        # first address number as an integer (numbers longer than 9 digits keep their last 9)
        is_num, vocab = atok[2], atok[4].to_numpy(zero_copy_only=False)
        vocab_int = np.full(len(is_num), -1, np.int64)
        idx = np.flatnonzero(is_num)
        vocab_int[idx] = pd.to_numeric(pd.Series(vocab[idx]).str[-9:], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        self.first_num = _first_number(atok, is_num, vocab_int)
        self.name = recs["name_norm"].to_numpy(dtype=object)
        self.name_ns = recs["name_norm"].str.replace(" ", "", regex=False).to_numpy(dtype=object)
        self.addr = recs["addr_norm"].to_numpy(dtype=object)
        self.addr_empty = (recs["addr_norm"].str.len() == 0).to_numpy()
        self.name_nonascii = ~recs["name_norm"].map(str.isascii).to_numpy(dtype=bool)
        self.is_s3 = recs["entity_id"].str.startswith("S3-").to_numpy()

    @staticmethod
    def _idf(B):
        df = np.asarray(B.sum(0)).ravel()
        return np.where(df > 0, np.log((B.shape[0] + 1) / (df + 1)) + 1.0, 0.0).astype(np.float32)


def _rowdot(A: sp.csr_matrix, B: sp.csr_matrix, li, ri, how="sum"):
    M = A[li].multiply(B[ri]).tocsr()
    if how == "sum":
        return np.asarray(M.sum(1)).ravel()
    return M.max(axis=1).toarray().ravel()


@register("features", "p1_match")
class MatchFeatures(FeatureSet):
    """Matcher v1 features; see module docstring."""

    transductive = True   # IDF from the processed dataset's own records (no labels)

    def __init__(self, chunk: int = 2_000_000, group_features: bool = True):
        self.chunk, self.group_features = int(chunk), bool(group_features)
        self._tables_key, self._tables = None, None

    def _tables_for(self, ds: Dataset) -> _Tables:
        key = (id(ds.records), len(ds.records))
        if self._tables_key != key:
            self._tables, self._tables_key = _Tables(ds.records), key
        return self._tables

    def _chunk_features(self, T: _Tables, li: np.ndarray, ri: np.ndarray) -> dict:
        w = _threads()
        f = {}
        a, b = T.name[li].tolist(), T.name[ri].tolist()
        f["name_ratio"] = process.cpdist(a, b, scorer=fuzz.ratio, workers=w)
        f["name_tset"] = process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=w)
        f["name_tsort"] = process.cpdist(a, b, scorer=fuzz.token_sort_ratio, workers=w)
        f["name_partial"] = process.cpdist(a, b, scorer=fuzz.partial_ratio, workers=w)
        f["name_jw"] = process.cpdist(a, b, scorer=JaroWinkler.normalized_similarity, workers=w)
        a2, b2 = T.name_ns[li].tolist(), T.name_ns[ri].tolist()
        f["name_ns_ratio"] = process.cpdist(a2, b2, scorer=fuzz.ratio, workers=w)
        f["name_ns_partial"] = process.cpdist(a2, b2, scorer=fuzz.partial_ratio, workers=w)
        a3, b3 = T.addr[li].tolist(), T.addr[ri].tolist()
        f["addr_tset"] = process.cpdist(a3, b3, scorer=fuzz.token_set_ratio, workers=w)
        f["addr_partial"] = process.cpdist(a3, b3, scorer=fuzz.partial_ratio, workers=w)
        f["addr_ratio"] = process.cpdist(a3, b3, scorer=fuzz.ratio, workers=w)
        # IDF overlaps
        for tag, W, B, mass in (("n", T.Wn, T.Bn, T.mass_n), ("w", T.Ww, T.Bw, T.mass_w)):
            shared = _rowdot(W, B, li, ri)
            ml, mr = mass[li], mass[ri]
            f[f"idf_{tag}_shared"] = shared
            f[f"idf_{tag}_jacc"] = shared / np.maximum(ml + mr - shared, 1e-6)
            f[f"idf_{tag}_left_only"] = ml - shared
            f[f"idf_{tag}_right_only"] = mr - shared
            f[f"idf_{tag}_max_shared"] = _rowdot(W, B, li, ri, how="max")
            f[f"idf_{tag}_cnt_shared"] = _rowdot(B, B, li, ri)
        # numbers
        cs = _rowdot(T.Bnum, T.Bnum, li, ri)
        nl, nr = T.cnt_num[li], T.cnt_num[ri]
        f["num_shared"] = cs
        f["num_jacc"] = cs / np.maximum(nl + nr - cs, 1)
        f["num_conflict"] = ((nl > 0) & (nr > 0) & (cs == 0)).astype(np.float32)
        f["num_missing_one"] = ((nl > 0) != (nr > 0)).astype(np.float32)
        x, y = T.first_num[li], T.first_num[ri]
        both = (x >= 0) & (y >= 0)
        f["num_first_eq"] = np.where(both, (x == y), np.nan).astype(np.float32)
        lx, ly = _digits_len(x), _digits_len(y)
        near = ((x // 10 == y) | (y // 10 == x)                                  # trailing digit dropped
                | ((lx > ly) & (x % (10 ** np.minimum(ly, 18)) == y))              # leading digits dropped
                | ((ly > lx) & (y % (10 ** np.minimum(lx, 18)) == x)))
        f["num_first_near"] = np.where(both, near & (x != y), np.nan).astype(np.float32)
        # flags
        f["addr_empty_l"] = T.addr_empty[li].astype(np.float32)
        f["addr_empty_r"] = T.addr_empty[ri].astype(np.float32)
        f["name_nonascii_r"] = T.name_nonascii[ri].astype(np.float32)
        f["cand_is_s3"] = T.is_s3[ri].astype(np.float32)
        return {k: np.asarray(v, dtype=np.float32) for k, v in f.items()}

    def transform(self, ds: Dataset, pairs: pd.DataFrame) -> pd.DataFrame:
        T = self._tables_for(ds)
        li_all = T.index.get_indexer(pairs["s1_id"])
        ri_all = T.index.get_indexer(pairs["cand_id"])
        parts = []
        for s in range(0, len(pairs), self.chunk):
            parts.append(self._chunk_features(T, li_all[s:s + self.chunk], ri_all[s:s + self.chunk]))
        cols = {k: np.concatenate([p[k] for p in parts]) if parts else np.zeros(0, np.float32) for k in
                (parts[0].keys() if parts else [])}
        out = pd.DataFrame({f"f_p1_{k}": v for k, v in cols.items()}, index=pairs.index)
        if self.group_features and len(pairs):
            out = pd.concat([out, self._group(out, li_all, ri_all)], axis=1)
        return out

    @staticmethod
    def _group(f: pd.DataFrame, li: np.ndarray, ri: np.ndarray) -> pd.DataFrame:
        """Rank / gap-to-best of key signals within each S1 entity and within each S2/S3 record."""
        key = f["f_p1_name_tset"].to_numpy() / 100 + f["f_p1_addr_tset"].to_numpy() / 100 \
            + f["f_p1_idf_n_jacc"].to_numpy() + f["f_p1_idf_w_jacc"].to_numpy() + np.minimum(f["f_p1_num_shared"].to_numpy(), 2)
        g = pd.DataFrame({"l": li, "r": ri, "key": key,
                          "name": f["f_p1_name_tset"].to_numpy(), "addr": f["f_p1_addr_tset"].to_numpy()})
        out = {}
        for side, col in (("s1", "l"), ("rec", "r")):
            grp = g.groupby(col, sort=False)
            for sig in ("key", "name", "addr"):
                mx = grp[sig].transform("max").to_numpy()
                out[f"f_p1_grp_{side}_{sig}_gap"] = (mx - g[sig].to_numpy()).astype(np.float32)
                out[f"f_p1_grp_{side}_{sig}_rank"] = grp[sig].rank(ascending=False, method="min").to_numpy(np.float32)
            out[f"f_p1_grp_{side}_size"] = grp["key"].transform("size").to_numpy(np.float32)
        # margin of this pair's key over the best OTHER candidate in its S1 group (second-best if it is the best)
        for side, col in (("s1", "l"), ("rec", "r")):
            k = g["key"].to_numpy()
            order = np.lexsort((-k, g[col].to_numpy()))
            gc = g[col].to_numpy()[order]
            ks = k[order]
            first = np.r_[True, gc[1:] != gc[:-1]]
            start = np.maximum.accumulate(np.where(first, np.arange(len(gc)), 0))
            best = ks[start]
            nxt_idx = np.minimum(start + 1, len(ks) - 1)
            has2 = (nxt_idx < len(ks)) & (gc[nxt_idx] == gc) & (nxt_idx != start)
            second = np.where(has2, ks[nxt_idx], np.nan)
            other_best = np.where(np.arange(len(ks)) == start, second, best)
            margin = np.empty(len(k), np.float32)
            margin[order] = ks - other_best
            out[f"f_p1_grp_{side}_key_margin"] = margin
        return pd.DataFrame(out, index=f.index)
