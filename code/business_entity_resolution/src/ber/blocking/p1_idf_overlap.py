"""Scalable blocker: IDF-weighted token overlap (cosine) within groups of equal `country` string.

For every group of records sharing the exact country string (generic grouping over whatever values
appear — EDA: 100% of true pairs share the string), each record is a binary bag of whitespace tokens
of `fields` (e.g. name_norm, addr_norm, or both). Tokens are IDF-weighted inside the group and rows
L2-normalised, so the score is an IDF cosine. Tokens with in-group document frequency > `max_df`
are left out of the inverted index (they still count in the norms): this keeps the sparse product
S1 x right near-linear. Top-k right records per S1 record are kept.

Vectorised: pyarrow tokenisation, scipy CSR products in chunks, top-k by lexsort on the CSR result.
Chunks run in forked worker processes (count = BER_THREADS).
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import pyarrow as pa  # noqa: F401  (pc needs pa loaded)
import pyarrow.compute as pc
import scipy.sparse as sp

from ber.data import Dataset
from ber.interfaces import Blocker
from ber.registry import register

_TOKEN_CACHE: dict = {}   # (id(records), len, field) -> (csr, vocab_size); shared by blocker instances


def token_matrix(values: pd.Series) -> sp.csr_matrix:
    """Binary CSR (n_rows x vocab) of the whitespace tokens of `values` (vocab ids are arbitrary)."""
    arr = pa.array(values.to_numpy(dtype=object), type=pa.large_string())
    lists = pc.utf8_split_whitespace(arr)
    offsets = lists.offsets.to_numpy().astype(np.int64)
    enc = pc.dictionary_encode(lists.flatten())
    idx = enc.indices.to_numpy().astype(np.int32)
    empty = np.flatnonzero(enc.dictionary.to_numpy(zero_copy_only=False) == "")
    m = sp.csr_matrix((np.ones(len(idx), np.float32), idx, offsets), shape=(len(values), len(enc.dictionary)))
    m.sum_duplicates()
    m.data[:] = 1.0
    if len(empty):
        m.data[np.isin(m.indices, empty)] = 0.0
        m.eliminate_zeros()
    return m


def _tokens(records: pd.DataFrame, fields: tuple[str, ...]) -> sp.csr_matrix:
    mats = []
    for f in fields:
        key = (id(records), len(records), f)
        if key not in _TOKEN_CACHE:
            _TOKEN_CACHE.clear() if len(_TOKEN_CACHE) > 8 else None
            _TOKEN_CACHE[key] = token_matrix(records[f])
        mats.append(_TOKEN_CACHE[key])
    return mats[0] if len(mats) == 1 else sp.hstack(mats, format="csr")   # separate namespaces per field


def topk_rows(C: sp.csr_matrix, k: int, min_score: float):
    """Top-k columns per row of a CSR matrix -> (row, col, value) arrays."""
    C = C.tocsr()
    n_per_row = np.diff(C.indptr)
    rows = np.repeat(np.arange(C.shape[0], dtype=np.int64), n_per_row)
    order = np.lexsort((-C.data, rows))
    rank = np.arange(len(order)) - np.repeat(C.indptr[:-1], n_per_row)
    keep = order[(rank < k)]
    keep = keep[C.data[keep] >= min_score]
    return rows[keep], C.indices[keep].astype(np.int64), C.data[keep]


# worker globals (set before fork so children share memory copy-on-write)
_W: dict = {}


def _chunk(bounds):
    a, b = bounds
    A, BT, k, min_score = _W["A"], _W["BT"], _W["k"], _W["min_score"]
    r, c, v = topk_rows(A[a:b] @ BT, k, min_score)
    return r + a, c, v


def _group_topk(X1: sp.csr_matrix, X2: sp.csr_matrix, k: int, max_df: int, min_score: float,
                hit_budget: float, workers: int):
    """IDF cosine top-k of every row of X1 among rows of X2 (both binary CSR, same vocab).
    `max_df` may be a scalar or a per-column array (columns with df > max_df are not indexed)."""
    df = np.asarray(X1.sum(0)).ravel() + np.asarray(X2.sum(0)).ravel()
    n = X1.shape[0] + X2.shape[0]
    idf = np.where(df > 0, np.log((n + 1) / (df + 1)) + 1.0, 0.0).astype(np.float32)

    def weigh(X):
        W = X.multiply(idf).tocsr().astype(np.float32)
        norm = np.sqrt(np.asarray(W.multiply(W).sum(1)).ravel())
        return sp.diags(1.0 / np.maximum(norm, 1e-9)).astype(np.float32) @ W

    keep = sp.diags(((df > 0) & (df <= max_df)).astype(np.float32))
    A = (weigh(X1) @ keep).tocsr()
    A.eliminate_zeros()
    BT = (weigh(X2) @ keep).T.tocsr()
    BT.eliminate_zeros()
    # adaptive chunks: each holds ~hit_budget (S1 token, posting) hits = upper bound of the product's nnz
    df2 = np.asarray(BT.getnnz(axis=1)).ravel().astype(np.float64)
    hits = np.cumsum(A.copy().astype(bool).astype(np.float64) @ df2)
    cuts = np.searchsorted(hits, np.arange(hit_budget, hits[-1] if len(hits) else 0, hit_budget))
    edges = np.unique(np.concatenate([[0], cuts, [A.shape[0]]])).astype(int)
    bounds = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    _W.update(A=A, BT=BT, k=k, min_score=min_score)
    try:
        if workers > 1 and len(bounds) > 1:
            with mp.get_context("fork").Pool(min(workers, len(bounds))) as pool:
                parts = pool.map(_chunk, bounds, chunksize=1)
        else:
            parts = [_chunk(bd) for bd in bounds]
    finally:
        _W.clear()
    if not parts:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    return tuple(np.concatenate(x) for x in zip(*parts))


@register("blocker", "p1_idf_overlap")
class IdfOverlapBlocker(Blocker):
    """Top-k right records per S1 by IDF-weighted token cosine, within equal-country-string groups."""

    transductive = True   # IDF / df cap computed on the records being blocked (no labels used)

    def __init__(self, fields: list[str] | str = "name_norm", k: int = 20, max_df: int = 20000,
                 min_score: float = 0.0, hit_budget: float = 1e7, group_by: str | None = "country"):
        self.fields = (fields,) if isinstance(fields, str) else tuple(fields)
        self.k, self.max_df, self.min_score = int(k), int(max_df), float(min_score)
        self.hit_budget, self.group_by = float(hit_budget), group_by

    def candidate_rows(self, recs: pd.DataFrame):
        """(s1_row, right_row, score) arrays of positional row indices into `recs`."""
        X = _tokens(recs, self.fields)
        is_s1 = (recs["source"] == 1).to_numpy()
        groups = recs[self.group_by].to_numpy() if self.group_by else np.zeros(len(recs), dtype=np.int8)
        codes, _ = pd.factorize(groups)
        workers = max(1, int(os.environ.get("BER_THREADS", "8")))
        R, C, V = [], [], []
        for g in np.unique(codes):
            in_g = codes == g
            i1, i2 = np.flatnonzero(in_g & is_s1), np.flatnonzero(in_g & ~is_s1)
            if len(i1) == 0 or len(i2) == 0:
                continue
            r, c, v = _group_topk(X[i1], X[i2], self.k, self.max_df, self.min_score, self.hit_budget, workers)
            R.append(i1[r]); C.append(i2[c]); V.append(v.astype(np.float32))
        if not R:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
        return np.concatenate(R), np.concatenate(C), np.concatenate(V)

    def candidates(self, ds: Dataset) -> pd.DataFrame:
        r, c, v = self.candidate_rows(ds.records)
        ids = ds.records["entity_id"].to_numpy()
        return pd.DataFrame({"s1_id": ids[r], "cand_id": ids[c], "score": v.astype(float)})
