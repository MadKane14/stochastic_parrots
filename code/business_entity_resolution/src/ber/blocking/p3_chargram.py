"""Char n-gram blocker (p3, T05): IDF cosine over character n-grams of the de-spaced name, per country-string group.

Why: p1_compound keys are whole tokens, so one typo (`hemandez`, `consturcrions`, `brownvi1le`) or a glued web-style
name (`supremepackaging`, `veteransdiamondministries com`) breaks every key containing that word (I-12, I-13).
Character 3/4-grams of the name with its spaces removed survive both: most grams are still shared.

Features per record: binary bag of hashed n-grams (n in `ngrams`) of `field` with spaces removed, capped at
`max_len` characters and wrapped in start/end marks (so prefixes/suffixes count); optionally plus the whole address
tokens of `addr_field` (tokenised like p1_compound, numbers included), scaled by `addr_weight`.
Within each group of equal `country` string (generic grouping, no value is special-cased): IDF over the group,
rows L2-normalised over ALL their features, only features with in-group df <= `max_df` are indexed (the cap keeps
the sparse product near-linear; frequent grams still count in the norms). Forward top-k right records per S1 and
optionally reverse top-k_rev S1 per right record (an S2/S3 record belongs to at most one S1, and S1 has ~5x fewer
records to compete: this is what reaches right records with an empty address, 22% of p1's misses).

Memory: the CSR is built once per group and weighted in place (no dense or duplicated copies); the product
S1 x right runs in row chunks of ~`hit_budget` postings in forked workers (same scheme as p1_idf_overlap).
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ber.blocking.p1_compound import _binary_csr, _gather, tokenize
from ber.blocking.p1_idf_overlap import topk_rows
from ber.data import Dataset
from ber.interfaces import Blocker
from ber.registry import register

log = logging.getLogger(__name__)
_CACHE: dict = {}   # (id(records), len, field, ...) -> char codes / address tokens; shared by instances

_FNV_OFF = np.uint64(0xCBF29CE484222325)
_FNV_PRIME = np.uint64(0x100000001B3)
_ADDR_TAG = np.uint64(0xADD) << np.uint64(52)   # address-token keys live apart from the gram hashes


def char_codes(values: pd.Series, max_len: int, chunk: int = 1_000_000):
    """Strings with spaces removed, first `max_len` characters, wrapped in marks \\x02 ... \\x03, as code points.
    Returns (offsets int64[n+1], codes uint32[total]); row i = codes[offsets[i]:offsets[i+1]]."""
    vals = values.to_numpy(dtype=object)
    offs, codes, base = [np.zeros(1, np.int64)], [], 0
    for a in range(0, len(vals), chunk):
        parts = [str(v).replace(" ", "")[:max_len] for v in vals[a:a + chunk]]
        lens = np.fromiter(map(len, parts), np.int64, len(parts)) + 2
        c = np.frombuffer(("\x02" + "\x03\x02".join(parts) + "\x03").encode("utf-32-le", "surrogatepass"),
                          dtype=np.uint32)
        codes.append(c)
        offs.append(base + np.cumsum(lens))
        base += int(lens.sum())
    return np.concatenate(offs), np.concatenate(codes) if codes else np.zeros(0, np.uint32)


def gram_keys(offsets, codes, rows, ngrams, chunk: int = 500_000):
    """(row_local int32, key uint64) for every n-gram (n in `ngrams`) of `rows` (positional record indices).
    Keys are 64-bit FNV-1a hashes of the code points, seeded by n (collisions are negligible at ~1e7 grams)."""
    all_lens = offsets[rows + 1] - offsets[rows]
    total = int(sum(np.maximum(all_lens - n + 1, 0).sum() for n in ngrams))
    R, K = np.empty(total, np.int32), np.empty(total, np.uint64)   # filled in place: no concatenate copy
    at = 0
    for a in range(0, len(rows), chunk):
        rr = rows[a:a + chunk]
        lens = all_lens[a:a + chunk]
        rloc = np.repeat(np.arange(a, a + len(rr), dtype=np.int32), lens)
        first = np.repeat(np.cumsum(lens) - lens, lens)
        pos = np.arange(int(lens.sum()), dtype=np.int64) - first
        c = codes[np.repeat(offsets[rr], lens) + pos].astype(np.uint64)
        room = np.repeat(lens, lens) - pos          # characters left from this position to the row end
        del first, pos
        for n in ngrams:
            ok = np.flatnonzero(room >= n)
            h = np.full(len(ok), _FNV_OFF ^ np.uint64(n), np.uint64)
            for t in range(n):
                h ^= c[ok + t]
                h *= _FNV_PRIME
            R[at:at + len(ok)], K[at:at + len(ok)] = rloc[ok], h
            at += len(ok)
    assert at == total
    return R, K


# worker globals (set before fork so children share memory copy-on-write)
_W: dict = {}


def _chunk(bounds):
    a, b = bounds
    r, c, v = topk_rows(_W["A"][a:b] @ _W["BT"], _W["k"], _W["min_score"])
    return r + a, c, v


def chunked_topk(A: sp.csr_matrix, B: sp.csr_matrix, k: int, min_score: float, hit_budget: float, workers: int):
    """Top-k columns of A @ B.T per row of A, in row chunks of ~hit_budget postings (bound on each chunk's nnz)."""
    BT = B.T.tocsr()
    df = np.asarray(BT.getnnz(axis=1)).ravel().astype(np.float64)
    hits = np.cumsum(A.astype(bool).astype(np.float64) @ df)
    cuts = np.searchsorted(hits, np.arange(hit_budget, hits[-1] if len(hits) else 0, hit_budget))
    edges = np.unique(np.concatenate([[0], cuts, [A.shape[0]]])).astype(int)
    bounds = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    log.info("  product: %.2e postings, %d chunks", hits[-1] if len(hits) else 0, len(bounds))
    _W.update(A=A, BT=BT, k=k, min_score=min_score)
    try:
        if workers > 1 and len(bounds) > 1 and "fork" in mp.get_all_start_methods():
            with mp.get_context("fork").Pool(min(workers, len(bounds))) as pool:
                parts = pool.map(_chunk, bounds, chunksize=1)
        else:
            parts = [_chunk(bd) for bd in bounds]
    finally:
        _W.clear()
    if not parts:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    return tuple(np.concatenate(x) for x in zip(*parts))


def idf_cosine_csr(X: sp.csr_matrix, max_df: int, col_weight: np.ndarray | None = None) -> sp.csr_matrix:
    """In place: binary CSR -> IDF-weighted, row-L2-normalised (norm over all features); then entries of features
    with df > max_df are removed (not indexed). `col_weight` multiplies the IDF of each column (e.g. address)."""
    n = X.shape[0]
    df = np.bincount(X.indices, minlength=X.shape[1])
    idf = (np.log((n + 1) / (df + 1)) + 1.0).astype(np.float32)
    if col_weight is not None:
        idf *= col_weight
    w = idf[X.indices]
    rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(X.indptr))
    norm = np.sqrt(np.bincount(rows, weights=w.astype(np.float64) ** 2, minlength=n)).astype(np.float32)
    w /= np.maximum(norm, 1e-9)[rows]
    del rows
    w[df[X.indices] > max_df] = 0.0
    X.data = w
    X.eliminate_zeros()
    return X


@register("blocker", "p3_chargram")
class CharGramBlocker(Blocker):
    """Top-k by IDF cosine over char n-grams of the de-spaced name (+ optional address tokens), per country group."""

    transductive = True   # IDF / df cap computed on the records being blocked (no labels used)

    def __init__(self, field: str = "name_norm", ngrams: list[int] | tuple[int, ...] = (3, 4), k: int = 10,
                 k_rev: int = 0, max_df: int = 1000, min_score: float = 0.0, min_score_rev: float | None = None,
                 max_len: int = 40, addr_field: str | None = None, addr_weight: float = 1.0, addr_cap: int = 20,
                 hit_budget: float = 1e7, group_by: str | None = "country", workers: int = 4):
        self.field, self.ngrams, self.max_len = field, tuple(int(n) for n in ngrams), int(max_len)
        self.k, self.k_rev, self.max_df = int(k), int(k_rev), int(max_df)
        self.min_score = float(min_score)
        self.min_score_rev = float(min_score if min_score_rev is None else min_score_rev)
        self.addr_field, self.addr_weight, self.addr_cap = addr_field, float(addr_weight), int(addr_cap)
        self.hit_budget, self.group_by, self.workers = float(hit_budget), group_by, int(workers)

    def _cached(self, recs: pd.DataFrame, what: str, fn):
        key = (id(recs), len(recs), what)
        if key not in _CACHE:
            if len(_CACHE) > 4:
                _CACHE.clear()
            _CACHE[key] = fn()
        return _CACHE[key]

    def _features(self, recs: pd.DataFrame, rows: np.ndarray):
        """Binary CSR (len(rows) x n_features) + per-column weight (address columns get addr_weight)."""
        offsets, codes = self._cached(recs, f"chars:{self.field}:{self.max_len}",
                                      lambda: char_codes(recs[self.field], self.max_len))
        r, keys = gram_keys(offsets, codes, rows, self.ngrams)
        if self.addr_field:
            tok = self._cached(recs, f"tok:{self.addr_field}", lambda: tokenize(recs[self.addr_field]))
            ar, _, at = _gather(tok[0], tok[1], rows, self.addr_cap)
            r = np.concatenate([r, ar.astype(np.int32)])
            keys = np.concatenate([keys, _ADDR_TAG | at.astype(np.uint64)])
        col, uniq = pd.factorize(keys)
        del keys
        X = _binary_csr(r, col, len(rows), len(uniq))
        weight = None
        if self.addr_field and self.addr_weight != 1.0:
            is_addr = (np.asarray(uniq, dtype=np.uint64) >> np.uint64(52)) == np.uint64(0xADD)
            weight = np.where(is_addr, self.addr_weight, 1.0).astype(np.float32)
        return X, weight

    def candidate_rows(self, recs: pd.DataFrame):
        """(s1_row, right_row, score) positional row indices into `recs` (forward + reverse, deduplicated)."""
        is_s1 = (recs["source"] == 1).to_numpy()
        groups = recs[self.group_by].to_numpy() if self.group_by else np.zeros(len(recs), dtype=np.int8)
        codes, _ = pd.factorize(groups)
        workers = max(1, min(self.workers, int(os.environ.get("BER_THREADS", "8"))))
        R, C, V = [], [], []
        for g in np.unique(codes):
            in_g = codes == g
            i1, i2 = np.flatnonzero(in_g & is_s1), np.flatnonzero(in_g & ~is_s1)
            if len(i1) == 0 or len(i2) == 0:
                continue
            t0 = time.time()
            X, weight = self._features(recs, np.r_[i1, i2])
            X = idf_cosine_csr(X, self.max_df, weight)
            A, B = X[: len(i1)], X[len(i1):]
            del X
            log.info("group %s: %d S1 x %d right, %.1f indexed features/S1, built in %.0fs",
                     g, len(i1), len(i2), A.nnz / max(1, len(i1)), time.time() - t0)
            if self.k:
                t0 = time.time()
                r, c, v = chunked_topk(A, B, self.k, self.min_score, self.hit_budget, workers)
                R.append(i1[r]); C.append(i2[c]); V.append(v.astype(np.float32))
                log.info("group %s forward top-%d: %d pairs, %.0fs", g, self.k, len(r), time.time() - t0)
            if self.k_rev:
                t0 = time.time()
                c, r, v = chunked_topk(B, A, self.k_rev, self.min_score_rev, self.hit_budget, workers)
                R.append(i1[r]); C.append(i2[c]); V.append(v.astype(np.float32))
                log.info("group %s reverse top-%d: %d pairs, %.0fs", g, self.k_rev, len(r), time.time() - t0)
            del A, B
        if not R:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
        r, c, v = np.concatenate(R), np.concatenate(C), np.concatenate(V)
        _, first = np.unique(r * len(recs) + c, return_index=True)   # forward and reverse can propose the same pair
        return r[first], c[first], v[first]

    def candidates(self, ds: Dataset) -> pd.DataFrame:
        r, c, v = self.candidate_rows(ds.records)
        ids = ds.records["entity_id"].to_numpy()
        return pd.DataFrame({"s1_id": ids[r], "cand_id": ids[c], "score": v.astype(float)})
