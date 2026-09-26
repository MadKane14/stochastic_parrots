"""Name-token x address-word blocker (p3, T15): IDF cosine over (name token, address word) key pairs, per country group.

Why: after p2_translit + the empty-address channel, a large class of full-density misses (exploration/p3/
kaggle_t11a/out/blk_dense_out.md, missed-pairs sample) has an exact or near-exact name and the same street / city
words, but a perturbed house number and re-spelled street types or states:
    `equinox llc | 6600 valley road newburgh in`  <>  `equinox llc | 660 valley rd newburgh indiana`
    `eye medicine | 303 8th avenue pipestone mn`   <>  `eye medicine | 301 eighth ave pipestone mn`
p1_compound's keys that could link them are either too common to be indexed (name pairs such as `eye medicine`,
single street words) or broken by the number (number x word / number x name). A key built from ONE name token and
ONE address word (`equinox`x`newburgh`, `medicine`x`pipestone`) is rare even when both words are common, and it
survives number perturbations and `road`/`rd` drift (those tokens are too common to be indexed anyway).

Features per record: keys (name token of `name_field`, address word of `addr_field`) for the first `name_cap`
name tokens x the first `addr_cap` non-number address tokens (p1_compound tokenisation: digits split from
letters, leading zeros stripped). Within each group of equal `country` string (generic grouping, no value is
special-cased): IDF over the group, rows L2-normalised over all their keys, only keys with in-group df <= `max_df`
indexed; forward top-k per S1 and reverse top-k_rev per right record; `rev_cap` caps reverse-only pairs per S1
(see p3_revcap). Right records without address words have no keys (the empty-address channel covers them).
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd

from ber.blocking.p1_compound import _binary_csr, _gather, tokenize
from ber.blocking.p3_chargram import chunked_topk, idf_cosine_csr
from ber.blocking.p3_revcap import cap_reverse, dedup_forward_first
from ber.data import Dataset
from ber.interfaces import Blocker
from ber.registry import register

log = logging.getLogger(__name__)
_CACHE: dict = {}
_BITS = np.uint64(28)            # token ids of each side must fit in 28 bits


def _first_per_row(rows: np.ndarray, cap: int) -> np.ndarray:
    """Mask keeping the first `cap` entries of each run of equal `rows` (rows sorted)."""
    if not len(rows):
        return np.zeros(0, bool)
    start = np.r_[True, rows[1:] != rows[:-1]]
    first_idx = np.maximum.accumulate(np.where(start, np.arange(len(rows)), 0))
    return (np.arange(len(rows)) - first_idx) < cap


@register("blocker", "p3_nameaddr")
class NameAddrBlocker(Blocker):
    """Top-k by IDF cosine over (name token x address word) keys within equal-country-string groups."""

    transductive = True   # IDF / df cap computed on the records being blocked (no labels used)

    def __init__(self, name_field: str = "name_core", addr_field: str = "addr_norm", name_cap: int = 4,
                 addr_cap: int = 10, min_word_len: int = 2, k: int = 10, k_rev: int = 1, max_df: int = 300,
                 min_score: float = 0.0, hit_budget: float = 1e7, group_by: str | None = "country",
                 workers: int = 4, rev_cap: int | None = None):
        self.name_field, self.addr_field = name_field, addr_field
        self.name_cap, self.addr_cap, self.min_word_len = int(name_cap), int(addr_cap), int(min_word_len)
        self.k, self.k_rev, self.max_df, self.min_score = int(k), int(k_rev), int(max_df), float(min_score)
        self.hit_budget, self.group_by, self.workers = float(hit_budget), group_by, int(workers)
        self.rev_cap = None if rev_cap is None else int(rev_cap)

    def _tokens(self, recs: pd.DataFrame, field: str):
        key = (id(recs), len(recs), field)
        if key not in _CACHE:
            if len(_CACHE) > 4:
                _CACHE.clear()
            _CACHE[key] = tokenize(recs[field])
        return _CACHE[key]

    def _keys(self, recs: pd.DataFrame, rows: np.ndarray):
        """(row_local int64, key uint64) for all (name token, address word) pairs of `rows`."""
        name_tok, addr_tok = self._tokens(recs, self.name_field), self._tokens(recs, self.addr_field)
        if max(len(name_tok[4]), len(addr_tok[4])) >= (1 << int(_BITS)):
            raise ValueError("vocabulary exceeds 28-bit key halves")
        nr, _, nt = _gather(name_tok[0], name_tok[1], rows, 64)
        name_len = pd.Series(name_tok[4].to_numpy(zero_copy_only=False)).str.len().to_numpy()
        ok = name_len[nt] >= self.min_word_len
        nr, nt = nr[ok], nt[ok]
        keep = _first_per_row(nr, self.name_cap)
        nr, nt = nr[keep], nt[keep]
        ar, _, at = _gather(addr_tok[0], addr_tok[1], rows, 64)
        addr_len = pd.Series(addr_tok[4].to_numpy(zero_copy_only=False)).str.len().to_numpy()
        ok = ~addr_tok[2][at] & (addr_len[at] >= self.min_word_len)     # words only: numbers are the noisy part
        ar, at = ar[ok], at[ok]
        keep = _first_per_row(ar, self.addr_cap)
        ar, at = ar[keep], at[keep]
        # cross product within each row: name tokens x address words
        cb = np.bincount(ar, minlength=len(rows))
        sb = np.r_[0, np.cumsum(cb)][:-1]
        cnt = cb[nr]
        rep_r = np.repeat(nr, cnt)
        rep_n = np.repeat(nt, cnt).astype(np.uint64)
        off = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        rep_a = at[np.repeat(sb[nr], cnt) + off].astype(np.uint64)
        return rep_r, (rep_n << _BITS) | rep_a

    def candidate_rows_prov(self, recs: pd.DataFrame):
        """(s1_row, right_row, score, reverse_only) positional row indices into `recs`, deduplicated."""
        is_s1 = (recs["source"] == 1).to_numpy()
        groups = recs[self.group_by].to_numpy() if self.group_by else np.zeros(len(recs), dtype=np.int8)
        codes, _ = pd.factorize(groups)
        workers = max(1, min(self.workers, int(os.environ.get("BER_THREADS", "8"))))
        fwd, rev = [], []
        for g in np.unique(codes):
            in_g = codes == g
            i1, i2 = np.flatnonzero(in_g & is_s1), np.flatnonzero(in_g & ~is_s1)
            if len(i1) == 0 or len(i2) == 0:
                continue
            t0 = time.time()
            rows = np.r_[i1, i2]
            r_loc, keys = self._keys(recs, rows)
            col, uniq = pd.factorize(keys)
            del keys
            X = idf_cosine_csr(_binary_csr(r_loc, col, len(rows), len(uniq)), self.max_df)
            del r_loc, col
            A, B = X[: len(i1)], X[len(i1):]
            del X
            log.info("group %s: %d S1 x %d right, %d keys, %.1f indexed keys/S1, built in %.0fs",
                     g, len(i1), len(i2), len(uniq), A.nnz / max(1, len(i1)), time.time() - t0)
            if self.k:
                t0 = time.time()
                r, c, v = chunked_topk(A, B, self.k, self.min_score, self.hit_budget, workers)
                fwd.append((i1[r], i2[c], v.astype(np.float32)))
                log.info("group %s forward top-%d: %d pairs, %.0fs", g, self.k, len(r), time.time() - t0)
            if self.k_rev:
                t0 = time.time()
                c, r, v = chunked_topk(B, A, self.k_rev, self.min_score, self.hit_budget, workers)
                rev.append((i1[r], i2[c], v.astype(np.float32)))
                log.info("group %s reverse top-%d: %d pairs, %.0fs", g, self.k_rev, len(r), time.time() - t0)
            del A, B
        return dedup_forward_first(fwd, rev, len(recs))

    def candidate_rows(self, recs: pd.DataFrame):
        """(s1_row, right_row, score): forward + reverse, reverse-only pairs capped per S1 by `rev_cap`."""
        r, c, v, is_rev = self.candidate_rows_prov(recs)
        keep = cap_reverse(r, v, is_rev, self.rev_cap)
        return r[keep], c[keep], v[keep]

    def candidates(self, ds: Dataset) -> pd.DataFrame:
        r, c, v = self.candidate_rows(ds.records)
        ids = ds.records["entity_id"].to_numpy()
        return pd.DataFrame({"s1_id": ids[r], "cand_id": ids[c], "score": v.astype(float)})
