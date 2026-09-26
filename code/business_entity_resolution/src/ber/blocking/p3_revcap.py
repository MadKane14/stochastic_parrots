"""p1_compound with a cap on the reverse flood (p3, T15).

p1_compound unions a forward top-k (right records per S1) with a reverse top-k_rev (S1 per right record). The
reverse direction is unbounded per S1: an S1 with a generic name/address is in the top-k_rev of hundreds of right
records (max ~380 cands for one S1 on full train, p1 blk_eval run 20:51), while the organizers rank smaller
candidate sets higher (docs/organizer_updates.md). `rev_cap` keeps every forward pair and, per S1, only the
`rev_cap` best-scoring reverse-only pairs. With `rev_cap: null` the candidates equal p1_compound's (same features,
IDF, products and dedup; the code below is p1_compound.candidate_rows with provenance kept).

`candidate_rows_prov` also returns which pairs came only from the reverse direction, so an evaluation can emulate
any rev_cap from one run (exploration/p3/blk_dense.py --rev-caps).
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd

from ber.blocking.p1_compound import CompoundKeyBlocker, _binary_csr, compound_features
from ber.blocking.p1_idf_overlap import _group_topk
from ber.registry import register

log = logging.getLogger(__name__)


def cap_reverse(r: np.ndarray, v: np.ndarray, is_rev: np.ndarray, cap: int | None) -> np.ndarray:
    """Keep mask: all forward pairs + per S1 row `r` the `cap` highest-scoring reverse-only pairs (ties: input order)."""
    keep = ~is_rev
    if cap is None:
        return np.ones(len(r), bool)
    idx = np.flatnonzero(is_rev)
    idx = idx[np.lexsort((-v[idx], r[idx]))]          # by S1, best score first (lexsort is stable)
    rr = r[idx]
    start = np.r_[0, np.flatnonzero(np.diff(rr)) + 1] if len(rr) else np.zeros(0, np.int64)
    pos = np.arange(len(rr)) - np.repeat(start, np.diff(np.r_[start, len(rr)]))
    keep[idx[pos < cap]] = True
    return keep


def dedup_forward_first(parts_f: list, parts_r: list, n: int):
    """Concatenate forward and reverse (row, col, score) parts; a pair found in both directions counts as forward.
    Returns r, c, v, is_rev (reverse-only flag)."""
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32))
    f = [np.concatenate(x) for x in zip(*parts_f)] if parts_f else list(empty)
    b = [np.concatenate(x) for x in zip(*parts_r)] if parts_r else list(empty)
    r, c, v = (np.concatenate([f[i], b[i]]) for i in range(3))
    is_rev = np.r_[np.zeros(len(f[0]), bool), np.ones(len(b[0]), bool)]
    _, first = np.unique(r * n + c, return_index=True)   # first occurrence = forward when both propose it
    return r[first], c[first], v[first], is_rev[first]


@register("blocker", "p3_compound_rc")
class RevCapCompoundBlocker(CompoundKeyBlocker):
    """p1_compound (imported, unchanged) + per-S1 cap on reverse-only candidates (`rev_cap`, null = no cap)."""

    transductive = True   # IDF / df caps computed on the records being blocked (no labels used), as p1_compound

    def __init__(self, rev_cap: int | None = None, **params):
        super().__init__(**params)
        self.rev_cap = None if rev_cap is None else int(rev_cap)

    def candidate_rows_prov(self, recs: pd.DataFrame):
        """(s1_row, right_row, score, reverse_only) positional row indices into `recs`, deduplicated."""
        t0 = time.time()
        name_tok, addr_tok = self._tokens(recs, self.name_field), self._tokens(recs, self.addr_field)
        log.info("tokenize %.0fs", time.time() - t0)
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
            r_loc, keys = compound_features(name_tok, addr_tok, rows, *self.caps)
            col, uniq = pd.factorize(keys)
            del keys
            X = _binary_csr(r_loc, col, len(rows), len(uniq))
            X1, X2 = X[: len(i1)], X[len(i1):]
            del X
            max_df = self.max_df
            if self.max_df_unigram is not None:   # families 0/1 = single tokens; pairs keep the general cap
                max_df = np.where((np.asarray(uniq) >> 58) <= 1, self.max_df_unigram, self.max_df)
            log.info("group %s: %d S1 x %d right, %d features (%.1f/record), built in %.0fs",
                     g, len(i1), len(i2), len(uniq), X1.nnz / max(1, len(i1)), time.time() - t0)
            if self.k:
                t0 = time.time()
                r, c, v = _group_topk(X1, X2, self.k, max_df, self.min_score, self.hit_budget, workers)
                fwd.append((i1[r], i2[c], v.astype(np.float32)))
                log.info("group %s forward top-%d: %.0fs", g, self.k, time.time() - t0)
            if self.k_rev:
                t0 = time.time()
                c, r, v = _group_topk(X2, X1, self.k_rev, max_df, self.min_score, self.hit_budget, workers)
                rev.append((i1[r], i2[c], v.astype(np.float32)))
                log.info("group %s reverse top-%d: %.0fs", g, self.k_rev, time.time() - t0)
        return dedup_forward_first(fwd, rev, len(recs))

    def candidate_rows(self, recs: pd.DataFrame):
        """(s1_row, right_row, score): forward + reverse, reverse-only pairs capped per S1 by `rev_cap`."""
        r, c, v, is_rev = self.candidate_rows_prov(recs)
        keep = cap_reverse(r, v, is_rev, self.rev_cap)
        if self.rev_cap is not None:
            log.info("rev_cap %d: kept %d of %d reverse-only pairs", self.rev_cap, int((keep & is_rev).sum()),
                     int(is_rev.sum()))
        return r[keep], c[keep], v[keep]
