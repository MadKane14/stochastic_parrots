"""Scalable blocker on compound keys: IDF cosine over unigrams + token-pair features, per country-string group.

Why: names are drawn from a small vocabulary and repeat across many entities, and city/street words are common,
so single tokens either flood the candidate lists (kept) or lose recall (dropped). Pairs of tokens are rare even
when both tokens are common, and they survive word reordering. Features per record:
    U_name  name tokens                    U_addr  address tokens (numbers included)
    P_name  unordered pairs of name tokens B_addr  adjacent address token bigrams
    NW      address number x address word  NN      address number x name token
    NV      number variant x word / name   (optional: number with its first or last digit dropped, so
                                             "3829"~"829", "160"~"16", "418"~"419" still share a key)
Digits are split from letters ("26th" -> "26 th", "5ahaya" -> "5 ahaya") and leading zeros stripped ("001954").
Within each group of equal `country` string (generic grouping, no value is special-cased), features are
IDF-weighted, rows L2-normalised over ALL their features, and only features with in-group df <= max_df are
indexed. Top-k per S1 (forward) and optionally top-k_rev S1 per right record (reverse; every S2/S3 record
belongs to at most one S1 entity, so its own best S1 matches are strong candidates).
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sp

from ber.blocking.p1_idf_overlap import _group_topk
from ber.data import Dataset
from ber.interfaces import Blocker
from ber.registry import register

log = logging.getLogger(__name__)
_TOK_CACHE: dict = {}


def tokenize(values: pd.Series):
    """Whitespace tokens with digit/letter split + leading zeros stripped.
    Returns (offsets[int64, n+1], token_ids[int64], is_number[bool per vocab id], vocab[list])."""
    arr = pa.array(values.to_numpy(dtype=object), type=pa.large_string())
    arr = pc.replace_substring_regex(arr, r"(\d)(\pL)", r"\1 \2")
    arr = pc.replace_substring_regex(arr, r"(\pL)(\d)", r"\1 \2")
    arr = pc.replace_substring_regex(arr, r"\b0+(\d)", r"\1")
    lists = pc.utf8_split_whitespace(arr)
    flat = lists.flatten()
    offsets = lists.offsets.to_numpy().astype(np.int64)
    enc = pc.dictionary_encode(flat)
    ids = enc.indices.to_numpy().astype(np.int64)
    vocab = enc.dictionary
    is_num = pc.utf8_is_digit(vocab).to_numpy(zero_copy_only=False)
    empty = pc.equal(vocab, "").to_numpy(zero_copy_only=False)
    if empty.any():   # drop empty tokens
        keep = ~empty[ids]
        row = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))[keep]
        ids = ids[keep]
        offsets = np.r_[0, np.cumsum(np.bincount(row, minlength=len(offsets) - 1))].astype(np.int64)
    return offsets, ids, is_num, _variant_ids(vocab, is_num)


def _variant_ids(vocab, is_num):
    """For number tokens with >= 2 digits: ids of the number with its first / last digit dropped. A variant that
    equals an existing number token gets that token's id; others get fresh ids above the vocabulary.
    Returns int64 array (vocab_size, 2), -1 where not applicable."""
    strs = vocab.to_numpy(zero_copy_only=False)
    out = np.full((len(strs), 2), -1, np.int64)
    nums = np.flatnonzero(is_num)
    if not len(nums):
        return out
    num_s = pd.Series(strs[nums])
    long_ = np.flatnonzero(num_s.str.len().to_numpy() >= 2)
    heads = num_s.iloc[long_].str[1:].str.lstrip("0").replace("", "0")
    tails = num_s.iloc[long_].str[:-1]
    codes, _ = pd.factorize(pd.concat([num_s, heads, tails], ignore_index=True))
    code_to_id = np.full(codes.max() + 1, -1, np.int64)
    code_to_id[codes[: len(nums)]] = nums                      # existing number tokens keep their id
    fresh = code_to_id < 0
    code_to_id[fresh] = len(strs) + np.arange(fresh.sum())      # unseen variants: new ids above the vocab
    m = len(long_)
    out[nums[long_], 0] = code_to_id[codes[len(nums): len(nums) + m]]
    out[nums[long_], 1] = code_to_id[codes[len(nums) + m:]]
    return out


def _gather(offsets, ids, rows, cap):
    """Tokens of `rows` (first `cap` per row) -> (row_local, pos, token)."""
    lens = np.minimum(offsets[rows + 1] - offsets[rows], cap)
    starts = np.repeat(offsets[rows], lens)
    first = np.repeat(np.cumsum(lens) - lens, lens)
    pos = np.arange(lens.sum()) - first
    return np.repeat(np.arange(len(rows)), lens), pos, ids[starts + pos]


def _cross(ra, ta, rb, tb, n_rows):
    """All (a, b) token pairs within the same row; ra/rb sorted by row."""
    cb = np.bincount(rb, minlength=n_rows)
    sb = np.r_[0, np.cumsum(cb)][:-1]
    cnt = cb[ra]
    rep_r = np.repeat(ra, cnt)
    rep_a = np.repeat(ta, cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return rep_r, rep_a, tb[np.repeat(sb[ra], cnt) + off]


def _key(fam: int, a, b):
    return (np.int64(fam) << 58) | (a.astype(np.int64) << 29) | b.astype(np.int64)


def compound_features(name_tok, addr_tok, rows, name_cap=8, addr_cap=20, num_cap=4, num_variants=False):
    """(row_local, feature_key int64) for `rows` (positional indices into the tokenised arrays)."""
    n = len(rows)
    nr, npos, nt = _gather(*name_tok[:2], rows, name_cap)
    ar, apos, at = _gather(*addr_tok[:2], rows, addr_cap)
    a_is_num = addr_tok[2][at]
    R, K = [nr, ar], [_key(0, np.zeros_like(nt), nt), _key(1, np.zeros_like(at), at)]
    for d in range(1, name_cap):   # unordered name token pairs
        i = np.flatnonzero(npos[:-d] + d == npos[d:]) if len(npos) > d else np.zeros(0, np.int64)
        i = i[nr[i] == nr[i + d]]
        a, b = nt[i], nt[i + d]
        R.append(nr[i]); K.append(_key(2, np.minimum(a, b), np.maximum(a, b)))
    if len(apos) > 1:              # adjacent address bigrams
        i = np.flatnonzero((apos[:-1] + 1 == apos[1:]) & (ar[:-1] == ar[1:]))
        R.append(ar[i]); K.append(_key(3, at[i], at[i + 1]))
    # address numbers (first num_cap per row)
    nr_all, nt_all = ar[a_is_num], at[a_is_num]
    if len(nr_all):
        first = np.r_[True, nr_all[1:] != nr_all[:-1]]
        grp_start = np.maximum.accumulate(np.where(first, np.arange(len(nr_all)), 0))
        sel = (np.arange(len(nr_all)) - grp_start) < num_cap
        num_r, num_t = nr_all[sel], nt_all[sel]
        w = ~a_is_num
        r, a, b = _cross(num_r, num_t, ar[w], at[w], n)
        R.append(r); K.append(_key(4, a, b))
        r, a, b = _cross(num_r, num_t, nr, nt, n)
        R.append(r); K.append(_key(5, a, b))
        if num_variants:   # variants of the first 2 numbers, paired with words and name tokens
            first2 = (np.arange(len(num_r)) - np.maximum.accumulate(
                np.where(np.r_[True, num_r[1:] != num_r[:-1]], np.arange(len(num_r)), 0))) < 2
            var = addr_tok[3][num_t[first2]]                       # (m, 2)
            vr = np.repeat(num_r[first2], 2)
            vt = var.ravel()
            ok = vt >= 0
            vr, vt = vr[ok], vt[ok]
            vr_s = np.argsort(vr, kind="stable")
            vr, vt = vr[vr_s], vt[vr_s]
            vt_all = np.r_[vt, num_t[first2]]                      # the original number joins its variants
            vr_all = np.r_[vr, num_r[first2]]
            o = np.argsort(vr_all, kind="stable")
            vr_all, vt_all = vr_all[o], vt_all[o]
            r, a, b = _cross(vr_all, vt_all, ar[w], at[w], n)
            R.append(r); K.append(_key(6, a, b))
            r, a, b = _cross(vr_all, vt_all, nr, nt, n)
            R.append(r); K.append(_key(7, a, b))
    return np.concatenate(R), np.concatenate(K)


def _binary_csr(rows, cols, n_rows, n_cols):
    m = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n_rows, n_cols))
    m.sum_duplicates()
    m.data[:] = 1.0
    return m


@register("blocker", "p1_compound")
class CompoundKeyBlocker(Blocker):
    """Top-k by IDF cosine over unigram + compound token-pair features within equal-country-string groups."""

    transductive = True   # IDF / df cap computed on the records being blocked (no labels used)

    def __init__(self, k: int = 30, k_rev: int = 0, max_df: int = 5000, min_score: float = 0.0,
                 hit_budget: float = 1e7, name_field: str = "name_norm", addr_field: str = "addr_norm",
                 group_by: str | None = "country", name_cap: int = 8, addr_cap: int = 20, num_cap: int = 4,
                 num_variants: bool = False, max_df_unigram: int | None = None):
        self.k, self.k_rev, self.max_df, self.min_score = int(k), int(k_rev), int(max_df), float(min_score)
        self.hit_budget, self.name_field, self.addr_field, self.group_by = float(hit_budget), name_field, addr_field, group_by
        self.caps = (int(name_cap), int(addr_cap), int(num_cap), bool(num_variants))
        self.max_df_unigram = int(max_df_unigram) if max_df_unigram else None   # lower cap for single tokens

    def _tokens(self, recs: pd.DataFrame, field: str):
        key = (id(recs), len(recs), field)
        if key not in _TOK_CACHE:
            if len(_TOK_CACHE) > 4:
                _TOK_CACHE.clear()
            _TOK_CACHE[key] = tokenize(recs[field])
        return _TOK_CACHE[key]

    def candidate_rows(self, recs: pd.DataFrame):
        """(s1_row, right_row, score) positional row indices into `recs` (forward and reverse top-k, deduplicated)."""
        t0 = time.time()
        name_tok, addr_tok = self._tokens(recs, self.name_field), self._tokens(recs, self.addr_field)
        log.info("tokenize %.0fs", time.time() - t0)
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
                R.append(i1[r]); C.append(i2[c]); V.append(v.astype(np.float32))
                log.info("group %s forward top-%d: %.0fs", g, self.k, time.time() - t0)
            if self.k_rev:
                t0 = time.time()
                c, r, v = _group_topk(X2, X1, self.k_rev, max_df, self.min_score, self.hit_budget, workers)
                R.append(i1[r]); C.append(i2[c]); V.append(v.astype(np.float32))
                log.info("group %s reverse top-%d: %.0fs", g, self.k_rev, time.time() - t0)
        if not R:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
        r, c, v = np.concatenate(R), np.concatenate(C), np.concatenate(V)
        key = r * len(recs) + c
        _, first = np.unique(key, return_index=True)   # forward and reverse can propose the same pair
        return r[first], c[first], v[first]

    def candidates(self, ds: Dataset) -> pd.DataFrame:
        r, c, v = self.candidate_rows(ds.records)
        ids = ds.records["entity_id"].to_numpy()
        return pd.DataFrame({"s1_id": ids[r], "cand_id": ids[c], "score": v.astype(float)})
