"""p4 (T14b, for p2): label-free, target-fitted address canonicalization on top of p2_translit (transductive).

Train-learned maps miss an unseen country's forms and can contradict them (train `st -> street`; French `st` = saint;
French `r` = rue is never learned because p2's learner skips 1-letter tokens). Per country-string group of the records
being normalized (no labels, no hand lists, no country values in code):
  1. confident candidate pairs: S1 x S2/S3 sharing >= 2 (house number, address word) keys + name token_set_ratio >= 90
  2. abbreviation map mined on them (p2's learn_canonical_map + strict 1-letter forms), overriding train entries
  3. tokens S2/S3 uses >= 100x more often than S1 after canonicalization are dropped (`no`/`n` from N°)
Evidence: reports/p4/T14b_for_p2.md (France test audit; validated without labels on India/US train: true-pair address
overlap +0.048 / +0.193, random +0.002 / +0.009).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from ber.normalize.p2_translit import TranslitNormalizer, is_abbrev, learn_canonical_map
from ber.registry import register


def confident_pairs(addr_tok: list[list[str]], name_str: list[str], is_s1: np.ndarray, max_df: int = 50,
                    min_shared: int = 2, min_name_sim: float = 90.0) -> tuple[np.ndarray, np.ndarray]:
    """Label-free high-precision pairs: S1 x S2/S3 sharing >= `min_shared` (house number, address word >= 4 letters)
    keys (keys held by > max_df right records are ignored) and with name token_set_ratio >= `min_name_sim`."""
    from rapidfuzz import fuzz, process
    rid, key = [], []
    for i, toks in enumerate(addr_tok):
        nums = [t for t in toks if t.isdigit()]
        if nums:
            for w in {t for t in toks if t.isalpha() and len(t) >= 4}:
                rid.append(i); key.append(nums[0] + "|" + w)
    k = pd.DataFrame({"row": np.array(rid, np.int64), "key": key})
    k["s1"] = is_s1[k["row"].to_numpy()] if len(k) else np.zeros(0, bool)
    L, R = k[k["s1"]], k[~k["s1"]]
    R = R[R["key"].map(R["key"].value_counts()) <= max_df]
    m = L[["row", "key"]].merge(R[["row", "key"]], on="key", suffixes=("_s1", "_r"))
    c = m.groupby(["row_s1", "row_r"]).size().rename("shared").reset_index()
    c = c[c["shared"] >= min_shared]
    if not len(c):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    s, r = c["row_s1"].to_numpy(), c["row_r"].to_numpy()
    sim = process.cpdist([name_str[i] for i in s], [name_str[i] for i in r], scorer=fuzz.token_set_ratio, workers=4)
    ok = sim >= min_name_sim
    return s[ok], r[ok]


def mine_canon(left: list[list[str]], right: list[list[str]], min_support: int = 50, min_precision: float = 0.5,
               min_support_1: int = 500, min_precision_1: float = 0.8) -> dict[str, str]:
    """Abbreviation -> full form from confident pairs (left = S1 tokens, right = S2/S3 tokens). >= 2-letter forms via
    p2's learner; 1-letter forms (e.g. `r`) only if the letter is unshared in >= min_support_1 pairs and one full form
    (which it abbreviates and which is unshared on the other side) accounts for >= min_precision_1 of them."""
    out = dict(learn_canonical_map(left, right, min_support, min_precision, 5))
    seen, link = Counter(), defaultdict(Counter)
    for L, R in zip(left, right):
        sl, sr = set(L), set(R)
        for a, other_only in ((sr - sl, sl - sr), (sl - sr, sr - sl)):
            for x in a:
                if len(x) == 1 and x.isalpha():
                    seen[x] += 1
                    for y in other_only:
                        if len(y) > 1 and y.isalpha() and is_abbrev(x, y):
                            link[x][y] += 1
    for x, n in seen.items():
        if n >= min_support_1 and link[x]:
            y, k = link[x].most_common(1)[0]
            if k / n >= min_precision_1 and x not in out:
                out[x] = out.get(y, y)
    return out


def merge_maps(train_map: dict[str, str], target_map: dict[str, str]) -> dict[str, str]:
    """Target evidence wins: a short form mined on the target data overrides the train entry (`st`: saint > street)."""
    return {**train_map, **target_map}


def side_skew_junk(tok_s1: list[list[str]], tok_right: list[list[str]], min_ratio: float = 100.0,
                   min_df_right: int = 1000, max_df_s1_frac: float = 1e-3) -> set[str]:
    """Tokens (after canonicalization) that S2/S3 records use often and S1 records almost never (`no`, `n` from `N°`):
    df_right/n_right >= min_ratio * df_s1/n_s1, df_right >= min_df_right, df_s1/n_s1 <= max_df_s1_frac."""
    d1, dr = Counter(), Counter()
    for t in tok_s1:
        d1.update(set(t))
    for t in tok_right:
        dr.update(set(t))
    n1, nr = max(1, len(tok_s1)), max(1, len(tok_right))
    return {t for t, c in dr.items()
            if c >= min_df_right and d1[t] / n1 <= max_df_s1_frac and (c / nr) >= min_ratio * max(d1[t], 1) / n1}


def apply(tokens: list[str], cmap: dict[str, str], junk: set[str]) -> list[str]:
    return [cmap.get(t, t) for t in tokens if cmap.get(t, t) not in junk]


@register("normalizer", "p4_tcanon")
class TargetCanonNormalizer(TranslitNormalizer):
    """p2_translit + label-free target canonicalization of `addr_norm` per country-string group (see module doc)."""
    transductive = True

    def __init__(self, tc_min_support: int = 50, tc_min_precision: float = 0.5, tc_min_support_1: int = 500,
                 tc_min_precision_1: float = 0.8, tc_junk_ratio: float = 100.0, tc_junk_min_df: int = 1000, **kw):
        super().__init__(**kw)
        self.tc = dict(min_support=int(tc_min_support), min_precision=float(tc_min_precision),
                       min_support_1=int(tc_min_support_1), min_precision_1=float(tc_min_precision_1))
        self.tc_junk_ratio, self.tc_junk_min_df = float(tc_junk_ratio), int(tc_junk_min_df)
        self.tc_stats: dict = {}

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        out = super().transform(records)
        addr = out["addr_norm"].to_numpy(dtype=object).copy()
        names = out["name_norm"].to_numpy(dtype=object)
        is_s1 = (out["source"] == 1).to_numpy()
        train_canon = getattr(self, "canon", {}).get("addr", {})
        for g, idx in pd.Series(np.arange(len(out))).groupby(out["country"].to_numpy()).groups.items():
            idx = np.asarray(idx)
            tok = [addr[i].split() for i in idx]
            s1 = is_s1[idx]
            if s1.sum() == 0 or (~s1).sum() == 0:
                continue
            cs, cr = confident_pairs(tok, [names[i] for i in idx], s1)
            cmap = merge_maps(train_canon, mine_canon([tok[i] for i in cs], [tok[i] for i in cr], **self.tc)) \
                if len(cs) else dict(train_canon)
            canon_tok = [[cmap.get(t, t) for t in x] for x in tok]
            junk = side_skew_junk([x for x, a in zip(canon_tok, s1) if a], [x for x, a in zip(canon_tok, s1) if not a],
                                  self.tc_junk_ratio, self.tc_junk_min_df)
            for k, i in enumerate(idx):
                addr[i] = " ".join(t for t in canon_tok[k] if t not in junk)
            self.tc_stats[str(len(self.tc_stats))] = {"records": int(len(idx)), "confident_pairs": int(len(cs)),
                                                      "map_size": len(cmap), "junk": sorted(junk)[:50]}
        out["addr_norm"] = addr
        return out
