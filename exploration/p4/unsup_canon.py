"""T14b (p4 for p2): LABEL-FREE canonicalization learned on the TARGET data (transductive; no labels, no hand lists).

Why: p2's canonical map is learned from TRAIN true pairs (US/India), so it misses an unseen country's forms and can
contradict them (train `st -> street`; in French addresses `st` = `saint`). See exploration/p4/france_audit_out.md.
Fix, all country-agnostic and fitted only on the records being normalized:
  1. confident candidate pairs of the target data: S1 x S2/S3 sharing >= 2 (house number, address word) keys and
     near-identical names (token_set_ratio >= 90)  -> `confident_pairs`
  2. abbreviation map mined on them: p2's `learn_canonical_map` (>= 2-letter forms) + single-letter forms under
     stricter thresholds (`r -> rue`), which p2's learner skips by design           -> `mine_canon`
  3. target evidence overrides conflicting train entries                             -> `merge_maps`
  4. residual tokens that appear almost only on the S2/S3 side after canonicalization (`no`, `n` from `N°`) -> junk
                                                                                     -> `side_skew_junk`
Usage (integration sketch for p2_translit.transform, target records): pairs = confident_pairs(...);
cmap = merge_maps(train_canon, mine_canon(...)); apply cmap, then drop side_skew_junk(...) tokens. Flag transductive.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "business_entity_resolution" / "src"))
from ber.normalize.p2_translit import is_abbrev, learn_canonical_map  # noqa: E402


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
