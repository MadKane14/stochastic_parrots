"""Profile of true pairs the blocker never proposed, from a dense-CV run's OOF pairs (p1).
    python exploration/p1/blk_miss_profile.py <dense_cv run name>
Categories on the RAW records (so normalizer-independent): right address empty / short, right name non-ASCII,
web-style name, shares a digit run with the S1 address, S1 has how many true matches; plus 25 examples.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code/business_entity_resolution/src"))
from ber.config import env  # noqa: E402
from ber.data import load_part  # noqa: E402

run = sys.argv[1]
art = Path(env("BER_ARTIFACTS"))
oof = pd.read_parquet(art / "dense_cv" / "oof" / f"{run}.parquet", columns=["li", "ri", "label"])
full = load_part(env("BER_DATA_DIR"), "train", cache_dir=art / "cache")
recs = full.records
n = len(recs)
idx = pd.Index(recs["entity_id"].to_numpy())
g_s = [s for s in full.gt for _ in full.gt[s]]
g_c = [c for s in full.gt for c in full.gt[s]]
L, R = idx.get_indexer(g_s).astype(np.int64), idx.get_indexer(g_c).astype(np.int64)
cand = np.unique(oof["li"].to_numpy().astype(np.int64) * n + oof["ri"].to_numpy().astype(np.int64))
miss = ~np.isin(L * n + R, cand)
Lm, Rm = L[miss], R[miss]
A, B = recs.iloc[Lm].reset_index(drop=True), recs.iloc[Rm].reset_index(drop=True)
digits = lambda s: set(re.findall(r"\d+", s))
ntrue = pd.Series(L).value_counts()
prof = pd.DataFrame({
    "right_addr_empty": B.business_address.str.strip().str.len() == 0,
    "right_addr_<=3_tokens": B.business_address.str.split().str.len().fillna(0) <= 3,
    "right_name_nonascii": ~B.business_name.map(str.isascii),
    "right_addr_nonascii": ~B.business_address.map(str.isascii),
    "web_name": B.business_name.str.contains(r"\.(?:com|in|net|org)\b", case=False, regex=True),
    "shares_digit_run": [bool(digits(a) & digits(b)) for a, b in zip(A.business_address, B.business_address)],
    "s1_has_>=5_matches": ntrue.reindex(Lm).to_numpy() >= 5,
})
print(f"{run}: {miss.sum():,} of {len(L):,} true pairs not in candidates ({miss.mean():.4f})")
print("share of misses with property:\n" + prof.mean().round(4).to_string())
print("by country:\n" + A.country.value_counts(normalize=True).round(3).to_string())
print("miss rate by right-address emptiness (all true pairs):")
allB = recs.iloc[R]
emp = (allB.business_address.str.strip().str.len() == 0).to_numpy()
print(f"  empty: {miss[emp].mean():.4f} of {emp.sum():,}   non-empty: {miss[~emp].mean():.4f} of {(~emp).sum():,}")
rng = np.random.default_rng(0)
for i in rng.choice(len(Lm), 25, replace=False):
    print(f"- `{A.business_name[i]}` | `{A.business_address[i]}` [{A.country[i]}]  ⇄  `{B.business_name[i]}` | `{B.business_address[i]}`")
