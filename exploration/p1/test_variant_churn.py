"""Label-free churn of decision variants of the #9 blend on test, per country (p1, Sun 27 Sep).

    python exploration/p1/test_variant_churn.py <run T18> <run T28>

For each variant (mean@0.85, mean@0.9, min@0.8, min@0.7) vs #9 (mean@0.8): pairs removed / added per country, as a share
of #9's pairs, plus the house-number conflict rate (both addresses have numbers, none shared) of the removed pairs.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src"))
from ber.config import env
from ber.data import load_part
from ber.deciders.p1_assign import AssignDecider

art = Path(env("BER_ARTIFACTS"))
cols = ["s1_id", "cand_id", "score"]
A = pd.read_parquet(art / "runs" / sys.argv[1] / "scored_pairs.parquet", columns=cols)
B = pd.read_parquet(art / "runs" / sys.argv[2] / "scored_pairs.parquet", columns=cols)
m = A.merge(B, on=["s1_id", "cand_id"], how="outer", suffixes=("_a", "_b"))
m["score_a"] = m["score_a"].fillna(m["score_b"])
m["score_b"] = m["score_b"].fillna(m["score_a"])
test = load_part(env("BER_DATA_DIR"), "test", cache_dir=art / "cache")
r = test.records
ad = "address" if "address" in r.columns else "business_address"
nums = dict(zip(r["entity_id"], [frozenset(x.lstrip("0") or "0" for x in v)
                                 for v in r[ad].astype(str).str.findall(r"\d+")]))
s1c = test.s1.set_index("entity_id")["country"]


def decide(score, t):
    d = m[["s1_id", "cand_id"]].assign(score=score)
    x = AssignDecider(threshold=t).decide(None, d)
    return set(zip(x.s1_id, x.cand_id))


def conflict(pairs):
    return np.mean([bool(nums[a]) and bool(nums[b]) and not (nums[a] & nums[b]) for a, b in pairs]) if pairs else 0.0


mean = (m["score_a"] + m["score_b"]).to_numpy() / 2
mn = np.minimum(m["score_a"], m["score_b"]).to_numpy()
base = decide(mean, 0.8)
bc = pd.Series([s1c[a] for a, _ in base]).value_counts()
print("#9 pairs per country:", bc.to_dict(), "conflict", round(conflict(list(base)), 4), flush=True)
for name, sc, t in (("mean@0.85", mean, 0.85), ("mean@0.9", mean, 0.9), ("min@0.8", mn, 0.8), ("min@0.7", mn, 0.7)):
    v = decide(sc, t)
    rows = []
    for c in sorted(bc.index):
        rem = [p for p in base - v if s1c[p[0]] == c]
        add = [p for p in v - base if s1c[p[0]] == c]
        rows.append(f"{c}: -{len(rem)} ({len(rem) / bc[c]:.2%}, conf {conflict(rem):.3f}) +{len(add)}")
    print(f"{name:10s}", " | ".join(rows), flush=True)
