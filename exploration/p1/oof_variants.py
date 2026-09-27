"""Dense-CV macro F0.5 of score-combination variants of several OOF runs on the same pairs (p1, Sun 27 Sep).

    python exploration/p1/oof_variants.py <run T18> <run T28> [<run T27>]

Variants: mean, T28-weighted mean, min, 3-way mean, and a house-number guard (conflict = both addresses have numbers
and share none): mean where there is no conflict, min(A, B) where there is.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ber.config import env
from ber.data import load_part
from dense_cv import evaluate_dense

art = Path(env("BER_ARTIFACTS"))
a = pd.read_parquet(art / "dense_cv" / "oof" / f"{sys.argv[1]}.parquet")
others = [pd.read_parquet(art / "dense_cv" / "oof" / f"{r}.parquet", columns=["li", "ri", "oof"]) for r in sys.argv[2:]]
for o in others:
    assert (a["li"].to_numpy() == o["li"].to_numpy()).all() and (a["ri"].to_numpy() == o["ri"].to_numpy()).all()
full = load_part(env("BER_DATA_DIR"), "train", cache_dir=art / "cache")
recs = full.records
n = len(recs)
idx = pd.Index(recs["entity_id"].to_numpy())
ntrue = np.zeros(n, np.int64)
np.add.at(ntrue, idx.get_indexer(list(full.gt)), [len(v) for v in full.gt.values()])
s1_rows = np.flatnonzero((recs["source"] == 1).to_numpy())
meta = {"n": n, "s1_rows": s1_rows, "ntrue": ntrue}
ctry = recs["country"].to_numpy()[s1_rows]
li, ri, lab = a["li"].to_numpy().astype(np.int64), a["ri"].to_numpy().astype(np.int64), a["label"].to_numpy()

ad = "address" if "address" in recs.columns else "business_address"
rows = np.unique(np.concatenate([li, ri]))
nums = pd.Series(recs[ad].to_numpy()[rows]).astype(str).str.findall(r"\d+")
nums = [frozenset(x.lstrip("0") or "0" for x in v) for v in nums]
pos = np.full(n, -1, np.int64)
pos[rows] = np.arange(len(rows))
nl, nr = [nums[i] for i in pos[li]], [nums[i] for i in pos[ri]]
conf = np.fromiter((bool(x) and bool(y) and not (x & y) for x, y in zip(nl, nr)), bool, len(li))
del nl, nr, nums, pos, rows
print(f"pairs {len(li)}, conflict share {conf.mean():.4f}, label rate among conflicts {lab[conf].mean():.4f}", flush=True)

pa = a["oof"].to_numpy().astype(np.float64)
pb = others[0]["oof"].to_numpy().astype(np.float64)
m = (pa + pb) / 2
pc = others[1]["oof"].to_numpy().astype(np.float64) if len(others) > 1 else None
del a, others
var = {"T18": lambda: pa, "T28": lambda: pb, "mean": lambda: m, "0.25/0.75": lambda: 0.25 * pa + 0.75 * pb,
       "min": lambda: np.minimum(pa, pb), "guard(mean,min)": lambda: np.where(conf, np.minimum(pa, pb), m)}
if pc is not None:
    var["mean3"] = lambda: (pa + pb + pc) / 3
    var["mean(T18,T27)"] = lambda: (pa + pc) / 2
for name, fn in var.items():
    p = fn()
    out = []
    for t in (0.7, 0.8, 0.9):
        f, keep = evaluate_dense(li, ri, lab, p, meta, t)
        out.append(f"@{t}: {f.mean():.5f} (" + ", ".join(f"{c} {f[ctry == c].mean():.5f}" for c in sorted(set(ctry))) +
                   f", prec {lab[keep].mean():.5f}, n {keep.sum()})")
    print(f"{name:16s}", " | ".join(out), flush=True)
