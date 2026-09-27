"""Dense-CV macro F0.5 of weighted blends of OOF runs, overall / per country / per fold (p1, Sun 27 Sep, T31 bagging).

    python exploration/p1/oof_bag.py <run A (T18)> <run B (T28)> <run A1 (T31_a1)>

Gate for T31_bag (fixed before the run): 0.25 A + 0.5 B + 0.25 A1 must beat the final 0.5 A + 0.5 B (0.98748 @0.8) by
>= 0.0002 at threshold 0.8, with both folds up.
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
runs = sys.argv[1:4]
a = pd.read_parquet(art / "dense_cv" / "oof" / f"{runs[0]}.parquet", columns=["li", "ri", "label", "fold", "oof"])
P = [a["oof"].to_numpy().astype(np.float64)]
for r in runs[1:]:
    o = pd.read_parquet(art / "dense_cv" / "oof" / f"{r}.parquet", columns=["li", "ri", "oof"])
    assert (a["li"].to_numpy() == o["li"].to_numpy()).all() and (a["ri"].to_numpy() == o["ri"].to_numpy()).all()
    P.append(o["oof"].to_numpy().astype(np.float64))
    del o
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
fold_of_s1 = np.full(n, -1, np.int8)
fold_of_s1[li] = a["fold"].to_numpy()
fs = fold_of_s1[s1_rows]                                   # S1 without candidates: -1 (identical in every blend)
del a
A, B, A1 = P
blends = {"A (T18)": A, "A1 (seed 1)": A1, "mean(A, A1)": (A + A1) / 2, "FINAL 0.5A+0.5B": (A + B) / 2,
          "0.25A+0.5B+0.25A1": 0.25 * A + 0.5 * B + 0.25 * A1, "mean(A, B, A1)": (A + B + A1) / 3}
for name, p in blends.items():
    out = []
    for t in (0.7, 0.8, 0.9):
        f, keep = evaluate_dense(li, ri, lab, p, meta, t)
        extra = ""
        if t == 0.8:
            extra = ", folds " + "/".join(f"{f[fs == k].mean():.5f}" for k in (0, 1))
        out.append(f"@{t}: {f.mean():.5f} (" + ", ".join(f"{c} {f[ctry == c].mean():.5f}" for c in sorted(set(ctry))) +
                   f", prec {lab[keep].mean():.5f}{extra})")
    print(f"{name:20s}", " | ".join(out), flush=True)
