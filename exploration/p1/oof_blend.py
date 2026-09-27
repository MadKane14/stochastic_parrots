"""Dense-CV macro F0.5 of a probability average of two OOF runs on the same pairs (p1, Sun 27 Sep).

    python exploration/p1/oof_blend.py <run A> <run B>
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
b = pd.read_parquet(art / "dense_cv" / "oof" / f"{sys.argv[2]}.parquet", columns=["li", "ri", "oof"])
assert (a["li"].to_numpy() == b["li"].to_numpy()).all() and (a["ri"].to_numpy() == b["ri"].to_numpy()).all()
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
fold = a["fold"].to_numpy()
pa, pb = a["oof"].to_numpy().astype(np.float64), b["oof"].to_numpy().astype(np.float64)
for name, p in (("A", pa), ("B", pb), ("mean(A,B)", (pa + pb) / 2)):
    out = []
    for t in (0.7, 0.8, 0.9):
        f, keep = evaluate_dense(li, ri, lab, p, meta, t)
        out.append(f"@{t}: {f.mean():.5f} (" + ", ".join(f"{c} {f[ctry == c].mean():.5f}" for c in sorted(set(ctry))) +
                   f", prec {lab[keep].mean():.5f})")
    print(name, " | ".join(out), flush=True)
