"""Where the remaining in-country dense-CV loss sits, for the #9-equivalent OOF (mean of T18 and T28) (p1, Sun 27 Sep).

    python exploration/p1/loss_breakdown.py <run T18> <run T28> [--threshold 0.8]

Entity groups (first match wins): singleton / has an empty-address true record / blocking miss / false positive /
FN inside candidates / perfect; plus the ceiling gain of perfect blocking, zero false positives and finding every
in-candidate true pair (each alone).
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ber.config import env
from ber.data import load_part
from dense_cv import evaluate_dense, fbeta_vec

ap = argparse.ArgumentParser()
ap.add_argument("run_a")
ap.add_argument("run_b")
ap.add_argument("--threshold", type=float, default=0.8)
args = ap.parse_args()
art = Path(env("BER_ARTIFACTS"))
a = pd.read_parquet(art / "dense_cv/oof" / f"{args.run_a}.parquet", columns=["li", "ri", "label", "oof"])
b = pd.read_parquet(art / "dense_cv/oof" / f"{args.run_b}.parquet", columns=["oof"])
full = load_part(env("BER_DATA_DIR"), "train", cache_dir=art / "cache")
recs = full.records
n = len(recs)
idx = pd.Index(recs["entity_id"].to_numpy())
ntrue = np.zeros(n, np.int64)
np.add.at(ntrue, idx.get_indexer(list(full.gt)), [len(v) for v in full.gt.values()])
rows = np.flatnonzero((recs["source"] == 1).to_numpy())
meta = {"n": n, "s1_rows": rows, "ntrue": ntrue}
li, ri = a["li"].to_numpy().astype(np.int64), a["ri"].to_numpy().astype(np.int64)
lab = a["label"].to_numpy().astype(np.float64)
p = (a["oof"].to_numpy().astype(np.float64) + b["oof"].to_numpy().astype(np.float64)) / 2
del b
f, keep = evaluate_dense(li, ri, lab, p, meta, args.threshold)
tp = np.bincount(li[keep], weights=lab[keep], minlength=n)[rows]
npred = np.bincount(li[keep], minlength=n)[rows]
incand = np.bincount(li, weights=lab, minlength=n)[rows]
nt = ntrue[rows]
miss, fp, fnc = nt - incand, npred - tp, incand - tp
print(f"OOF mean(A,B) F0.5 @{args.threshold}: {f.mean():.5f}  loss {1 - f.mean():.5f}  S1 {len(rows):,}")
print(f"true pairs {nt.sum():,}; not in candidates {miss.sum():,.0f} ({miss.sum() / nt.sum():.4f}); "
      f"FN inside candidates {fnc.sum():,.0f}; FP {fp.sum():,.0f}")
print(f"ceiling gain if: blocking perfect {fbeta_vec(tp + miss, npred + miss, nt).mean() - f.mean():.5f} | "
      f"no false positives {fbeta_vec(tp, tp, nt).mean() - f.mean():.5f} | "
      f"every in-candidate true pair found {fbeta_vec(incand, npred + fnc, nt).mean() - f.mean():.5f}")

ad = "address" if "address" in recs.columns else "business_address"
empty = recs[ad].astype(str).str.strip().eq("").to_numpy()
s1_of = np.repeat(idx.get_indexer(list(full.gt)), [len(v) for v in full.gt.values()])
r_of = idx.get_indexer([x for v in full.gt.values() for x in v])
has_empty = np.zeros(n, bool)
has_empty[s1_of[empty[r_of]]] = True
he = has_empty[rows]
single = nt == 0
cat = np.select([single, he, miss > 0, fp > 0, fnc > 0],
                ["singleton", "has empty-address true record", "blocking miss", "false positive", "FN in candidates"],
                "perfect")
loss = 1 - f
df = pd.DataFrame({"cat": cat, "loss": loss})
t = df.groupby("cat").agg(share_S1=("loss", lambda x: len(x) / len(df)), mean_F=("loss", lambda x: 1 - x.mean()),
                          loss_contrib=("loss", lambda x: x.sum() / len(df)))
t["share_of_loss"] = t["loss_contrib"] / loss.mean()
print(t.round(4).sort_values("loss_contrib", ascending=False).to_string())
m = he & ~single
print(f"empty-address S1: {m.mean():.4f} of S1, loss share {loss[m].sum() / loss.sum():.3f}; their true pairs missed by "
      f"blocking {miss[m].sum() / nt[m].sum():.4f}, FN in candidates {fnc[m].sum() / nt[m].sum():.4f}")
print("by country:", pd.Series(f, index=recs["country"].to_numpy()[rows]).groupby(level=0).mean().round(5).to_dict())
