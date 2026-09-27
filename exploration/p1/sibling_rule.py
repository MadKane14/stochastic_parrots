"""Cross-copy rule on the #9-equivalent OOF (p1, Sun 27 Sep; idea from reports/p3/path_to_099.md §5).

    python exploration/p1/sibling_rule.py <run A (T18)> <run B (T28)>

After the normal decision (one owner per S2/S3 record, threshold 0.8), an UNASSIGNED right record r joins S1 x when every
assigned right record with r's exact name key (lowercase alphanumerics) is owned by x, (x, r) is a candidate pair, and
its score >= floor. Label-free and country-agnostic. Variants: only address-less r vs every r; sibling with an address
required or not; floors. Reports dense-CV macro F0.5 vs the base decision and precision of the added pairs.
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ber.config import env
from ber.data import load_part
from dense_cv import evaluate_dense, fbeta_vec

art = Path(env("BER_ARTIFACTS"))
a = pd.read_parquet(art / "dense_cv/oof" / f"{sys.argv[1]}.parquet", columns=["li", "ri", "label", "oof"])
b = pd.read_parquet(art / "dense_cv/oof" / f"{sys.argv[2]}.parquet", columns=["oof"])
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
del a, b
f0, keep = evaluate_dense(li, ri, lab, p, meta, 0.8)
print(f"base (#9-equivalent) @0.8: {f0.mean():.5f}", flush=True)

nm = "name" if "name" in recs.columns else "business_name"
ad = "address" if "address" in recs.columns else "business_address"
key = recs[nm].astype(str).str.lower().str.replace(r"[^0-9a-z\u0080-￿]+", "", regex=True).to_numpy()
nk = pd.factorize(key)[0].astype(np.int64)
empty = recs[ad].astype(str).str.strip().eq("").to_numpy()
blank_name = (key == "")

owner = np.full(n, -1, np.int64)
owner[ri[keep]] = li[keep]
pair_key = li * n + ri
order = np.argsort(pair_key)
sk = pair_key[order]

tp0 = np.bincount(li[keep], weights=lab[keep], minlength=n)
np0 = np.bincount(li[keep], minlength=n).astype(np.float64)
R = np.unique(ri)
print(f"right records in candidates {len(R):,}; assigned {int((owner[R] >= 0).sum()):,}; "
      f"unassigned address-less {int(((owner[R] < 0) & empty[R]).sum()):,}", flush=True)

for need_sib_addr in (False, True):
    ar = R[(owner[R] >= 0) & ~blank_name[R]]
    if need_sib_addr:
        ar = ar[~empty[ar]]
    g = pd.DataFrame({"k": nk[ar], "o": owner[ar]}).groupby("k")["o"].agg(["nunique", "first"])
    uniq = g[g["nunique"] == 1]["first"]
    own_of_key = np.full(nk.max() + 1, -1, np.int64)
    own_of_key[uniq.index.to_numpy()] = uniq.to_numpy()
    for only_empty in (True, False):
        cand = R[(owner[R] < 0) & ~blank_name[R]]
        if only_empty:
            cand = cand[empty[cand]]
        x = own_of_key[nk[cand]]
        cand, x = cand[x >= 0], x[x >= 0]
        q = x * n + cand
        pos = np.searchsorted(sk, q)
        ok = (pos < len(sk)) & (sk[np.minimum(pos, len(sk) - 1)] == q)
        pi = order[pos[ok]]                                   # pair index of the proposed (x, r)
        for floor in (0.0, 0.05, 0.2, 0.5):
            sel = pi[p[pi] >= floor]
            tp = tp0 + np.bincount(li[sel], weights=lab[sel], minlength=n)
            npd = np0 + np.bincount(li[sel], minlength=n)
            f = fbeta_vec(tp[rows], npd[rows], ntrue[rows])
            print(f"sibling_addr={need_sib_addr!s:5} only_empty={only_empty!s:5} floor={floor:4}: added {len(sel):7,} "
                  f"(precision {lab[sel].mean() if len(sel) else 0:.3f})  F0.5 {f.mean():.5f}  Δ {f.mean() - f0.mean():+.5f}",
                  flush=True)
