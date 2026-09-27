"""Empty-address S2/S3 records in dense-CV OOF decisions (p1, Sun 27 Sep): assigned rate (comparable to the test rate
from empty_addr_rates.py), correct / wrong owner, per country, at thresholds.

    python exploration/p1/empty_oof.py <dense_cv run name>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ber.config import env
from ber.data import load_part
from dense_decide import owned

art = Path(env("BER_ARTIFACTS"))
oof = pd.read_parquet(art / "dense_cv" / "oof" / f"{sys.argv[1]}.parquet")
d = load_part(env("BER_DATA_DIR"), "train", cache_dir=art / "cache")
r = d.records
ad = "address" if "address" in r.columns else "business_address"
empty = r[ad].astype(str).str.strip().eq("").to_numpy()
src = r["source"].to_numpy()
ctry = r["country"].to_numpy()
idx = pd.Index(r["entity_id"].to_numpy())
m = set(x for v in d.gt.values() for x in v)
is_matched = r["entity_id"].isin(m).to_numpy()
li, ri = oof["li"].to_numpy().astype(np.int64), oof["ri"].to_numpy().astype(np.int64)
lab, p = oof["label"].to_numpy(), oof["oof"].to_numpy().astype(np.float64)
own = owned(li, ri, p)
er = (src != 1) & empty
in_cand = np.zeros(len(r), bool); in_cand[ri] = True
true_in_cand = np.zeros(len(r), bool); true_in_cand[ri[lab == 1]] = True
for t in (0.5, 0.7, 0.8, 0.9):
    k = own[p[own] >= t]
    ass = np.zeros(len(r), bool); ass[ri[k]] = True
    ok = np.zeros(len(r), bool); ok[ri[k[lab[k] == 1]]] = True
    rows = []
    for c in sorted(set(ctry)):
        s = er & (ctry == c)
        rows.append({"thr": t, "country": c, "empty_right": int(s.sum()), "assigned": round(ass[s].mean(), 4),
                     "correct": round(ok[s].mean(), 4), "wrong_owner": round((ass & ~ok)[s].mean(), 4),
                     "true_in_cands": round(true_in_cand[s & is_matched].sum() / max((s & is_matched).sum(), 1), 4),
                     "matched_GT": round(is_matched[s].mean(), 4)})
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
