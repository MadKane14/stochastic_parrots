"""What are the false merges of an unseen-country model? (p1, Sun 27 Sep) Profiles owned predictions p >= thr of a
LOCO OOF run vs the in-country OOF run: FP share by kind of right record, and prints examples.

    python exploration/p1/loco_fp.py <loco run> <in-country run> [thr] [n_examples]
kinds: empty address; right record is a true copy of ANOTHER S1 (sibling confusion); unmatched distractor.
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
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
nex = int(sys.argv[4]) if len(sys.argv) > 4 else 12
d = load_part(env("BER_DATA_DIR"), "train", cache_dir=art / "cache")
r = d.records
nm = "name" if "name" in r.columns else "business_name"
ad = "address" if "address" in r.columns else "business_address"
empty = r[ad].astype(str).str.strip().eq("").to_numpy()
m = set(x for v in d.gt.values() for x in v)
matched = r["entity_id"].isin(m).to_numpy()
ctry = r["country"].to_numpy()
rng = np.random.default_rng(0)
for run in sys.argv[1:3]:
    o = pd.read_parquet(art / "dense_cv" / "oof" / f"{run}.parquet")
    li, ri = o["li"].to_numpy().astype(np.int64), o["ri"].to_numpy().astype(np.int64)
    lab, p = o["label"].to_numpy(), o["oof"].to_numpy().astype(np.float64)
    own = owned(li, ri, p)
    k = own[p[own] >= thr]
    fp = k[lab[k] == 0]
    kind = np.where(empty[ri[fp]], "empty_addr", np.where(matched[ri[fp]], "copy_of_other_S1", "distractor"))
    print(f"\n## {run} thr {thr}: predicted {len(k)}, FP {len(fp)} ({len(fp) / len(k):.4f})")
    print(pd.crosstab(ctry[li[fp]], kind, margins=True).to_string())
    if "loco" in run:
        for i in rng.choice(fp, nex, replace=False):
            a, b = li[i], ri[i]
            print(f"  p={p[i]:.3f} [{ctry[a]}] S1 {r.iat[a, r.columns.get_loc(nm)]} | {r.iat[a, r.columns.get_loc(ad)]}\n"
                  f"          R  {r.iat[b, r.columns.get_loc(nm)]} | {r.iat[b, r.columns.get_loc(ad)]}  ({'copy of other' if matched[b] else 'distractor'})")
