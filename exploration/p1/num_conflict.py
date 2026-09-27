"""House-number conflicts among PREDICTED pairs by country (label-free check for unseen-country false merges; p1).

Numbers = digit runs of the address (leading zeros stripped). conflict = both sides have numbers and share none.
Train: GT true pairs (noise baseline) and dense-CV OOF predictions at thr (FP / TP split). Test: a submission's pairs.

    python exploration/p1/num_conflict.py <submission dir> <in-country dense_cv run> [thr]
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
from dense_decide import owned

art = Path(env("BER_ARTIFACTS"))
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
DIG = re.compile(r"\d+")


def numsets(addr: pd.Series) -> np.ndarray:
    return np.array([frozenset(x.lstrip("0") or "0" for x in DIG.findall(a)) for a in addr.astype(str)], dtype=object)


def rates(A, B):
    has = np.array([bool(a) and bool(b) for a, b in zip(A, B)])
    disj = np.array([bool(a) and bool(b) and not (a & b) for a, b in zip(A, B)])
    return has.mean(), disj.mean()


for part in ("train", "test"):
    d = load_part(env("BER_DATA_DIR"), part, cache_dir=art / "cache")
    r = d.records
    ad = "address" if "address" in r.columns else "business_address"
    ns = numsets(r[ad])
    idx = pd.Index(r["entity_id"].to_numpy())
    ctry = r["country"].to_numpy()
    rows = []
    if part == "train":
        o = pd.read_parquet(art / "dense_cv" / "oof" / f"{sys.argv[2]}.parquet")
        li, ri = o["li"].to_numpy().astype(np.int64), o["ri"].to_numpy().astype(np.int64)
        lab, p = o["label"].to_numpy(), o["oof"].to_numpy()
        own = owned(li, ri, p.astype(np.float64))
        k = own[p[own] >= thr]
        for name, sel in (("pred_TP", k[lab[k] == 1]), ("pred_FP", k[lab[k] == 0]), ("pred_all", k)):
            for c in sorted(set(ctry)):
                s = sel[ctry[li[sel]] == c]
                h, dj = rates(ns[li[s]], ns[ri[s]])
                rows.append({"part": part, "set": name, "country": c, "pairs": len(s), "both_have_num": round(h, 4), "conflict": round(dj, 4)})
    else:
        s = pd.read_csv(Path(sys.argv[1]) / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
        s.columns = ["s1", "ids"]
        x = s.assign(r=s["ids"].str.split(",")).explode("r")
        x = x[x["r"].fillna("") != ""]
        li, ri = idx.get_indexer(x["s1"]), idx.get_indexer(x["r"])
        for c in sorted(set(ctry)):
            sel = ctry[li] == c
            h, dj = rates(ns[li[sel]], ns[ri[sel]])
            rows.append({"part": part, "set": "pred_all", "country": c, "pairs": int(sel.sum()), "both_have_num": round(h, 4), "conflict": round(dj, 4)})
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
