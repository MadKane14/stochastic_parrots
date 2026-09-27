"""Are the extra test S2/S3 records orphans (copies of entities absent from S1)? (p1, Sun 27 Sep)

Train has 4.68 S2/S3 records per S1, test 5.5-5.8 with the same predicted matches per S1. Crude label-free keys:
  nm   = country + lowercase alphanumeric name
Per right record: s1_nm = some S1 of the same country has the same nm key; oth_nm = a right record of the OTHER source has it.
Orphan clusters (copies of a missing entity) → many right records with oth_nm but no s1_nm.
Train rows split by GT (matched / unmatched); test rows split by submission #7b (assigned / not).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src"))
from ber.config import env
from ber.data import load_part

art = Path(env("BER_ARTIFACTS"))
sub = art / "submissions" / "T17_thr080" / "matching_results.tsv"


def keys(r):
    nm = r["name"] if "name" in r else r["business_name"]
    return r["country"].astype(str) + "|" + nm.astype(str).str.lower().str.replace(r"[^0-9a-zऀ-෿]", "", regex=True)


for part in ("train", "test"):
    d = load_part(env("BER_DATA_DIR"), part, cache_dir=art / "cache")
    r = d.records
    k = keys(r)
    src = r["source"].to_numpy()
    s1k = set(k[src == 1])
    right = r[src != 1].copy()
    right["k"] = k[src != 1].to_numpy()
    right["s1_nm"] = right["k"].isin(s1k)
    cnt = right.groupby(["k", "source"]).size().unstack(fill_value=0)
    other = np.where(right["source"].to_numpy() == 2, cnt.reindex(right["k"])[3].to_numpy(), cnt.reindex(right["k"])[2].to_numpy())
    right["oth_nm"] = other > 0
    if d.gt:
        m = set(x for v in d.gt.values() for x in v)
        right["grp"] = np.where(right["entity_id"].isin(m), "matched(GT)", "unmatched(GT)")
    else:
        s = pd.read_csv(sub, sep="\t", dtype=str, keep_default_na=False)
        m = set(x for v in s.iloc[:, 1] for x in v.split(",") if x)
        right["grp"] = np.where(right["entity_id"].isin(m), "assigned(#7b)", "unassigned(#7b)")
    right["orphan_like"] = right["oth_nm"] & ~right["s1_nm"]
    t = right.groupby(["country", "grp"]).agg(n=("k", "size"), s1_nm=("s1_nm", "mean"), oth_nm=("oth_nm", "mean"),
                                              orphan_like=("orphan_like", "mean"))
    t["n_per_s1"] = t["n"] / r[src == 1].groupby("country").size().reindex(t.index.get_level_values(0)).to_numpy()
    print(part); print(t.round(4).to_string()); print(flush=True)
