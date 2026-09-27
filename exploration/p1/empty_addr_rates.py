"""Empty-address S2/S3 records by country, train vs test, and how often we assign them (p1, Sun 27 Sep).

Train: share among all / GT-matched right records and GT true-pair share. Test: share among right records and among
records assigned by a submission (argv[1] = submission dir).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src"))
from ber.config import env
from ber.data import load_part

art = Path(env("BER_ARTIFACTS"))
sub = Path(sys.argv[1])
for part in ("train", "test"):
    d = load_part(env("BER_DATA_DIR"), part, cache_dir=art / "cache")
    r = d.records
    ad = "address" if "address" in r.columns else "business_address"
    right = r[r["source"] != 1].copy()
    right["empty"] = right[ad].astype(str).str.strip().eq("")
    if d.gt:
        m = set(x for v in d.gt.values() for x in v)
        right["grp"] = np.where(right["entity_id"].isin(m), "matched(GT)", "unmatched(GT)")
    else:
        s = pd.read_csv(sub / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
        m = set(x for v in s.iloc[:, 1] for x in v.split(",") if x)
        right["grp"] = np.where(right["entity_id"].isin(m), "assigned", "unassigned")
    t = right.groupby(["country", "grp"])["empty"].agg(["size", "mean", "sum"])
    t["empty_per_s1"] = t["sum"] / r[r["source"] == 1].groupby("country").size().reindex(t.index.get_level_values(0)).to_numpy()
    print(part); print(t.round(4).to_string())
    print(right.groupby("country")["empty"].mean().round(4).to_string(), "\n", flush=True)
