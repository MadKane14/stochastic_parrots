"""House-number conflict rate of the pairs one submission adds / removes relative to another, per country (label-free;
p1, Sun 27 Sep). conflict = both addresses have numbers and share none (leading zeros stripped).

    python exploration/p1/diff_numconf.py <old run dir> <new run dir>
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src"))
from ber.config import env
from ber.data import load_part

DIG = re.compile(r"\d+")
r = load_part(env("BER_DATA_DIR"), "test", cache_dir=Path(env("BER_ARTIFACTS")) / "cache").records
ad = "address" if "address" in r.columns else "business_address"
nums = dict(zip(r["entity_id"], [frozenset(x.lstrip("0") or "0" for x in DIG.findall(a)) for a in r[ad].astype(str)]))
ctry = dict(zip(r["entity_id"], r["country"]))


def pairs(p):
    s = pd.read_csv(Path(p) / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    return {(a, c) for a, v in zip(s.iloc[:, 0], s.iloc[:, 1]) for c in v.split(",") if c}


A, B = pairs(sys.argv[1]), pairs(sys.argv[2])
rows = []
for name, S in (("added", B - A), ("removed", A - B), ("kept", A & B)):
    df = pd.DataFrame(list(S), columns=["s1", "r"])
    df["country"] = df["s1"].map(ctry)
    a = df["s1"].map(nums); b = df["r"].map(nums)
    df["has"] = [bool(x) and bool(y) for x, y in zip(a, b)]
    df["conf"] = [bool(x) and bool(y) and not (x & y) for x, y in zip(a, b)]
    for c, g in df.groupby("country"):
        rows.append({"set": name, "country": c, "pairs": len(g), "both_have_num": round(g["has"].mean(), 4),
                     "conflict": round(g["conf"].mean(), 4)})
print(pd.DataFrame(rows).to_string(index=False))
