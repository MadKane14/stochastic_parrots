"""Compare two submission folders pair by pair (p1; reproduction check and #7 vs #8 diffs).

    python exploration/p1/compare_subs.py <run_dir_a> <run_dir_b>

For matching_results.tsv and candidate_pairs.tsv: sha256 of each file, S1 rows, pairs, pairs in both, and the macro
F0.5 of b measured against a (1.0 = identical decisions).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def pairs(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    df.columns = ["s1", "ids"]
    x = df.assign(r=df["ids"].str.split(",")).explode("r")
    return df, x[x["r"].fillna("") != ""][["s1", "r"]]


def main():
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    print(f"a = {a}\nb = {b}")
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        da, pa = pairs(a / f)
        db, pb = pairs(b / f)
        both = pa.merge(pb, on=["s1", "r"])
        na = pa.groupby("s1").size().reindex(da["s1"], fill_value=0).to_numpy()
        nb = pb.groupby("s1").size().reindex(da["s1"], fill_value=0).to_numpy()
        tp = both.groupby("s1").size().reindex(da["s1"], fill_value=0).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            f05 = np.where((na == 0) & (nb == 0), 1.0, 1.25 * tp / (0.25 * na + nb))
        print(f"{f}: sha a {sha(a / f)} b {sha(b / f)} | S1 rows {len(da)} / {len(db)} | pairs {len(pa)} / {len(pb)} "
              f"| in both {len(both)} | F0.5(b vs a) {np.nan_to_num(f05).mean():.5f}")


if __name__ == "__main__":
    main()
