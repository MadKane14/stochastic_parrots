"""Name-word document frequency among S1 names per country (p1, Sun 27 Sep): how many distinct words exceed the 1%
"boilerplate" cut used for core names (p1_match group_idf common_frac=0.01), and what share of S1 names lose all but
<= 1 word to that cut. Label-free; raw names lowercased, split on non-alphanumerics."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src"))
from ber.config import env
from ber.data import load_part

art = Path(env("BER_ARTIFACTS"))
for part in ("train", "test"):
    r = load_part(env("BER_DATA_DIR"), part, cache_dir=art / "cache").records
    nm = "name" if "name" in r.columns else "business_name"
    s1 = r[r["source"] == 1]
    for c, g in s1.groupby("country"):
        toks = g[nm].str.lower().str.findall(r"[^\W_]+").map(set)
        df = pd.Series([t for s in toks for t in s]).value_counts() / len(g)
        common = set(df.index[df > 0.01])
        core_len = toks.map(lambda s: len(s - common))
        print(f"{part} {c}: S1 {len(g)}, vocab {len(df)}, words >1% df: {len(common)}, "
              f"S1 names with <=1 core word: {np.mean(core_len <= 1):.3f}, 0 core words: {np.mean(core_len == 0):.3f}")
        print("   top:", ", ".join(f"{t} {v:.3f}" for t, v in df.head(25).items()))
