"""Print random S1 entities of one country with their pruned candidates and our decision (p1; eyeballing only).

    python exploration/p1/peek_country.py <submission dir> <country> <n> [seed]
Candidates marked * are in matching_results.tsv.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src"))
from ber.config import env
from ber.data import load_part

sub, country, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
d = load_part(env("BER_DATA_DIR"), "test", cache_dir=Path(env("BER_ARTIFACTS")) / "cache")
r = d.records.set_index("entity_id")
cand = pd.read_csv(sub / "candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
mat = pd.read_csv(sub / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
m = dict(zip(mat.iloc[:, 0], mat.iloc[:, 1]))
s1 = cand[cand.iloc[:, 0].map(r["country"]) == country].sample(n, random_state=seed)
nm = "name" if "name" in r.columns else "business_name"
ad = "address" if "address" in r.columns else "business_address"
for s, cs in zip(s1.iloc[:, 0], s1.iloc[:, 1]):
    got = set(x for x in m.get(s, "").split(",") if x)
    print(f"\n{s} | {r.at[s, nm]} | {r.at[s, ad]}")
    for c in [x for x in cs.split(",") if x]:
        print(f"   {'*' if c in got else ' '} {c[:2]} | {r.at[c, nm]} | {r.at[c, ad]}")
