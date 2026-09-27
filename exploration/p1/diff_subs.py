"""Where do two submissions disagree, per country? Counts of changed S1 by kind, and examples (p1, Sun 27 Sep).

    python exploration/p1/diff_subs.py <old run dir> <new run dir> [country for examples] [n]
kinds per S1: added-only (new predicts a superset), removed-only, swapped (both add and remove).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src"))
from ber.config import env
from ber.data import load_part

a, b = Path(sys.argv[1]), Path(sys.argv[2])
cex = sys.argv[3] if len(sys.argv) > 3 else None
n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
d = load_part(env("BER_DATA_DIR"), "test", cache_dir=Path(env("BER_ARTIFACTS")) / "cache")
r = d.records.set_index("entity_id")
nm = "name" if "name" in r.columns else "business_name"
ad = "address" if "address" in r.columns else "business_address"


def load(p):
    s = pd.read_csv(p / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    return {k: set(x for x in v.split(",") if x) for k, v in zip(s.iloc[:, 0], s.iloc[:, 1])}


A, B = load(a), load(b)
rows = []
for s1 in A:
    x, y = A[s1], B.get(s1, set())
    if x != y:
        kind = "added_only" if x < y else "removed_only" if y < x else "swapped"
        rows.append((s1, r.at[s1, "country"], kind, len(y - x), len(x - y), len(y - x & {c for c in y if not str(r.at[c, ad]).strip()})))
df = pd.DataFrame(rows, columns=["s1", "country", "kind", "n_added", "n_removed", "n_added_empty_addr"])
tot = pd.Series({s: r.at[s, "country"] for s in A}).value_counts()
t = df.groupby(["country", "kind"]).agg(s1=("s1", "size"), added=("n_added", "sum"), removed=("n_removed", "sum"),
                                         added_empty=("n_added_empty_addr", "sum"))
t["s1_share"] = (t["s1"] / tot.reindex(t.index.get_level_values(0)).to_numpy()).round(4)
print(t.to_string())
if cex:
    rng = np.random.default_rng(0)
    ex = df[df["country"] == cex]
    for s1 in ex["s1"].to_numpy()[rng.choice(len(ex), min(n, len(ex)), replace=False)]:
        print(f"\n{s1} | {r.at[s1, nm]} | {r.at[s1, ad]}")
        for c in sorted(A[s1] | B[s1]):
            tag = "both" if c in A[s1] and c in B[s1] else "OLD " if c in A[s1] else "NEW "
            print(f"   {tag} {c[:2]} | {r.at[c, nm]} | {r.at[c, ad]}")
