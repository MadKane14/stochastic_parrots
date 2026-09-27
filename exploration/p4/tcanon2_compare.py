"""p4: p4_tcanon (in FINAL) vs p4_tcanon2 on one country's records — address token Jaccard of pairs, before any model.
France (test, no labels): label-free confident pairs (same (house number, address word) keys + name token_set_ratio >= 90
on basic-clean text, fixed independently of the normalizers) vs random pairs. India / US (train): 50k true pairs vs
50k random pairs. Usage: python exploration/p4/tcanon2_compare.py --country France --part test --out <md>"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code/business_entity_resolution/src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from ber.data import Dataset
from ber.normalize.basic import clean
from ber.normalize.p4_tcanon import TargetCanonNormalizer, confident_pairs
from ber.normalize.p4_tcanon2 import TargetCanon2Normalizer
P = {'legal_tokens': ['inc','llc','ltd','pvt','private','limited','corp','corporation','co','company','plc','llp'], 'mode': 'replace',
     'min_support': 3, 'min_precision': 0.5, 'fold_scripts': False, 'map_address': True, 'web_names': True, 'join_initials': True}

def load(part, country):
    d = REPO / "data/dataset" / part; parts = []
    for s in (1, 2, 3):
        for ch in pd.read_csv(d / f"{part}_source{s}.tsv", sep="\t", dtype=str, keep_default_na=False, na_filter=False, chunksize=1_000_000):
            parts.append(ch[ch["country"] == country].assign(source=s))
    return pd.concat(parts, ignore_index=True)

def wjacc(a, b, idf):
    sa, sb = set(a.split()), set(b.split()); u = sa | sb
    return sum(idf.get(t, 0) for t in sa & sb) / max(1e-9, sum(idf.get(t, 0) for t in u)) if u else 0.0

def jacc(a, b):
    sa, sb = set(a.split()), set(b.split()); return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0

ap = argparse.ArgumentParser(); ap.add_argument("--country"); ap.add_argument("--part"); ap.add_argument("--out"); a = ap.parse_args()
recs = load(a.part, a.country); is_s1 = (recs["source"] == 1).to_numpy(); rng = np.random.default_rng(0)
if a.part == "test":
    at = [clean(x).split() for x in recs["business_address"]]; nm = recs["business_name"].map(clean).tolist()
    s, r = confident_pairs(at, nm, is_s1); kind = f"confident pairs ({len(s):,}, label-free)"
else:
    row = pd.Series(np.arange(len(recs)), index=recs["entity_id"].to_numpy())
    gt = pd.read_csv(REPO / "data/dataset/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    pr = [(x, c.strip()) for x, l in zip(gt["source1_entity_id"], gt["matched_entity_ids"]) if x in row.index for c in l.split(",") if c.strip()]
    pr = [pr[i] for i in rng.choice(len(pr), min(50000, len(pr)), replace=False)]
    s, r = row[[p[0] for p in pr]].to_numpy(), row[[p[1] for p in pr]].to_numpy(); kind = f"true pairs ({len(s):,})"
rs, rr = rng.choice(np.flatnonzero(is_s1), 50000), rng.choice(np.flatnonzero(~is_s1), 50000)
tiny = Dataset("tiny", recs.iloc[:2000].reset_index(drop=True), None)
res, info = {}, {}
for name, cls in (("p4_tcanon (FINAL)", TargetCanonNormalizer), ("p4_tcanon2", TargetCanon2Normalizer)):
    n = cls(**P); n.fit(tiny); ad = n.transform(recs)["addr_norm"].to_numpy(object)
    from collections import Counter
    df = Counter(t for x in ad for t in set(x.split())); N = len(ad); idf = {t: np.log(N / c) for t, c in df.items()}
    res[name] = (np.mean([jacc(ad[i], ad[j]) for i, j in zip(s, r)]), np.mean([jacc(ad[i], ad[j]) for i, j in zip(rs, rr)]),
                 np.mean([wjacc(ad[i], ad[j], idf) for i, j in zip(s, r)]), np.mean([wjacc(ad[i], ad[j], idf) for i, j in zip(rs, rr)]))
    info[name] = n.tc_stats.get("0", {}); del ad, n
L = [f"\n## {a.country} ({a.part}) — {len(recs):,} records", "| normalizer | " + kind + " Jaccard | random Jaccard | " + kind + " IDF-weighted | random IDF-weighted |", "|---|---|---|---|---|"]
L += [f"| {k} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | {v[3]:.4f} |" for k, v in res.items()]
L += [f"- p4_tcanon2 segment junk: {info['p4_tcanon2'].get('segment_junk')}", f"- p4_tcanon2 token junk: {info['p4_tcanon2'].get('junk')}; map size {info['p4_tcanon2'].get('map_size')}"]
open(a.out, "a").write("\n".join(L) + "\n"); print("\n".join(L))
