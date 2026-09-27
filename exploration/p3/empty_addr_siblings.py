"""p3 (Sun 27 Sep, gap analysis): can the address-less S2/S3 copies be resolved through the entity's OTHER copies?

For every true pair whose S2/S3 record has an empty address:
  * does the same S1 have another true copy WITH an address, and how similar are the two copies' names
    (exact normalized name / same token set)?
  * how ambiguous is the address-less copy's name among S1 (how many S1 in the same country have the exact name)?
Also a structure check: do matched S2/S3 row positions / id numbers correlate with the S1's (generator leakage)?
Output: exploration/p3/empty_addr_siblings_out.md
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
D = Path(os.environ.get("BER_DATA_DIR", REPO / "data/dataset")) / "train"
rd = lambda f, cols=None: pd.read_csv(D / f, sep="\t", dtype=str, keep_default_na=False, na_filter=False, usecols=cols)
norm = lambda s: re.sub(r"[^0-9a-z]+", " ", s.lower()).strip()

s1 = rd("train_source1.tsv")
r = pd.concat([rd("train_source2.tsv"), rd("train_source3.tsv")], ignore_index=True)
gt = rd("train_ground_truth.tsv")
gt.columns = ["s1", "m"]
pairs = gt.assign(c=gt.m.str.split(",")).explode("c")
pairs = pairs[pairs.c != ""][["s1", "c"]]
rpos = pd.Series(np.arange(len(r)), index=r.entity_id)
r["name_n"] = r.business_name.map(norm)
r["is_empty"] = r.business_address.str.strip() == ""
p = pairs.assign(ri=rpos.reindex(pairs.c).to_numpy())
p = p[~np.isnan(p.ri)].astype({"ri": int})
p["empty_addr"] = r["is_empty"].to_numpy()[p.ri]
p["name_n"] = r.name_n.to_numpy()[p.ri]
lines = [f"# Address-less copies and their sibling copies (train) — exploration/p3/empty_addr_siblings.py\n",
         f"- true pairs {len(p):,}; with empty right address {int(p.empty_addr.sum()):,} ({p.empty_addr.mean():.4f})"]
e = p[p.empty_addr]
full = p[~p.empty_addr].groupby("s1")["name_n"].apply(set)
has_other = e.s1.map(lambda s: s in full.index)
same_name = [n in full.get(s, ()) for s, n in zip(e.s1, e.name_n)]
tok = lambda n: frozenset(n.split())
same_tok = [any(tok(n) == tok(o) for o in full.get(s, ())) for s, n in zip(e.s1, e.name_n)]
lines += [f"- empty-address copies whose S1 has another copy WITH address: {has_other.mean():.4f}",
          f"- … and that copy has the identical normalized name: {np.mean(same_name):.4f}; same token set: {np.mean(same_tok):.4f}"]
s1n = s1.assign(name_n=s1.business_name.map(norm))
cnt = s1n.groupby(["country", "name_n"]).size()
country = s1.set_index("entity_id").country
amb = [cnt.get((country[s], n), 0) for s, n in zip(e.s1, e.name_n)]
amb = np.array(amb)
lines += [f"- exact-name S1 count for the empty-address copy's name: 0 → {np.mean(amb == 0):.4f}, 1 → {np.mean(amb == 1):.4f}, "
          f"≥ 2 → {np.mean(amb >= 2):.4f}"]
# generator structure: id numbers / row order
s1pos = pd.Series(np.arange(len(s1)), index=s1.entity_id)
a = s1pos.reindex(p.s1).to_numpy().astype(float)
b = p.ri.to_numpy().astype(float)
sid = p.s1.str[3:].astype(float).to_numpy()
cid = p.c.str[3:].astype(float).to_numpy()
lines += [f"- Spearman(S1 row, right row) {pd.Series(a).corr(pd.Series(b), method='spearman'):.4f}; "
          f"Spearman(S1 id number, right id number) {pd.Series(sid).corr(pd.Series(cid), method='spearman'):.4f}"]
out = REPO / "exploration/p3/empty_addr_siblings_out.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
