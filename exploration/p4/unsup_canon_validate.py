"""T14b validation: the label-free fix (unsup_canon.py) fitted WITHOUT labels on one TRAIN country (as if unseen),
then scored WITH its true pairs: address token Jaccard of true S1-S2/S3 pairs vs random pairs, raw vs fixed.
Usage (repo root): python exploration/p4/unsup_canon_validate.py --country US --out exploration/p4/unsup_canon_validate_out.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "business_entity_resolution" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ber.normalize.basic import clean  # noqa: E402
from unsup_canon import apply, confident_pairs, mine_canon, side_skew_junk  # noqa: E402


def jacc(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-eval", type=int, default=50_000)
    a = ap.parse_args()
    t0 = time.time()
    d = REPO / "data" / "dataset" / "train"
    parts = []
    for s in (1, 2, 3):
        for ch in pd.read_csv(d / f"train_source{s}.tsv", sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                              chunksize=1_000_000):
            parts.append(ch[ch["country"] == a.country].assign(source=s))
    recs = pd.concat(parts, ignore_index=True)
    is_s1 = (recs["source"] == 1).to_numpy()
    atok = [x.split() for x in recs["business_address"].map(clean)]
    names = recs["business_name"].map(clean).tolist()
    # --- label-free fit (no ground truth touched) ---
    cs, cr = confident_pairs(atok, names, is_s1)
    cmap = mine_canon([atok[i] for i in cs], [atok[i] for i in cr])
    canon_all = [[cmap.get(t, t) for t in x] for x in atok]
    junk = side_skew_junk([x for x, s1 in zip(canon_all, is_s1) if s1], [x for x, s1 in zip(canon_all, is_s1) if not s1])
    fit_s = time.time() - t0
    # --- evaluation with labels ---
    row = pd.Series(np.arange(len(recs)), index=recs["entity_id"].to_numpy())
    gt = pd.read_csv(d / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    gt = gt[gt["source1_entity_id"].isin(row.index)]
    pairs = [(s, c.strip()) for s, lst in zip(gt["source1_entity_id"], gt["matched_entity_ids"]) for c in lst.split(",") if c.strip()]
    rng = np.random.default_rng(0)
    sel = rng.choice(len(pairs), min(a.n_eval, len(pairs)), replace=False)
    tp = [(row[pairs[i][0]], row[pairs[i][1]]) for i in sel if pairs[i][1] in row.index]
    rp = list(zip(rng.choice(np.flatnonzero(is_s1), len(tp)), rng.choice(np.flatnonzero(~is_s1), len(tp))))
    fx = lambda t: apply(t, cmap, junk)   # noqa: E731
    res = {k: (np.mean([jacc(atok[i], atok[j]) for i, j in P]), np.mean([jacc(fx(atok[i]), fx(atok[j])) for i, j in P]))
           for k, P in (("true", tp), ("random", rp))}
    conf_prec = np.mean([pairs_set_hit for pairs_set_hit in
                         (recs["entity_id"].iat[j] in set(gt_map.get(recs["entity_id"].iat[i], ())) for i, j in zip(cs[:20000], cr[:20000]))]) \
        if (gt_map := {s: set(x.strip() for x in l.split(",") if x.strip()) for s, l in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}) else float("nan")
    L = [f"\n## {a.country} (train, fitted label-free, scored with labels) — {dt.datetime.now():%Y-%m-%d %H:%M}",
         f"- records {len(recs):,}; confident pairs {len(cs):,} (label precision of the confident pairs, first 20k: {conf_prec:.4f}); fit {fit_s:.0f}s",
         f"- mined map ({len(cmap)}): {dict(sorted(cmap.items())[:40])}",
         f"- junk: {sorted(junk)}",
         "| pairs | address token Jaccard raw | with fix | change |", "|---|---|---|---|",
         f"| true pairs ({len(tp):,}) | {res['true'][0]:.4f} | {res['true'][1]:.4f} | {res['true'][1] - res['true'][0]:+.4f} |",
         f"| random pairs ({len(rp):,}) | {res['random'][0]:.4f} | {res['random'][1]:.4f} | {res['random'][1] - res['random'][0]:+.4f} |"]
    with open(a.out, "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
