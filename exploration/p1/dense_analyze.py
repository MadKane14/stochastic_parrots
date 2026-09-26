"""Error analysis of a dense-CV run (p1): where does macro F0.5 go at test-like density?

    nice -n 10 python exploration/p1/dense_analyze.py <dense_cv run name> [--threshold 0.5]

For every train S1 entity, errors of the p1_assign rule on the out-of-fold scores are typed:
  FP: on a singleton / extra wrong match next to right ones / all wrong; and whether the wrongly matched S2/S3
      record belongs to ANOTHER S1 entity (ownership error: a better assignment could fix it) or to none
      (unmatched distractor: only the model can reject it); plus a "sibling" flag (same first address number,
      names share their first token).
  FN: not a candidate (blocking) / lost to another S1 (exclusivity) / below threshold.
Loss attribution: how much macro F0.5 each entity category loses. Appends to exploration/p1/dense_analysis.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code/business_entity_resolution/src"))
sys.path.insert(0, str(REPO / "exploration/p1"))
from ber.config import env  # noqa: E402
from ber.data import load_part  # noqa: E402
from dense_cv import fbeta_vec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n_examples", type=int, default=10)
    a = ap.parse_args()
    art = Path(env("BER_ARTIFACTS"))
    rec = json.loads((REPO / "exploration/p1/dense_cv" / f"{a.run}.json").read_text())
    oof = pd.read_parquet(art / "dense_cv" / "oof" / f"{a.run}.parquet")
    full = load_part(env("BER_DATA_DIR"), "train", cache_dir=art / "cache")
    recs = full.records   # same row order as the normalized records used by dense_cv (normalizer keeps order)
    n = len(recs)
    ids = recs["entity_id"].to_numpy()
    idx = pd.Index(ids)
    li, ri = oof["li"].to_numpy().astype(np.int64), oof["ri"].to_numpy().astype(np.int64)
    y, p = oof["label"].to_numpy().astype(bool), oof["oof"].to_numpy()

    owner = np.full(n, -1, np.int64)                       # S1 row that truly owns each S2/S3 row
    g_s = [s for s in full.gt for _ in full.gt[s]]
    g_c = [c for s in full.gt for c in full.gt[s]]
    owner[idx.get_indexer(g_c)] = idx.get_indexer(g_s)
    ntrue = np.bincount(idx.get_indexer(g_s), minlength=n)

    order = np.lexsort((-p, ri))
    first = np.r_[True, ri[order][1:] != ri[order][:-1]]
    best_s1_of_rec = np.full(n, -1, np.int64); best_s1_of_rec[ri[order[first]]] = li[order[first]]
    own = order[first]
    keep = own[p[own] >= a.threshold]
    pred = np.zeros(len(li), bool); pred[keep] = True
    tp = np.bincount(li[keep], weights=y[keep], minlength=n)
    npred = np.bincount(li[keep], minlength=n)
    s1_rows = np.flatnonzero((recs["source"] == 1).to_numpy())
    f = fbeta_vec(tp[s1_rows], npred[s1_rows], ntrue[s1_rows])

    # --- FP typing ---
    fp = np.flatnonzero(pred & ~y)
    s_fp = li[fp]
    fp_kind = np.where(ntrue[s_fp] == 0, "on singleton", np.where(tp[s_fp] > 0, "extra, entity also has right ones",
                                                                    "all predictions wrong"))
    fp_owner = np.where(owner[ri[fp]] >= 0, "record owned by another S1", "record unmatched (distractor)")
    # was the true owner even a candidate of this record? (then exclusivity picked the wrong S1)
    cand_key = pd.Index(li * n + ri)
    own_is_cand = np.array([o >= 0 and (o * n + r) in cand_key for o, r in zip(owner[ri[fp]], ri[fp])])
    # --- FN typing ---
    in_cand = np.zeros(n, np.int64)
    np.add.at(in_cand, li[y], 1)
    fn_blocking = ntrue - in_cand                             # per S1: true pairs never proposed
    fn = np.flatnonzero(y & ~pred)
    fn_kind = np.where(best_s1_of_rec[ri[fn]] != li[fn], "lost to another S1 (exclusivity)", "below threshold")

    lines = [f"\n## {a.run} — threshold {a.threshold} (dense CV F0.5 {rec['metrics']['f05']})\n",
             f"- recomputed macro F0.5 {f.mean():.5f}; S1 {len(s1_rows):,}; predicted pairs {len(keep):,}; "
             f"FP {len(fp):,}; FN in candidates {len(fn):,}; FN blocking {int(fn_blocking[s1_rows].sum()):,}", ""]
    t = pd.crosstab(pd.Series(fp_kind, name="FP type"), pd.Series(fp_owner, name="record"), margins=True)
    lines += ["FP by type × ownership of the wrongly matched record:", "```", t.to_string(), "```",
              f"- FPs whose record's true owner S1 was ALSO a candidate of that record: {own_is_cand.sum():,} "
              f"({own_is_cand.mean():.1%} of FPs)", ""]
    lines += ["FN inside candidates:", "```", pd.Series(fn_kind).value_counts().to_string(), "```", ""]
    # loss attribution by entity category
    single = ntrue[s1_rows] == 0
    has_fp = np.bincount(li[fp], minlength=n)[s1_rows] > 0
    has_fnb = fn_blocking[s1_rows] > 0
    cat = np.where(single, "singleton", np.where(has_fp & has_fnb, "non-singleton: FP + blocking miss",
                   np.where(has_fp, "non-singleton: FP", np.where(has_fnb, "non-singleton: blocking miss only",
                                                                 "non-singleton: other"))))
    loss = pd.DataFrame({"cat": cat, "loss": 1 - f}).groupby("cat")["loss"].agg(["size", "sum"])
    loss["macro_loss"] = (loss["sum"] / len(s1_rows)).round(5)
    lines += ["Macro-F0.5 loss by entity category:", "```", loss[["size", "macro_loss"]].to_string(), "```", ""]
    # by country
    ctry = recs["country"].to_numpy()[s1_rows]
    lines += ["By country: " + json.dumps({c: round(float(f[ctry == c].mean()), 5) for c in sorted(set(ctry))}), ""]

    R = recs.set_index("entity_id")
    rng = np.random.default_rng(0)
    def ex(rows, title, pi):
        out = [f"#### {title}"]
        for j in rng.permutation(len(rows))[: a.n_examples]:
            i = rows[j]
            L, Rr = R.loc[ids[li[i]]], R.loc[ids[ri[i]]]
            o = owner[ri[i]]
            extra = f" | true owner: `{R.loc[ids[o]].business_name}` | `{R.loc[ids[o]].business_address}`" if o >= 0 and o != li[i] else ""
            out.append(f"- p={pi[i]:.2f} `{L.business_name}` | `{L.business_address}`  ⇄  `{Rr.business_name}` | "
                       f"`{Rr.business_address}` [{ids[ri[i]][:2]}]{extra}")
        return out
    for k in np.unique(fp_kind):
        for o in np.unique(fp_owner):
            rows = fp[(fp_kind == k) & (fp_owner == o)]
            if len(rows):
                lines += ex(rows, f"FP {k} — {o} ({len(rows):,})", p)
    for k in np.unique(fn_kind):
        rows = fn[fn_kind == k]
        lines += ex(rows, f"FN {k} ({len(rows):,})", p)
    out = REPO / "exploration/p1/dense_analysis.md"
    if not out.exists():
        out.write_text("# Dense-CV error analysis (p1) — generated by exploration/p1/dense_analyze.py\n")
    with open(out, "a") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
