"""T03 (p1): threshold curve under the real decision rule + false-positive / false-negative breakdown of one run.

Reads $BER_ARTIFACTS/runs/<run_id>/scored_pairs.parquet (written by the harness) and rebuilds the run's evaluation
split to get the ground truth (incl. true pairs the blocker never proposed). Metric = the frozen ber.eval.metrics.

    nice -n 10 python exploration/p1/t03_analyze.py <run_id> [--out exploration/p1/t03_analysis.md]
NOTE: a threshold picked on this curve is tuned on the evaluated split → optimistic; confirm on another split.
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
from ber.config import env  # noqa: E402
from ber.data import load_part  # noqa: E402
from ber.deciders.p1_assign import AssignDecider  # noqa: E402
from ber.eval.harness import DEV_SEED, parse_split  # noqa: E402
from ber.eval.metrics import macro_fbeta, pairs_to_sets, per_entity_scores  # noqa: E402
from ber.eval.splits import make_split, subsample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--out", default=str(REPO / "exploration/p1/t03_analysis.md"))
    ap.add_argument("--n_examples", type=int, default=12)
    a = ap.parse_args()
    member = a.run_id.split("_")[1]
    rec = json.loads((REPO / "results" / member / f"{a.run_id}.json").read_text())
    art = Path(env("BER_ARTIFACTS"))
    pairs = pd.read_parquet(art / "runs" / a.run_id / "scored_pairs.parquet")
    full = load_part(env("BER_DATA_DIR"), "train", cache_dir=art / "cache")
    base, frac = parse_split(rec["split"])
    _, val = make_split(full, base)
    if frac < 1:
        val = subsample(val, frac, seed=DEV_SEED)
    s1_ids, gt = val.s1_ids, val.gt
    lines = [f"\n## {a.run_id} (split {rec['split']}, F0.5 {rec['metrics']['f05']})\n"]

    # ---- threshold curve with the real decider ----
    rows = []
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        for t1 in (None, 0.1, 0.2, 0.3):
            if t1 is not None and t1 >= t:
                continue
            m = AssignDecider(threshold=t, top1_min=t1).decide(None, pairs)
            f = per_entity_scores(pairs_to_sets(m, s1_ids), gt, s1_ids)
            rows.append({"threshold": t, "top1_min": t1, "F0.5": round(float(f.mean()), 5), "n_pred": len(m)})
    cur = pd.DataFrame(rows)
    lines += ["### Threshold curve (p1_assign: one owner per S2/S3 record)", "", cur.to_string(index=False), ""]

    # ---- error breakdown at the run's own decider params ----
    dp = rec["config"]["decider"]["params"]
    matches = AssignDecider(**dp).decide(None, pairs)
    pred = pairs_to_sets(matches, s1_ids)
    cands = pairs_to_sets(pairs, s1_ids)
    best_owner = pairs.sort_values("score", ascending=False).drop_duplicates("cand_id").set_index("cand_id")["s1_id"]
    score = {(s, c): v for s, c, v in zip(pairs["s1_id"], pairs["cand_id"], pairs["score"])}
    fp_rows, fn_rows = [], []
    for s in s1_ids:
        t, p = set(gt[s]), pred[s]
        for c in p - t:
            fp_rows.append({"s1": s, "cand": c, "score": score[(s, c)],
                            "type": "FP on singleton" if not t else ("FP, entity also has a right match" if p & t
                                                                     else "FP, all predictions wrong")})
        for c in t - p:
            if c not in cands[s]:
                kind = "FN: not a candidate (blocking)"
            elif best_owner.get(c) != s:
                kind = "FN: lost to another S1 (exclusivity)"
            else:
                kind = "FN: below threshold"
            fn_rows.append({"s1": s, "cand": c, "score": score.get((s, c), np.nan), "type": kind})
    fp, fn = pd.DataFrame(fp_rows), pd.DataFrame(fn_rows)
    n_true = sum(len(gt[s]) for s in s1_ids)
    lines += ["### Error breakdown at the run's decider " + json.dumps(dp), "",
              f"- predicted pairs {len(matches):,}; true pairs {n_true:,}; FP {len(fp):,}; FN {len(fn):,}",
              "", "FP by type:", (fp["type"].value_counts().to_string() if len(fp) else "none"), "",
              "FN by type:", (fn["type"].value_counts().to_string() if len(fn) else "none"), ""]
    # per-entity loss attribution: how much macro F0.5 each error type costs
    f_ent = per_entity_scores(pred, gt, s1_ids)
    single = np.array([len(gt[s]) == 0 for s in s1_ids])
    lines += [f"- mean F0.5 on singletons {f_ent[single].mean():.4f} (n={single.sum()}), "
              f"non-singletons {f_ent[~single].mean():.4f}; total loss {1 - f_ent.mean():.4f} = "
              f"{(1 - f_ent[single]).sum() / len(f_ent):.4f} from singletons + "
              f"{(1 - f_ent[~single]).sum() / len(f_ent):.4f} from non-singletons", ""]

    recs = val.records.set_index("entity_id")
    def show(df, title):
        out = [f"#### {title}"]
        for _, r in df.head(a.n_examples).iterrows():
            L, R = recs.loc[r["s1"]], recs.loc[r["cand"]]
            out.append(f"- [{r['type']}, p={r['score']:.2f}] `{L.business_name}` | `{L.business_address}` [{L.country}]"
                       f"  ⇄  `{R.business_name}` | `{R.business_address}` [{r['cand'][:2]}]")
        return out
    rng = np.random.default_rng(0)
    for typ in (fp["type"].unique() if len(fp) else []):
        d = fp[fp["type"] == typ]
        lines += show(d.iloc[rng.permutation(len(d))], f"FP examples — {typ}")
    for typ in (fn["type"].unique() if len(fn) else []):
        d = fn[fn["type"] == typ]
        lines += show(d.iloc[rng.permutation(len(d))], f"FN examples — {typ}")
    out = Path(a.out)
    if not out.exists():
        out.write_text("# T03 matcher analysis (p1) — generated by exploration/p1/t03_analyze.py\n")
    with open(out, "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
