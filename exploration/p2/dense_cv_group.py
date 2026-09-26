"""T10 (p2): p1's dense cross-validation restricted to ONE country-string group, so it fits a 31 GB machine.

Why exact for that group: the blocker (p1_compound) and the features (p1_match / p1_extra with `idf_group: country`)
work within equal-country-string groups, so blocking, IDF statistics and candidate density of the group's records
are identical to a full-train run. What differs from p1's full dense CV: the matcher is trained only on the group's
pairs (2-fold out-of-fold by S1 hash, as in p1), so the number is not comparable to p1's all-country dense CV, only
to other runs of this script. The group string is an evaluation choice (like the xc_* splits), not pipeline logic.

Reuses p1's functions unchanged (`exploration/p1/dense_cv.py`: build_pairs, oof_scores, evaluate_dense, fbeta_vec);
writes to exploration/p2/dense_cv/<run>.json + exploration/p2/dense_cv_out.md.

    python exploration/p2/dense_cv_group.py -c code/business_entity_resolution/configs/exp/p2/T10_d.yaml --group India
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("p1_dense_cv", REPO / "exploration/p1/dense_cv.py")
dcv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dcv)

from ber.config import config_hash, load_config  # noqa: E402
from ber.data import Dataset, load_part  # noqa: E402
from ber.eval.metrics import pairs_to_sets, per_entity_scores  # noqa: E402

log = logging.getLogger("dense_cv_group")


def group_loader(group: str):
    """load_part replacement: only the records (and GT entries) of one country-string group."""
    def load(data_dir, part, cache_dir=None):
        full = load_part(data_dir, part, cache_dir=cache_dir)
        recs = full.records[full.records["country"] == group].reset_index(drop=True)
        if not len(recs):
            raise SystemExit(f"no records with country == {group!r}")
        s1 = set(recs.loc[recs["source"] == 1, "entity_id"])
        gt = {s: v for s, v in full.gt.items() if s in s1}
        return Dataset(f"{part}[{group}]", recs, gt, dict(full.meta, group=group))
    return load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--group", required=True, help="country string of the group to evaluate (evaluation choice)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    cfg = load_config(a.config)
    art = Path(cfg["paths"]["artifacts_dir"])
    t0 = time.time()
    dcv.load_part = group_loader(a.group)                       # p1's build_pairs picks this up
    df, meta = dcv.build_pairs(cfg, art)
    oof, fold = dcv.oof_scores(df, cfg)
    li, ri = df["s1_id"].to_numpy().astype(np.int64), df["ri"].to_numpy().astype(np.int64)
    label = df["label"].to_numpy()
    dp = cfg["decider"]["params"]
    f, keep = dcv.evaluate_dense(li, ri, label, oof, meta, dp.get("threshold", 0.5), dp.get("top1_min"))
    rows = meta["s1_rows"]
    ntrue = meta["ntrue"][rows]
    single = ntrue == 0
    s1_fold = np.full(meta["n"], -1, np.int8); s1_fold[li] = fold
    cands = np.bincount(li, minlength=meta["n"])[rows]
    tp_c = np.bincount(li, weights=label, minlength=meta["n"])[rows]
    res = {"group": a.group, "f05": round(float(f.mean()), 5),
           "by_fold": {int(k): round(float(f[s1_fold[rows] == k].mean()), 5) for k in np.unique(fold)},
           "singleton_acc": round(float(f[single].mean()), 4), "f05_nonsingleton": round(float(f[~single].mean()), 5),
           "n_s1": int(len(rows)), "n_pairs": int(len(df)),
           "cands_per_s1": {"mean": round(float(cands.mean()), 2), "p95": float(np.quantile(cands, 0.95)),
                            "max": int(cands.max())},
           "pair_precision": round(float(label[keep].mean()), 5),
           "pair_recall": round(float(label[keep].sum() / len(meta["gt_keys"])), 5),
           "blocking_pair_recall": round(float(label.sum() / len(meta["gt_keys"])), 5),
           "oracle_f05": round(float(dcv.fbeta_vec(tp_c, tp_c, ntrue).mean()), 5)}
    ids = meta["records"]["entity_id"].to_numpy()
    m = pd.DataFrame({"s1_id": ids[li[keep]], "cand_id": ids[ri[keep]]})
    s1_ids = ids[rows].tolist()
    res["f05_frozen_metric"] = round(float(per_entity_scores(pairs_to_sets(m, s1_ids), meta["gt"], s1_ids).mean()), 5)
    res["threshold_curve"] = {str(t): round(float(dcv.evaluate_dense(li, ri, label, oof, meta, t)[0].mean()), 5)
                              for t in (0.5, 0.6, 0.7, 0.8)}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "code", "exploration/p1/dense_cv.py",
                                 "exploration/p2/dense_cv_group.py"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = f"{stamp}_densecv_{a.group}_{Path(a.config).stem}_{config_hash(cfg)[:6]}{('_' + a.tag) if a.tag else ''}"
    rec = {"run": run, "config_path": a.config, "config": {k: v for k, v in cfg.items() if k != "_meta"},
           "git": {"commit": commit, "dirty": dirty}, "fold_seed": dcv.FOLD_SEED, "feature_cache": meta["feature_cache"],
           "metrics": res, "runtime_s": round(time.time() - t0), "peak_rss_gb": round(dcv.peak_gb(), 1)}
    out_dir = REPO / "exploration/p2/dense_cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run}.json").write_text(json.dumps(rec, indent=1, default=str))
    md = REPO / "exploration/p2/dense_cv_out.md"
    if not md.exists():
        md.write_text("# Dense CV on one country group (p2) — generated by exploration/p2/dense_cv_group.py\n")
    with open(md, "a") as fh:
        fh.write(f"\n## {run} (commit {commit}{' DIRTY' if dirty else ''})\n\n```\n{json.dumps(res, indent=1)}\n```\n")
    print(json.dumps(res, indent=1))
    log.info("done in %.0fs, peak %.1f GB", time.time() - t0, dcv.peak_gb())


if __name__ == "__main__":
    main()
