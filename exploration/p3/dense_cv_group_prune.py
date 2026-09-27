"""T18 (p3): p2's dense_cv_group.py, run UNCHANGED (it writes its JSON to exploration/p2/dense_cv/ as usual), plus the
pipeline's learned pruning stage evaluated out of fold on the same run — without refitting the final model.

How: p1's dense_cv.build_pairs / oof_scores are wrapped (not modified) so the pairs table, the final model's
out-of-fold scores and the folds of p2's run are kept in memory. Then, exactly like Pipeline._prune:
  * the configured stage-1 scorer (`prune.scorer`, on `prune.features` only) is fitted on the training candidates
    of each fold and scores the held-out fold's candidates (the pipeline fits it on all train candidates and applies
    it to test);
  * per S1 keep candidates with stage-1 score >= `prune.min_score`, best first, at most `prune.max_per_s1`;
  * the final model is NOT refitted (the pipeline trains it on unpruned train candidates too): the same out-of-fold
    scores are decided on the survivors with p1_assign's rule (dense_cv.evaluate_dense), so pruned-away true pairs
    count as misses.
Writes exploration/p3/dense_cv/<p2 run>_prune.json (+ exploration/p3/dense_cv_out.md).

    python exploration/p3/dense_cv_group_prune.py -c code/business_entity_resolution/configs/exp/p3/T15_sub5.yaml --group India
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code/business_entity_resolution/src"))
_spec = importlib.util.spec_from_file_location("p2_dense_cv_group", REPO / "exploration/p2/dense_cv_group.py")
dcg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dcg)
dcv = dcg.dcv                                     # p1's dense_cv module as loaded by p2's script

from ber.registry import build  # noqa: E402

CAP: dict = {}
_build_pairs, _oof_scores = dcv.build_pairs, dcv.oof_scores


# ---- memory hygiene only (results unchanged): the 30 GiB Kaggle box OOM-killed the 3-blocker run right when the
# feature caches are concatenated (t18d, exploration/p3/kaggle_t18d/out). (1) the blockers' module-level token caches
# (~GBs for 5M records) are cleared once blocking is done; (2) build_pairs' gc.collect() before the concat also
# returns freed heap to the OS (glibc malloc_trim), so the freed string pairs do not stay reserved.
import ctypes  # noqa: E402
import gc  # noqa: E402
import importlib  # noqa: E402

from ber.pipeline import Pipeline  # noqa: E402

_collect = gc.collect


def _collect_trim(*args, **kw):
    r = _collect(*args, **kw)
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
    return r


_orig_block = Pipeline._block


def _block_then_free(self, ds):
    out = _orig_block(self, ds)
    for mod, attr in (("ber.blocking.p1_compound", "_TOK_CACHE"), ("ber.blocking.p1_idf_overlap", "_TOKEN_CACHE"),
                      ("ber.blocking.p3_chargram", "_CACHE"), ("ber.blocking.p3_nameaddr", "_CACHE")):
        try:
            getattr(importlib.import_module(mod), attr).clear()
        except (ImportError, AttributeError):
            pass
    _collect_trim()
    return out


def _capture_build(cfg, art):
    df, meta = _build_pairs(cfg, art)
    CAP.update(df=df, meta=meta, cfg=cfg)
    return df, meta


def _capture_oof(df, cfg, *args, **kw):
    oof, fold = _oof_scores(df, cfg, *args, **kw)
    CAP.update(oof=oof, fold=fold)
    return oof, fold


def prune_eval(df, meta, cfg, oof, fold) -> dict:
    """Out-of-fold emulation of Pipeline._prune + decision on the survivors (see module docstring)."""
    pc = cfg["prune"]
    feats = [c for c in pc["features"] if c in df.columns]
    missing = [c for c in pc["features"] if c not in df.columns]
    view = df[["s1_id", "label"] + feats]
    q = np.zeros(len(df), np.float64)
    t0 = time.time()
    for k in np.unique(fold):
        pr = build("scorer", pc["scorer"])
        pr.fit(None, view.loc[fold != k])
        q[fold == k] = pr.score(None, view.loc[fold == k])
    fit_s = time.time() - t0
    li, ri = df["s1_id"].to_numpy().astype(np.int64), df["ri"].to_numpy().astype(np.int64)
    label = df["label"].to_numpy()
    cand = np.flatnonzero(q >= float(pc.get("min_score", 0.0)))
    cap = pc.get("max_per_s1")
    if cap:                                        # per S1: best stage-1 score first, at most `cap`
        o = cand[np.lexsort((-q[cand], li[cand]))]
        g = li[o]
        start = np.maximum.accumulate(np.where(np.r_[True, g[1:] != g[:-1]], np.arange(len(g)), 0))
        cand = np.sort(o[(np.arange(len(g)) - start) < int(cap)])
    sel = cand
    dp = cfg["decider"]["params"]
    thr = dp.get("threshold", 0.5)
    f, keep = dcv.evaluate_dense(li[sel], ri[sel], label[sel], oof[sel], meta, thr, dp.get("top1_min"))
    rows = meta["s1_rows"]
    n = meta["n"]
    s1_fold = np.full(n, -1, np.int8)
    s1_fold[li] = fold
    nc = np.bincount(li[sel], minlength=n)[rows]
    tp_c = np.bincount(li[sel], weights=label[sel], minlength=n)[rows]
    ntrue = meta["ntrue"][rows]
    n_gt = len(meta["gt_keys"])
    return {"prune_features_used": feats, "prune_features_missing": missing,
            "min_score": pc.get("min_score"), "max_per_s1": cap, "stage1_fit_score_s": round(fit_s),
            "f05": round(float(f.mean()), 5),
            "by_fold": {int(k): round(float(f[s1_fold[rows] == k].mean()), 5) for k in np.unique(fold)},
            "n_pairs": int(len(sel)),
            "cands_per_s1": {"mean": round(float(nc.mean()), 2), "p95": float(np.quantile(nc, 0.95)),
                             "max": int(nc.max())},
            "pair_recall_after_prune": round(float(label[sel].sum() / n_gt), 5),
            "oracle_f05_after_prune": round(float(dcv.fbeta_vec(tp_c, tp_c, ntrue).mean()), 5),
            "threshold_curve": {str(t): round(float(dcv.evaluate_dense(li[sel], ri[sel], label[sel], oof[sel],
                                                                         meta, t)[0].mean()), 5)
                                for t in (0.5, 0.6, 0.7, 0.8)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dcv.build_pairs, dcv.oof_scores = _capture_build, _capture_oof
    gc.collect, Pipeline._block = _collect_trim, _block_then_free
    t0 = time.time()
    dcg.main()                                     # p2's script, unchanged (parses the same sys.argv)
    p2_json = max((REPO / "exploration/p2/dense_cv").glob(f"*_densecv_{a.group}_{Path(a.config).stem}_*.json"),
                  key=lambda p: p.stat().st_mtime)
    rec = json.loads(p2_json.read_text())
    res = prune_eval(CAP["df"], CAP["meta"], CAP["cfg"], CAP["oof"], CAP["fold"]) if CAP["cfg"].get("prune") else None
    out = {"run": rec["run"] + "_prune", "p2_run": rec["run"], "config_path": a.config, "group": a.group,
           "git": rec["git"], "no_prune": rec["metrics"], "prune": res, "runtime_s_total": round(time.time() - t0),
           "peak_rss_gb": round(dcv.peak_gb(), 1), "p2_peak_rss_gb": rec["peak_rss_gb"]}
    d = REPO / "exploration/p3/dense_cv"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{out['run']}.json").write_text(json.dumps(out, indent=1, default=str))
    md = REPO / "exploration/p3/dense_cv_out.md"
    if not md.exists():
        md.write_text("# India-group dense CV + out-of-fold pruning (p3) — generated by exploration/p3/dense_cv_group_prune.py\n")
    with open(md, "a", encoding="utf-8") as fh:
        fh.write(f"\n## {out['run']} ({dt.datetime.now():%Y-%m-%d %H:%M}, commit {rec['git']['commit']}"
                 f"{' DIRTY' if rec['git']['dirty'] else ''})\n\n```\n{json.dumps({k: out[k] for k in ('no_prune', 'prune', 'peak_rss_gb')}, indent=1)}\n```\n")
    print(json.dumps({"no_prune": rec["metrics"], "prune": res}, indent=1))


if __name__ == "__main__":
    main()
