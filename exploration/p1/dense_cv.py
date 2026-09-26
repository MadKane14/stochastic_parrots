"""Dense cross-validation (p1): out-of-fold F0.5 on the FULL train set, i.e. at test-like density.

Why: val_v1 / val_v1@0.1 keep only 20% / 2% of the other entities' S2/S3 records, so near-duplicate "sibling"
records (same address, one name word changed) almost never compete there, while in test they are everywhere.
Submission #1 scored 0.9487 on the LB vs 0.9917 on val_v1@0.1 (INSIGHTS I-15).

What it does (no change to the frozen eval or to core files):
  1. normalize + block the full train set with the config's pipeline (hits the pipeline cache of a previous
     `test` run of the same config: the training side of test mode is exactly this)
  2. featurize all candidate pairs (cached under $BER_ARTIFACTS/dense_cv/)
  3. 2-fold out-of-fold scores, folds = hash of the S1 id (whole S1 groups)
  4. decide with p1_assign; macro F0.5 over ALL train S1 (true pairs the blocker missed count as misses);
     headline number cross-checked with the frozen ber.eval.metrics
Outputs: exploration/p1/dense_cv_out.md (appended) + exploration/p1/dense_cv/<run>.json; OOF parquet in artifacts.

    nice -n 10 python exploration/p1/dense_cv.py -c code/business_entity_resolution/configs/exp/p1/T03_sub.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code/business_entity_resolution/src"))
os.environ.setdefault("BER_THREADS", "8")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ["BER_THREADS"])

from ber.config import config_hash, env, load_config  # noqa: E402
from ber.data import load_part, unit_hash  # noqa: E402
from ber.eval.metrics import pairs_to_sets, per_entity_scores  # noqa: E402
from ber.pipeline import Pipeline, _component_fp  # noqa: E402
from ber.registry import build  # noqa: E402

log = logging.getLogger("dense_cv")
FOLD_SEED = 777


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def fbeta_vec(tp, npred, ntrue):
    """Per-entity F0.5 exactly as the frozen metric defines it."""
    p = np.where(npred > 0, tp / np.maximum(npred, 1), 0.0)
    r = np.where(ntrue > 0, tp / np.maximum(ntrue, 1), 0.0)
    f = np.where(tp > 0, 1.25 * p * r / np.maximum(0.25 * p + r, 1e-12), 0.0)
    return np.where((ntrue == 0) & (npred == 0), 1.0, f)


def build_pairs(cfg: dict, art: Path):
    """Full-train normalized records + candidate pairs with integer row codes, labels and features (cached)."""
    full = load_part(env("BER_DATA_DIR"), "train", cache_dir=art / "cache")
    pipe = Pipeline(cfg, art)
    pipe._train_fp = full.fingerprint() + full.name        # identical to Pipeline.fit in test mode → cache hits
    pipe.normalizer.fit(full)
    t = time.time(); train = pipe._normalize(full); log.info("normalize %.0fs", time.time() - t)
    for b in pipe.blockers:
        b.fit(train)
    t = time.time(); pairs = pipe._block(train); log.info("block %.0fs: %d pairs", time.time() - t, len(pairs))
    recs = train.records
    n = len(recs)
    idx = pd.Index(recs["entity_id"].to_numpy())
    li = idx.get_indexer(pairs["s1_id"]).astype(np.int64)
    ri = idx.get_indexer(pairs["cand_id"]).astype(np.int64)
    gt = full.gt
    g_s = [s for s in gt for _ in gt[s]]
    g_c = [c for s in gt for c in gt[s]]
    gt_keys = np.unique(idx.get_indexer(g_s).astype(np.int64) * n + idx.get_indexer(g_c).astype(np.int64))
    label = np.isin(li * n + ri, gt_keys).astype(np.int8)
    # cache: one parquet with ids/label/blocker columns, plus one parquet PER feature set (adding a feature set
    # to a config only computes that set)
    base_key = hashlib.sha256("|".join([pipe._train_fp, *[_component_fp(b) for b in pipe.blockers],
                                        _component_fp(pipe.normalizer)]).encode()).hexdigest()[:16]
    cdir = art / "dense_cv"
    cdir.mkdir(parents=True, exist_ok=True)
    names = [f"base_{base_key}"]
    paths = [cdir / f"base_{base_key}.parquet"]
    if not paths[0].exists():
        blk = pairs[[c for c in pairs.columns if c.startswith("blk_")]].astype(np.float32)
        pd.concat([pd.DataFrame({"s1_id": li.astype(np.int32), "ri": ri.astype(np.int32), "label": label},
                                index=pairs.index), blk], axis=1).to_parquet(paths[0])
    for fs in pipe.featuresets:   # compute + write missing feature sets first (they need `pairs`)
        fkey = hashlib.sha256(f"{base_key}|{_component_fp(fs)}".encode()).hexdigest()[:16]
        fpath = cdir / f"feat_{fs.registry_name}_{fkey}.parquet"
        names.append(fpath.stem); paths.append(fpath)
        if fpath.exists():
            log.info("cached features %s", fpath.name)
        else:
            t = time.time()
            fs.fit(train, pairs)          # label-fitted feature sets (e.g. p1_noise) learn from the train gt here
            fs.transform(train, pairs).reset_index(drop=True).to_parquet(fpath)
            log.info("features %s %.0fs (peak %.1f GB)", fs.registry_name, time.time() - t, peak_gb())
    del pairs                     # free the 70M+ string pairs BEFORE loading every feature file
    import gc; gc.collect()
    df = pd.concat([pd.read_parquet(p_).reset_index(drop=True) for p_ in paths], axis=1, copy=False)
    fpath = Path("+".join(names))
    s1_rows = np.flatnonzero((recs["source"] == 1).to_numpy())
    ntrue = np.zeros(n, np.int32)
    np.add.at(ntrue, idx.get_indexer(list(gt)), [len(v) for v in gt.values()])
    meta = {"records": recs[["entity_id", "country"]], "s1_rows": s1_rows, "ntrue": ntrue, "gt": gt,
            "gt_keys": gt_keys, "n": n, "feature_cache": fpath.name}
    return df, meta


def oof_scores(df: pd.DataFrame, cfg: dict, folds: int = 2, s1_country: np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probabilities; folds by hash of the S1 row code, or (s1_country given) leave-one-country-out:
    each country is predicted by a model trained on the other countries only (unseen-country proxy at density)."""
    if s1_country is not None:
        codes, uniq = pd.factorize(s1_country[df["s1_id"].to_numpy()])
        fold, folds = codes.astype(np.int8), len(uniq)
    else:
        u = np.unique(df["s1_id"].to_numpy())
        fold_of_u = np.array([int(unit_hash(str(x), FOLD_SEED) * folds) for x in u], dtype=np.int8)
        fold = fold_of_u[np.searchsorted(u, df["s1_id"].to_numpy())]
    oof = np.zeros(len(df), np.float64)
    for k in range(folds):
        t = time.time()
        sc = build("scorer", cfg["scorer"])
        sc.fit(None, df.loc[fold != k])                  # s1_id = S1 row code: the scorer samples whole S1 groups
        oof[fold == k] = sc.score(None, df.loc[fold == k])
        log.info("fold %d: fit+score %.0fs (best_iteration %s, peak %.1f GB)", k, time.time() - t,
                 getattr(sc, "best_iteration", None), peak_gb())
    return oof, fold


def evaluate_dense(li, ri, label, score, meta, threshold=0.5, top1_min=None):
    """Vectorised macro F0.5 over all train S1 with the p1_assign rule (one owner per S2/S3 record)."""
    order = np.lexsort((-score, ri))                    # per right record: best S1 first
    first = np.r_[True, ri[order][1:] != ri[order][:-1]]
    own = order[first]                                   # each right record kept only for its best S1
    keep = own[score[own] >= threshold]
    if top1_min is not None:
        has = np.zeros(meta["n"], bool); has[li[keep]] = True
        rest = own[(score[own] >= top1_min) & ~has[li[own]]]
        rest = rest[np.lexsort((-score[rest], li[rest]))]
        rest = rest[np.r_[True, li[rest][1:] != li[rest][:-1]]] if len(rest) else rest
        keep = np.r_[keep, rest]
    n = meta["n"]
    tp = np.bincount(li[keep], weights=label[keep], minlength=n)
    npred = np.bincount(li[keep], minlength=n)
    rows = meta["s1_rows"]
    f = fbeta_vec(tp[rows], npred[rows], meta["ntrue"][rows])
    return f, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--tag", default="")
    ap.add_argument("--loco", action="store_true", help="leave-one-country-out folds instead of S1-hash folds")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    cfg = load_config(a.config, a.set)
    art = Path(cfg["paths"]["artifacts_dir"])
    t0 = time.time()
    df, meta = build_pairs(cfg, art)
    oof, fold = oof_scores(df, cfg, s1_country=meta["records"]["country"].to_numpy() if a.loco else None)
    li, ri = df["s1_id"].to_numpy().astype(np.int64), df["ri"].to_numpy().astype(np.int64)
    label = df["label"].to_numpy()
    dp = cfg["decider"]["params"]
    f, keep = evaluate_dense(li, ri, label, oof, meta, dp.get("threshold", 0.5), dp.get("top1_min"))

    recs, rows = meta["records"], meta["s1_rows"]
    country = recs["country"].to_numpy()[rows]
    ntrue = meta["ntrue"][rows]
    single = ntrue == 0
    s1_fold = np.full(meta["n"], -1, np.int8); s1_fold[li] = fold
    res = {"f05": round(float(f.mean()), 5), "by_country": {c: round(float(f[country == c].mean()), 5)
                                                            for c in sorted(set(country))},
           "by_fold": {int(k): round(float(f[s1_fold[rows] == k].mean()), 5) for k in np.unique(fold)},
           "folds": "leave-one-country-out" if a.loco else f"2 x S1 hash (seed {FOLD_SEED})",
           "singleton_acc": round(float(f[single].mean()), 4), "f05_nonsingleton": round(float(f[~single].mean()), 5),
           "n_s1": int(len(rows)), "n_pairs": int(len(df)),
           "pair_precision": round(float(label[keep].mean()), 5),
           "pair_recall": round(float(label[keep].sum() / len(meta["gt_keys"])), 5),
           "blocking_pair_recall": round(float(label.sum() / len(meta["gt_keys"])), 5)}
    tp_c = np.bincount(li, weights=label, minlength=meta["n"])[rows]
    res["oracle_f05"] = round(float(fbeta_vec(tp_c, tp_c, ntrue).mean()), 5)
    # frozen-metric cross-check of the headline number
    ids = recs["entity_id"].to_numpy()
    m = pd.DataFrame({"s1_id": ids[li[keep]], "cand_id": ids[ri[keep]]})
    s1_ids = ids[rows].tolist()
    frozen = float(per_entity_scores(pairs_to_sets(m, s1_ids), meta["gt"], s1_ids).mean())
    res["f05_frozen_metric"] = round(frozen, 5)
    curve = {}
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        curve[str(t)] = round(float(evaluate_dense(li, ri, label, oof, meta, t)[0].mean()), 5)
    res["threshold_curve"] = curve

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "code", "exploration/p1/dense_cv.py"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip())
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = f"{stamp}_densecv_{Path(a.config).stem}_{config_hash(cfg)[:6]}{('_' + a.tag) if a.tag else ''}"
    rec = {"run": run, "config_path": a.config, "overrides": a.set, "config": {k: v for k, v in cfg.items() if k != "_meta"},
           "git": {"commit": commit, "dirty": dirty}, "fold_seed": FOLD_SEED, "feature_cache": meta["feature_cache"],
           "metrics": res, "runtime_s": round(time.time() - t0), "peak_rss_gb": round(peak_gb(), 1)}
    out_dir = REPO / "exploration/p1/dense_cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run}.json").write_text(json.dumps(rec, indent=1, default=str))
    oof_dir = art / "dense_cv" / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"li": li.astype(np.int32), "ri": ri.astype(np.int32), "label": label, "fold": fold,
                  "oof": oof.astype(np.float32)}).to_parquet(oof_dir / f"{run}.parquet")
    md = REPO / "exploration/p1/dense_cv_out.md"
    if not md.exists():
        md.write_text("# Dense CV (full train, test-like density) — generated by exploration/p1/dense_cv.py\n")
    with open(md, "a") as fh:
        fh.write(f"\n## {run} (commit {commit}{' DIRTY' if dirty else ''})\n\n```\n{json.dumps(res, indent=1)}\n```\n")
    print(json.dumps(res, indent=1))
    log.info("done in %.0fs, peak %.1f GB", time.time() - t0, peak_gb())


if __name__ == "__main__":
    main()
