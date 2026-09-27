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


def rss_gb() -> float:
    """Current resident memory (not the peak) from /proc, for memory diagnostics."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


def build_pairs_lean(cfg: dict, art: Path):
    """Same steps, caches and outputs as p1's `dense_cv.build_pairs` (records -> normalize -> block -> labels -> one
    parquet per feature set -> one frame), but every per-record structure is released BEFORE the feature files are
    loaded: the feature-set objects (p1_match/p1_extra keep names, skeletons and token matrices of all records), the
    blockers' module-level token caches, the pipeline and the raw/normalized record tables. On a 31 GB machine the
    full p1+p2+p3+p4 stack was OOM-killed exactly at that load in p1's version (logs in exploration/p2/kaggle_logs)."""
    import gc
    import hashlib

    import pyarrow.parquet as pq
    from ber.pipeline import Pipeline, _component_fp
    full = dcv.load_part(dcv.env("BER_DATA_DIR"), "train", cache_dir=art / "cache")
    pipe = Pipeline(cfg, art)
    pipe._train_fp = full.fingerprint() + full.name
    pipe.normalizer.fit(full)
    t = time.time(); train = pipe._normalize(full); log.info("normalize %.0fs (rss %.1f GB)", time.time() - t, rss_gb())
    for b in pipe.blockers:
        b.fit(train)
    t = time.time(); pairs = pipe._block(train)
    log.info("block %.0fs: %d pairs (rss %.1f GB)", time.time() - t, len(pairs), rss_gb())
    recs = train.records
    n = len(recs)
    idx = pd.Index(recs["entity_id"].to_numpy())
    li = idx.get_indexer(pairs["s1_id"]).astype(np.int64)
    ri = idx.get_indexer(pairs["cand_id"]).astype(np.int64)
    gt = full.gt
    g_s = [s for s in gt for _ in gt[s]]
    g_c = [c for s in gt for c in gt[s]]
    gt_keys = np.unique(idx.get_indexer(g_s).astype(np.int64) * n + idx.get_indexer(g_c).astype(np.int64))
    del g_s, g_c
    label = np.isin(li * n + ri, gt_keys).astype(np.int8)
    base_key = hashlib.sha256("|".join([pipe._train_fp, *[_component_fp(b) for b in pipe.blockers],
                                        _component_fp(pipe.normalizer)]).encode()).hexdigest()[:16]
    cdir = art / "dense_cv"
    cdir.mkdir(parents=True, exist_ok=True)
    names, paths = [f"base_{base_key}"], [cdir / f"base_{base_key}.parquet"]
    if not paths[0].exists():
        blk = pairs[[c for c in pairs.columns if c.startswith("blk_")]].astype(np.float32)
        pd.concat([pd.DataFrame({"s1_id": li.astype(np.int32), "ri": ri.astype(np.int32), "label": label},
                                index=pairs.index), blk], axis=1).to_parquet(paths[0])
        del blk
    for i, fs in enumerate(pipe.featuresets):
        fkey = hashlib.sha256(f"{base_key}|{_component_fp(fs)}".encode()).hexdigest()[:16]
        fpath = cdir / f"feat_{fs.registry_name}_{fkey}.parquet"
        names.append(fpath.stem); paths.append(fpath)
        if not fpath.exists():
            t = time.time()
            fs.fit(train, pairs)
            fs.transform(train, pairs).reset_index(drop=True).to_parquet(fpath)
            log.info("features %s %.0fs (peak %.1f GB, rss %.1f GB)", fs.registry_name, time.time() - t, dcv.peak_gb(), rss_gb())
        pipe.featuresets[i] = None                  # drop the set's per-record arrays right away
        del fs
        gc.collect()
    # keep only what scoring/evaluation needs, release everything else before loading the feature files
    meta_recs = recs[["entity_id", "country"]].copy()
    s1_rows = np.flatnonzero((recs["source"] == 1).to_numpy())
    ntrue = np.zeros(n, np.int32)
    np.add.at(ntrue, idx.get_indexer(list(gt)), [len(v) for v in gt.values()])
    del pairs, recs, train, full, pipe, idx, li, ri, label
    import importlib
    for mod, attrs in (("ber.blocking.p1_compound", ["_TOK_CACHE"]), ("ber.blocking.p1_idf_overlap", ["_TOKEN_CACHE", "_W"]),
                       ("ber.blocking.p3_chargram", ["_CACHE", "_W"]), ("ber.blocking.p3_nameaddr", ["_CACHE"]),
                       ("ber.normalize.p2_translit", ["_PREP_CACHE"])):
        try:
            m = importlib.import_module(mod)
            for a in attrs:
                getattr(m, a, {}).clear()
        except ImportError:
            pass
    gc.collect()
    log.info("released per-record structures (rss %.1f GB)", rss_gb())
    parts = []
    for p_ in paths:
        part = pq.read_table(p_).to_pandas(self_destruct=True, split_blocks=True).reset_index(drop=True)
        f64 = [c for c in part.columns if part[c].dtype == np.float64]
        if f64:
            part[f64] = part[f64].astype(np.float32)
        parts.append(part)
        log.info("loaded %s (rss %.1f GB)", p_.name, rss_gb())
    df = pd.concat(parts, axis=1, copy=False)
    del parts
    gc.collect()
    log.info("feature frame %d x %d (rss %.1f GB)", len(df), df.shape[1], rss_gb())
    meta = {"records": meta_recs, "s1_rows": s1_rows, "ntrue": ntrue, "gt": gt, "gt_keys": gt_keys, "n": n,
            "feature_cache": "+".join(names)}
    return df, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--group", required=True, help="country string of the group to evaluate (evaluation choice)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lean", action="store_true", help="release per-record structures before loading features")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    cfg = load_config(a.config)
    art = Path(cfg["paths"]["artifacts_dir"])
    t0 = time.time()
    dcv.load_part = group_loader(a.group)                       # p1's build_pairs picks this up
    df, meta = build_pairs_lean(cfg, art) if a.lean else dcv.build_pairs(cfg, art)
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
