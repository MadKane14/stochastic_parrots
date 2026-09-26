"""T05 (p3): full-density blocking eval of a config's blocker UNION, including the pipeline's per-S1 cap.

Same data, query sampling and metrics as exploration/p1/blk_eval.py (imported), plus:
  * each blocker runs in its own subprocess and its (pair key, rank) output is cached under
    $BER_ARTIFACTS/p3_blk/<fingerprint>.npz (fingerprint = normalizer + blocker spec + source code + qfrac), so
    p1_compound runs once for all p3 variants, and peak memory = max over blockers, not their sum
    (fits a 30 GB box where blk_eval.py --config with two blockers does not);
  * `blocking.max_candidates` (and --caps) is emulated exactly like Pipeline._block: per S1, keep the candidates
    with the smallest best rank over blockers (rank = order by score inside each blocker).

Usage (repo root, Linux, env as in CLAUDE.md):
    nice -n 10 python exploration/p3/blk_union.py --config code/business_entity_resolution/configs/exp/p3/T05_x.yaml \
        --qfrac 0.1 --caps 40,50,60 --out exploration/p3/blk_eval_out.md [--misses 50]
--qfrac: see blk_eval.py (forward recall unbiased; reverse retrieval optimistic -> confirm finalists with --qfrac 1).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code/business_entity_resolution/src"))
sys.path.insert(0, str(REPO / "exploration/p1"))
os.environ.setdefault("BER_THREADS", "4")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, os.environ["BER_THREADS"])

import blk_eval as be  # noqa: E402
from ber.config import env, load_config  # noqa: E402
from ber.data import Dataset, unit_hash  # noqa: E402
from ber.pipeline import _component_fp  # noqa: E402
from ber.registry import build  # noqa: E402


def load(cfg: dict, qfrac: float):
    """Normalized full train restricted to the query S1 (+ all S2/S3), gt of the queries, labels fit may use."""
    recs, gt = be.load_normalized(1.0, build("normalizer", cfg["normalize"]), qfrac)
    train_gt = be.fit_gt(gt, qfrac)
    if qfrac < 1:
        is_s1 = recs["source"] == 1
        keep_s1 = recs["entity_id"].map(lambda e: unit_hash(e, 5) < qfrac) & is_s1
        recs = recs[keep_s1 | ~is_s1].reset_index(drop=True)
        kept = set(recs.loc[recs["source"] == 1, "entity_id"])
        gt = {s: v for s, v in gt.items() if s in kept}
    return recs, gt, train_gt


def store_path(cfg: dict, i: int, qfrac: float) -> Path:
    spec = cfg["blocking"]["blockers"][i]
    fp = "|".join([_component_fp(build("normalizer", cfg["normalize"])), _component_fp(build("blocker", spec)),
                   json.dumps(spec, sort_keys=True), str(qfrac)])
    return Path(env("BER_ARTIFACTS")) / "p3_blk" / f"{spec['name']}_{hashlib.sha256(fp.encode()).hexdigest()[:12]}.npz"


def run_one(cfg: dict, i: int, qfrac: float, out: Path):
    """Run blocker i on the full-density data; save (key = s1_row * n + right_row, rank by score within S1)."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    t0 = time.time()
    recs, _, train_gt = load(cfg, qfrac)
    print(f"loaded {len(recs):,} records ({time.time() - t0:.0f}s, peak {be.peak_gb():.1f} GB)", flush=True)
    n = len(recs)
    b = build("blocker", cfg["blocking"]["blockers"][i])
    t = time.time()
    b.fit(Dataset("train", recs, train_gt))
    if hasattr(b, "candidate_rows"):
        r, c, v = b.candidate_rows(recs)
    else:
        row_of = pd.Series(np.arange(n), index=recs["entity_id"].to_numpy())
        cand = b.candidates(Dataset("train", recs))
        r, c, v = row_of[cand["s1_id"]].to_numpy(), row_of[cand["cand_id"]].to_numpy(), cand["score"].to_numpy()
    secs = time.time() - t
    order = np.lexsort((-v, r))
    r, c = r[order], c[order]
    start = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
    rank = np.arange(len(r)) - np.repeat(start, np.diff(np.r_[start, len(r)]))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".tmp.npz"), key=r.astype(np.int64) * n + c, rank=rank.astype(np.int32),
             secs=secs, peak_gb=be.peak_gb(), n=n)
    out.with_suffix(".tmp.npz").replace(out)
    print(f"blocker {i}: {len(r):,} pairs in {secs:.0f}s, peak {be.peak_gb():.1f} GB -> {out}", flush=True)


def union_keys(chans: list[tuple[np.ndarray, np.ndarray]], n: int, cap: int | None) -> np.ndarray:
    """Unique pair keys of the union; with `cap`, per S1 the `cap` pairs with the smallest best rank
    (= Pipeline._block with blocking.max_candidates)."""
    keys = np.concatenate([k for k, _ in chans])
    ranks = np.concatenate([r for _, r in chans])
    o = np.lexsort((ranks, keys))
    keys, ranks = keys[o], ranks[o]
    first = np.r_[True, keys[1:] != keys[:-1]]
    keys, best = keys[first], ranks[first]
    if cap:
        s1 = keys // n
        o = np.lexsort((keys, best, s1))
        keys, s1 = keys[o], s1[o]
        start = np.r_[0, np.flatnonzero(np.diff(s1)) + 1]
        pos = np.arange(len(keys)) - np.repeat(start, np.diff(np.r_[start, len(keys)]))
        keys = np.sort(keys[pos < cap])
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="eval", choices=["eval", "one"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--qfrac", type=float, default=0.1)
    ap.add_argument("--i", type=int, default=0, help="(mode one) blocker index")
    ap.add_argument("--caps", default="", help="extra per-S1 caps to report, e.g. 40,50,60")
    ap.add_argument("--misses", type=int, default=0, help="print miss analysis of the union at this cap (0 = off)")
    ap.add_argument("--out", default=str(REPO / "exploration/p3/blk_eval_out.md"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    specs = cfg["blocking"]["blockers"]
    if a.mode == "one":
        return run_one(cfg, a.i, a.qfrac, store_path(cfg, a.i, a.qfrac))

    paths = [store_path(cfg, i, a.qfrac) for i in range(len(specs))]
    for i, p in enumerate(paths):   # one process per blocker (memory is returned to the OS in between)
        if p.exists():
            print(f"blocker {i} ({specs[i]['name']}): cached {p.name}", flush=True)
            continue
        rc = subprocess.run([sys.executable, __file__, "one", "--config", a.config, "--qfrac", str(a.qfrac),
                             "--i", str(i)]).returncode
        if rc != 0:
            sys.exit(f"blocker {i} failed (rc={rc})")

    t0 = time.time()
    recs, gt, _ = load(cfg, a.qfrac)
    n = len(recs)
    zs = [np.load(p) for p in paths]
    assert all(int(z["n"]) == n for z in zs), "record order differs between cached runs"
    chans = [(z["key"], z["rank"]) for z in zs]
    row_of = pd.Series(np.arange(n), index=recs["entity_id"].to_numpy())
    s1_rows = np.flatnonzero((recs["source"] == 1).to_numpy())
    g_s1 = [s for s in gt for _ in gt[s]]
    g_c = [c for s in gt for c in gt[s]]
    gt_keys = np.unique(row_of[g_s1].to_numpy() * n + row_of[g_c].to_numpy())
    truth_n = np.zeros(n, np.int32)
    np.add.at(truth_n, row_of[list(gt)].to_numpy(), [len(v) for v in gt.values()])
    country = recs["country"].to_numpy()

    def metrics(keys: np.ndarray) -> dict:   # = blk_eval.main.metrics on a set of unique keys
        hit = np.isin(keys, gt_keys, assume_unique=True)
        s1r = keys // n
        n_c = np.bincount(s1r, minlength=n)[s1_rows]
        found = np.bincount(s1r[hit], minlength=n)[s1_rows]
        truth = truth_n[s1_rows]
        rr = np.where(truth > 0, found / np.maximum(truth, 1), 0.0)
        f = np.where(truth == 0, 1.0, np.where(found > 0, 1.25 * rr / (0.25 + rr), 0.0))
        out = {"pair_recall": hit.sum() / len(gt_keys), "oracle_f05": f.mean(), "cands_mean": n_c.mean(),
               "cands_p95": np.quantile(n_c, 0.95), "cands_max": n_c.max()}
        for cn in sorted(set(country[s1_rows])):
            m = country[s1_rows] == cn
            out[f"recall_{cn}"] = found[m].sum() / max(1, truth[m].sum())
        return out

    names = [f"b{i}_{s['name']}" for i, s in enumerate(specs)]
    rows = []
    for nm, ch in zip(names, chans):
        rows.append({"candidates": nm, **metrics(np.unique(ch[0]))})
    caps = sorted({int(x) for x in a.caps.split(",") if x} | ({int(cfg["blocking"]["max_candidates"])}
                                                             if cfg["blocking"].get("max_candidates") else set()))
    if len(chans) > 1:
        rows.append({"candidates": "union", **metrics(union_keys(chans, n, None))})
    for cap in caps:
        rows.append({"candidates": f"union cap {cap}" + (" (config)" if cap == cfg["blocking"].get("max_candidates") else ""),
                     **metrics(union_keys(chans, n, cap))})
    rows = [{k: (round(float(v), 4) if not isinstance(v, str) else v) for k, v in r.items()} for r in rows]
    for r in rows:
        print(r, flush=True)
    if a.misses:
        u = union_keys(chans, n, a.misses)
        be.miss_analysis(recs, gt_keys, {"u": (u, np.zeros(len(u), np.int32))}, {"u": 1}, n)

    s1_n = len(s1_rows)
    secs = [float(z["secs"]) for z in zs]
    lines = [f"\n## Run {dt.datetime.now():%Y-%m-%d %H:%M} — config={a.config} qfrac={a.qfrac} "
             f"BER_THREADS={os.environ['BER_THREADS']} (exploration/p3/blk_union.py)\n",
             "- blockers: " + "; ".join(f"`{s['name']} {json.dumps(s.get('params') or {})}`" for s in specs),
             f"- records {n:,} (S1 {s1_n:,}, right {n - s1_n:,}); GT pairs {len(gt_keys):,}",
             "- blocking time (s) / peak RSS (GB) per blocker process: "
             + ", ".join(f"{nm}={t:.0f}s/{float(z['peak_gb']):.1f}GB" for nm, t, z in zip(names, secs, zs))
             + f"; cache files: {', '.join(p.name for p in paths)}",
             f"- test projection of blocking time (forward scales with #S1 queries; x {1.73e6 / max(1, s1_n):.1f}): "
             f"~{sum(secs) * 1.73e6 / max(1, s1_n):.0f}s (upper bound at qfrac < 1)\n",
             be.md_table(rows)]
    out = Path(a.out)
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# Blocker-union evaluation on full train (test-like density) — "
                       "generated by exploration/p3/blk_union.py\n")
    with open(out, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"eval {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
