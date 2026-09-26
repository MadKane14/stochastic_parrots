"""T04 (p2): full-density blocking recall of a config's normalizer + blockers, with subgroup breakdowns.

Same protocol and metric code as p1's shared tool `exploration/p1/blk_eval.py` (imported, not copied):
full train (all 10.3M S2/S3 records), a hash-fraction `--qfrac` of S1 entities as queries, the normalizer fitted
only on the labels of the OTHER S1 entities (`fit_gt`), blockers exactly as configured. Differences:
  * normalized records are pruned to the columns blockers use before caching (the shared tool keeps raw columns
    too, which pushes a 31 GB Kaggle machine over its limit);
  * recall is also reported on subsets of true pairs defined from the RAW S2/S3 record (so every variant is
    measured on the same pairs): non-Latin name, web-style name, empty address, single-letter runs in the name;
  * a JSON with every number is written next to the markdown (`--json`).

Usage (repo root):
    python exploration/p2/blk_eval_p2.py --config code/business_entity_resolution/configs/exp/p2/T04_a.yaml \
        --qfrac 0.1 --label T04_a --out exploration/p2/blk_eval_out.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code/business_entity_resolution/src"))
os.environ.setdefault("BER_THREADS", "4")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, os.environ["BER_THREADS"])

_spec = importlib.util.spec_from_file_location("p1_blk_eval", REPO / "exploration/p1/blk_eval.py")
p1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p1)

from ber.config import env, load_config  # noqa: E402
from ber.data import Dataset, load_part, unit_hash  # noqa: E402
from ber.pipeline import _component_fp  # noqa: E402
from ber.registry import build  # noqa: E402

KEEP = ["entity_id", "source", "country", "name_norm", "addr_norm"]
_WEB = re.compile(r"\.[A-Za-z]{2,6}\s*$|^\s*www\.", re.IGNORECASE)
_RUN = re.compile(r"(?<![A-Za-z])[A-Za-z][ .]+[A-Za-z](?:[ .]+[A-Za-z])*(?![A-Za-z])")


def _nonlatin(s: str) -> bool:
    import unicodedata
    return any(ch.isalpha() and not ch.isascii() and not unicodedata.name(ch, "").startswith("LATIN") for ch in s)


def load(config: dict, qfrac: float):
    art = Path(env("BER_ARTIFACTS"))
    ds = load_part(env("BER_DATA_DIR"), "train", cache_dir=art / "cache")
    norm = build("normalizer", config["normalize"])
    cache = art / "cache" / f"p2blk_norm_{_component_fp(norm)}_q{qfrac}.parquet"
    stats = {}
    if cache.exists():
        recs = pd.read_parquet(cache)
    else:
        t = time.time()
        norm.fit(Dataset("train", ds.records, p1.fit_gt(ds.gt, qfrac)))
        recs = norm.transform(ds.records)[KEEP]
        recs.to_parquet(cache)
        stats = {"normalize_s": round(time.time() - t, 1), **getattr(norm, "stats", {})}
        for attr in ("name_map", "addr_map"):
            m = getattr(norm, attr, None)
            if m:
                (art / "cache" / f"p2blk_{attr}_{_component_fp(norm)}_q{qfrac}.json").write_text(
                    json.dumps(m, ensure_ascii=False))
        print(f"normalized {len(recs):,} records in {stats['normalize_s']:.0f}s; stats {stats}", flush=True)
    raw = ds.records[["business_name", "business_address"]]
    return recs, ds.gt, raw, stats


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--qfrac", type=float, default=0.1)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=str(REPO / "exploration/p2/blk_eval_out.md"))
    ap.add_argument("--json", default="", help="default: exploration/p2/blk_eval_json/<label>_q<qfrac>.json")
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(a.config)
    recs, gt, raw, nstats = load(cfg, a.qfrac)
    assert len(raw) == len(recs)
    train_gt = p1.fit_gt(gt, a.qfrac)
    is_s1 = (recs["source"] == 1).to_numpy()
    if a.qfrac < 1:
        keep_s1 = recs["entity_id"].map(lambda e: unit_hash(e, 5) < a.qfrac).to_numpy() & is_s1
        keep = keep_s1 | ~is_s1
        recs, raw = recs[keep].reset_index(drop=True), raw[keep].reset_index(drop=True)
        kept = set(recs.loc[recs["source"] == 1, "entity_id"])
        gt = {s: v for s, v in gt.items() if s in kept}
    print(f"loaded {len(recs):,} records, {len(gt):,} S1 queries ({time.time() - t0:.0f}s, peak {p1.peak_gb():.1f} GB)",
          flush=True)

    n = len(recs)
    row_of = pd.Series(np.arange(n), index=recs["entity_id"].to_numpy())
    s1_rows = np.flatnonzero((recs["source"] == 1).to_numpy())
    g_s1 = row_of[[s for s in gt for _ in gt[s]]].to_numpy()
    g_c = row_of[[c for s in gt for c in gt[s]]].to_numpy()
    gt_keys, first = np.unique(g_s1 * n + g_c, return_index=True)
    g_c = g_c[first]
    truth_n = np.zeros(n, np.int32)
    np.add.at(truth_n, row_of[list(gt)].to_numpy(), [len(v) for v in gt.values()])
    country = recs["country"].to_numpy()

    # subgroups of true pairs, defined on the RAW right record (identical for every variant)
    rn = raw["business_name"].to_numpy()[g_c]
    ra = raw["business_address"].to_numpy()[g_c]
    subgroups = {
        "nonlatin_name": np.fromiter((_nonlatin(s) for s in rn), bool, len(rn)),
        "web_name": np.fromiter((bool(_WEB.search(s)) for s in rn), bool, len(rn)),
        "empty_addr": np.fromiter((not s.strip() for s in ra), bool, len(ra)),
        "letter_run_name": np.fromiter((bool(_RUN.search(s)) for s in rn), bool, len(rn)),
    }
    del raw

    specs = cfg["blocking"]["blockers"]
    keys_all, timings = [], {}
    for i, spec in enumerate(specs):
        t = time.time()
        b = build("blocker", spec)
        b.fit(Dataset("train", recs, train_gt))
        r, c, v = b.candidate_rows(recs)
        keys_all.append(r * n + c)
        tag = f"b{i}_{spec['name']}"
        timings[tag] = round(time.time() - t, 1)
        print(f"blocker {tag}: {len(r):,} pairs in {timings[tag]:.0f}s (peak {p1.peak_gb():.1f} GB)", flush=True)
    keys = np.unique(np.concatenate(keys_all))
    mc = cfg["blocking"].get("max_candidates")
    assert not mc, "max_candidates is not applied here; evaluate without it"

    hit = np.isin(keys, gt_keys, assume_unique=True)
    found_pair = np.isin(gt_keys, keys, assume_unique=True)
    s1r = keys // n
    n_c = np.bincount(s1r, minlength=n)[s1_rows]
    found = np.bincount(s1r[hit], minlength=n)[s1_rows]
    truth = truth_n[s1_rows]
    rr = np.where(truth > 0, found / np.maximum(truth, 1), 0.0)
    f = np.where(truth == 0, 1.0, np.where(found > 0, 1.25 * rr / (0.25 + rr), 0.0))
    m = {"pair_recall": hit.sum() / len(gt_keys), "oracle_f05": f.mean(), "cands_mean": n_c.mean(),
         "cands_p95": float(np.quantile(n_c, 0.95)), "cands_max": int(n_c.max())}
    for cn in sorted(set(country[s1_rows])):
        sel = country[s1_rows] == cn
        m[f"recall_{cn}"] = found[sel].sum() / max(1, truth[sel].sum())
    for name, mask in subgroups.items():
        m[f"recall_{name}"] = found_pair[mask].mean() if mask.any() else float("nan")
        m[f"n_{name}"] = int(mask.sum())
    m = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}

    head = (f"\n## {a.label} — {dt.datetime.now():%Y-%m-%d %H:%M} — config={a.config} qfrac={a.qfrac} "
            f"BER_THREADS={os.environ['BER_THREADS']} host={os.uname().nodename}\n")
    lines = [head,
             f"- records {n:,} (S1 queries {len(s1_rows):,}, right {n - len(s1_rows):,}); GT pairs {len(gt_keys):,}",
             f"- blocking time (s): {timings}; peak RSS {p1.peak_gb():.1f} GB; normalizer stats: {nstats}\n",
             p1.md_table([{"label": a.label, **{k: v for k, v in m.items() if not k.startswith("n_")}}]),
             "\nsubgroup sizes (true pairs): " + ", ".join(f"{k[2:]}={v:,}" for k, v in m.items() if k.startswith("n_"))]
    out = Path(a.out)
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# T04 (p2) full-density blocking evaluation — generated by exploration/p2/blk_eval_p2.py\n")
    with open(out, "a") as fh:
        fh.write("\n".join(lines) + "\n")
    jp = Path(a.json) if a.json else REPO / "exploration/p2/blk_eval_json" / f"{a.label}_q{a.qfrac}.json"
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps({"label": a.label, "config": a.config, "qfrac": a.qfrac, "metrics": m, "timings_s": timings,
                              "peak_rss_gb": round(p1.peak_gb(), 1), "normalizer_stats": nstats,
                              "host": os.uname().nodename, "time": f"{dt.datetime.now():%Y-%m-%d %H:%M}"}, indent=1,
                             default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
