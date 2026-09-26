"""T11 (p3): full-density blocking eval of config blocker UNIONS at TRUE density, fitting in 30 GB.

Differences from exploration/p1/blk_eval.py and exploration/p3/blk_union.py:
  * `--dense` keeps ALL train S1 records in the data (not only the qfrac queries). Recall / candidate counts are
    measured on the query S1 (unit_hash(s1, 5) < qfrac) only, but every S1 competes in the reverse direction, so
    reverse top-k_rev is no longer optimistic (blk_union at qfrac 0.1 inflated p1_compound to 143 cands/S1 vs 33
    at qfrac 1, T05 report). Label-fitted components (p2_translit's token map) are fitted on the labels of the
    non-query S1 only, exactly as with qfrac < 1 — at plain qfrac 1 blk_eval.py would fit them WITHOUT labels.
  * one subprocess per (blocker, country-string group), reading a slim column/row subset of the normalized cache
    through pyarrow: peak RSS = one group's blocking, not the 17 GB object-string load + blocking. Blockers group
    by the `country` string themselves (generic grouping), so per-group runs give the same candidates. Values are
    read from the data, never hard-coded.
  * several configs per invocation share cached blocker outputs (fingerprint = normalizer + blocker spec + source
    code + qfrac + dense + group) -> each channel is computed once.
  * per-S1 caps emulate Pipeline._block exactly (best rank over blockers); recall is also reported on raw-defined
    subsets of true pairs (right address empty / right name non-ASCII after normalization / no shared name token),
    plus each blocker's unique contribution to the uncapped union.
  * T15: blockers exposing `candidate_rows_prov` (p3_compound_rc, p3_chargram) also store which pairs came only from
    the reverse direction; `--rev-caps 5,10,20` then emulates their `rev_cap` option (keep all forward pairs + the
    best `rev_cap` reverse-only pairs per S1, ranks recomputed) crossed with the per-S1 caps, from one run.

Usage (repo root, Linux, env as in CLAUDE.md):
    nice -n 10 python exploration/p3/blk_dense.py --dense --qfrac 0.1 --caps 50,70,100 \
        --config code/business_entity_resolution/configs/exp/p3/T11_a.yaml [--config ... ] \
        --out exploration/p3/blk_dense_out.md [--misses 40] [--rev-caps 5,10,20]
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

import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import blk_eval as be  # noqa: E402
from ber.config import env, load_config  # noqa: E402
from ber.data import unit_hash  # noqa: E402
from ber.blocking.p3_revcap import cap_reverse  # noqa: E402
from ber.pipeline import _component_fp  # noqa: E402
from ber.registry import build  # noqa: E402

SLIM = ["entity_id", "source", "country", "name_norm", "addr_norm", "name_core"]
GROUP = "country"


def art() -> Path:
    return Path(env("BER_ARTIFACTS"))


def slim_path(cfg: dict, qfrac: float) -> Path:
    fp = _component_fp(build("normalizer", cfg["normalize"]))
    return art() / "p3_dense" / f"slim_{fp}_q{qfrac}.parquet"


def prep(cfg: dict, qfrac: float):
    """Normalize full train (normalizer fitted on non-query labels, blk_eval.load_normalized) -> slim parquet."""
    out = slim_path(cfg, qfrac)
    recs, _ = be.load_normalized(1.0, build("normalizer", cfg["normalize"]), qfrac)
    out.parent.mkdir(parents=True, exist_ok=True)
    recs[[c for c in SLIM if c in recs.columns]].to_parquet(out.with_suffix(".tmp"))
    out.with_suffix(".tmp").replace(out)
    print(f"slim cache {out} ({len(recs):,} rows, peak {be.peak_gb():.1f} GB)", flush=True)


def query_mask(ids: np.ndarray, is_s1: np.ndarray, qfrac: float) -> np.ndarray:
    """True for query S1 rows (unit_hash(s1, 5) < qfrac, = blk_eval's sampling)."""
    q = np.zeros(len(ids), bool)
    s1 = np.flatnonzero(is_s1)
    q[s1] = np.fromiter((unit_hash(e, 5) < qfrac for e in ids[s1]), bool, len(s1))
    return q


def store_path(cfg: dict, i: int, qfrac: float, dense: bool, group: str) -> Path:
    spec = cfg["blocking"]["blockers"][i]
    fp = "|".join([_component_fp(build("normalizer", cfg["normalize"])), _component_fp(build("blocker", spec)),
                   json.dumps(spec, sort_keys=True), str(qfrac), str(dense), group])
    return art() / "p3_dense" / f"{spec['name']}_{hashlib.sha256(fp.encode()).hexdigest()[:12]}.npz"


def run_one(cfg: dict, i: int, qfrac: float, dense: bool, group: str, out: Path):
    """Blocker i on one country group; save global pair keys (s1_pos * N + right_pos) and rank within S1."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    t0 = time.time()
    tab = pq.read_table(slim_path(cfg, qfrac))
    n_all = tab.num_rows
    in_g = pc.equal(tab[GROUP], group).to_numpy(zero_copy_only=False)
    is_s1 = (tab["source"].to_numpy() == 1)
    keep = in_g & (~is_s1 | (True if dense else query_mask(tab["entity_id"].to_numpy(zero_copy_only=False), is_s1, qfrac)))
    pos = np.flatnonzero(keep)
    recs = tab.take(pos).to_pandas()
    del tab
    print(f"group {group}: {len(recs):,} of {n_all:,} rows ({time.time() - t0:.0f}s, peak {be.peak_gb():.1f} GB)",
          flush=True)
    b = build("blocker", cfg["blocking"]["blockers"][i])
    t = time.time()
    if hasattr(b, "candidate_rows_prov"):            # candidates + reverse-only flag (rev_cap emulation)
        r, c, v, is_rev = b.candidate_rows_prov(recs)
        keep = cap_reverse(r, v, is_rev, getattr(b, "rev_cap", None))
        r, c, v, is_rev = r[keep], c[keep], v[keep], is_rev[keep]
    else:
        r, c, v = b.candidate_rows(recs)
        is_rev = np.zeros(len(r), bool)
    secs = time.time() - t
    order = np.lexsort((-v, r))
    r, c, is_rev = r[order], c[order], is_rev[order]
    start = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
    rank = np.arange(len(r)) - np.repeat(start, np.diff(np.r_[start, len(r)]))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".tmp.npz"), key=pos[r].astype(np.int64) * n_all + pos[c], rank=rank.astype(np.int32),
             is_rev=is_rev, secs=secs, peak_gb=be.peak_gb(), n=n_all)
    out.with_suffix(".tmp.npz").replace(out)
    print(f"blocker {i} group {group}: {len(r):,} pairs in {secs:.0f}s, peak {be.peak_gb():.1f} GB -> {out}", flush=True)


def union_keys(chans: list[tuple[np.ndarray, np.ndarray]], n: int, cap: int | None) -> np.ndarray:
    """Unique pair keys of the union; with `cap`, per S1 the `cap` pairs with the smallest best rank
    (= Pipeline._block with blocking.max_candidates; ties broken by key)."""
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


def rev_capped(chan: tuple, n: int, cap: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Emulate a blocker's `rev_cap` on its stored (key, rank, is_rev): keep forward pairs + the `cap` best-ranked
    reverse-only pairs per S1, then recompute the rank within S1 over the kept pairs (what the pipeline sees)."""
    key, rank, is_rev = chan
    if cap is None or not is_rev.any():
        return key, rank
    s1 = key // n
    keep = cap_reverse(s1, -rank.astype(np.float64), is_rev, cap)
    key, rank, s1 = key[keep], rank[keep], s1[keep]
    o = np.lexsort((rank, s1))
    key, s1 = key[o], s1[o]
    start = np.r_[0, np.flatnonzero(np.diff(s1)) + 1] if len(s1) else np.zeros(0, np.int64)
    new_rank = np.arange(len(s1)) - np.repeat(start, np.diff(np.r_[start, len(s1)]))
    return key, new_rank.astype(np.int32)


def read_gt() -> dict:
    df = pd.read_csv(Path(env("BER_DATA_DIR")) / "train" / "train_ground_truth.tsv", sep="\t", dtype=str,
                     keep_default_na=False, na_filter=False)
    return {s: [x for x in m.split(",") if x] for s, m in zip(df.iloc[:, 0], df.iloc[:, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="eval", choices=["eval", "one", "prep"])
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--qfrac", type=float, default=0.1)
    ap.add_argument("--dense", action="store_true", help="keep all S1 (true reverse density); score query S1 only")
    ap.add_argument("--i", type=int, default=0)
    ap.add_argument("--group", default="")
    ap.add_argument("--caps", default="", help="extra per-S1 caps, e.g. 50,70,100")
    ap.add_argument("--misses", type=int, default=0, help="print this many missed pairs of the first config (config cap)")
    ap.add_argument("--rev-caps", default="", help="reverse-only caps per S1 to emulate, e.g. 5,10,20 (T15)")
    ap.add_argument("--out", default=str(REPO / "exploration/p3/blk_dense_out.md"))
    a = ap.parse_args()
    cfgs = [load_config(c) for c in a.config]
    if a.mode == "prep":
        return prep(cfgs[0], a.qfrac)
    if a.mode == "one":
        return run_one(cfgs[0], a.i, a.qfrac, a.dense, a.group,
                       store_path(cfgs[0], a.i, a.qfrac, a.dense, a.group))

    base = [sys.executable, __file__]
    sub = lambda *x: subprocess.run([*base, *x], check=True)  # noqa: E731
    for path, cfg in zip(a.config, cfgs):
        for spec in cfg["blocking"]["blockers"]:
            if (spec.get("params") or {}).get("group_by", GROUP) != GROUP:
                sys.exit(f"{spec['name']}: per-group runs need group_by={GROUP}")
        if not slim_path(cfg, a.qfrac).exists():
            sub("prep", "--config", path, "--qfrac", str(a.qfrac))
    run_log = {}
    for path, cfg in zip(a.config, cfgs):
        groups = sorted(set(pq.read_table(slim_path(cfg, a.qfrac), columns=[GROUP])[GROUP].to_pylist()))
        for i, spec in enumerate(cfg["blocking"]["blockers"]):
            for g in groups:
                p = store_path(cfg, i, a.qfrac, a.dense, g)
                if not p.exists():
                    t = time.time()
                    sub("one", "--config", path, "--qfrac", str(a.qfrac), "--i", str(i), "--group", g,
                        *(["--dense"] if a.dense else []))
                    run_log[p.name] = time.time() - t

    t0 = time.time()
    gt_all = read_gt()
    sections = []
    for path, cfg in zip(a.config, cfgs):
        tab = pq.read_table(slim_path(cfg, a.qfrac), columns=["entity_id", "source", "country", "name_norm", "addr_norm"])
        ids = tab["entity_id"].to_numpy(zero_copy_only=False)
        n = len(ids)
        is_s1 = tab["source"].to_numpy() == 1
        q = query_mask(ids, is_s1, a.qfrac)
        country = tab["country"].to_numpy(zero_copy_only=False)
        row_of = pd.Series(np.arange(n), index=ids)
        q_ids = ids[q]
        gt = {s: gt_all.get(s, []) for s in q_ids}
        g_s1 = np.array([s for s in q_ids for _ in gt[s]], dtype=object)
        g_c = np.array([c for s in q_ids for c in gt[s]], dtype=object)
        gt_keys = np.unique(row_of[g_s1].to_numpy() * n + row_of[g_c].to_numpy())
        truth_n = np.zeros(n, np.int32)
        np.add.at(truth_n, row_of[q_ids].to_numpy(), [len(gt[s]) for s in q_ids])
        q_rows = np.flatnonzero(q)
        # raw-defined subsets of true pairs (right side), for per-subset recall
        rr = gt_keys % n
        sr = gt_keys // n
        name = tab["name_norm"].to_numpy(zero_copy_only=False)
        addr = tab["addr_norm"].to_numpy(zero_copy_only=False)
        del tab
        subsets = {
            "addr_empty": np.fromiter((not addr[j] for j in rr), bool, len(rr)),
            "name_nonascii": np.fromiter((not name[j].isascii() for j in rr), bool, len(rr)),
            "no_name_tok": np.fromiter((not (set(name[i].split()) & set(name[j].split())) for i, j in zip(sr, rr)),
                                       bool, len(rr)),
        }
        specs = cfg["blocking"]["blockers"]
        groups = sorted(set(country))
        chans, secs, peaks = [], [], []
        for i in range(len(specs)):
            zs = [np.load(store_path(cfg, i, a.qfrac, a.dense, g)) for g in groups]
            assert all(int(z["n"]) == n for z in zs), "record order differs between cached runs"
            k = np.concatenate([z["key"] for z in zs])
            r = np.concatenate([z["rank"] for z in zs])
            f = np.concatenate([z["is_rev"] if "is_rev" in z.files else np.zeros(len(z["key"]), bool) for z in zs])
            sel = q[k // n]                       # only query S1 are scored (per-S1 caps are per S1: exact)
            chans.append((k[sel], r[sel], f[sel]))
            secs.append(sum(float(z["secs"]) for z in zs))
            peaks.append(max(float(z["peak_gb"]) for z in zs))

        def metrics(keys: np.ndarray) -> dict:
            hit = np.isin(keys, gt_keys, assume_unique=True)
            s1r = keys // n
            n_c = np.bincount(s1r, minlength=n)[q_rows]
            found = np.bincount(s1r[hit], minlength=n)[q_rows]
            truth = truth_n[q_rows]
            rrate = np.where(truth > 0, found / np.maximum(truth, 1), 0.0)
            f = np.where(truth == 0, 1.0, np.where(found > 0, 1.25 * rrate / (0.25 + rrate), 0.0))
            got = np.isin(gt_keys, keys[hit], assume_unique=True)
            out = {"pair_recall": got.mean(), "oracle_f05": f.mean(), "cands_mean": n_c.mean(),
                   "cands_p95": np.quantile(n_c, 0.95), "cands_max": n_c.max()}
            for cn in groups:
                m = country[q_rows] == cn
                out[f"recall_{cn}"] = found[m].sum() / max(1, truth[m].sum())
            for nm, m in subsets.items():
                out[f"rec_{nm}"] = got[m].mean() if m.any() else float("nan")
            return out

        names = [f"b{i}_{s['name']}" for i, s in enumerate(specs)]
        rows = [{"candidates": nm, **metrics(np.unique(ch[0]))} for nm, ch in zip(names, chans)]
        rev_share = [float(ch[2].mean()) if len(ch[2]) else 0.0 for ch in chans]
        chans2 = [(ch[0], ch[1]) for ch in chans]
        cfg_cap = cfg["blocking"].get("max_candidates")
        caps = sorted({int(x) for x in a.caps.split(",") if x} | ({int(cfg_cap)} if cfg_cap else set()))
        full = union_keys(chans2, n, None)
        if len(chans2) > 1:
            rows.append({"candidates": "union", **metrics(full)})
            for i, nm in enumerate(names):   # unique contribution: recall lost when blocker i is removed
                rest = union_keys([c for j, c in enumerate(chans2) if j != i], n, None)
                rows.append({"candidates": f"union - {nm}", **metrics(rest)})
        for cap in caps:
            rows.append({"candidates": f"union cap {cap}" + (" (config)" if cap == cfg_cap else ""),
                         **metrics(union_keys(chans2, n, cap))})
        rev_caps = [int(x) for x in a.rev_caps.split(",") if x]
        for rc in rev_caps:   # T15: reverse-flood caps (per blocker with provenance) x per-S1 caps
            rchans = [rev_capped(ch, n, rc) for ch in chans]
            for cap in [None, *caps]:
                rows.append({"candidates": f"rev_cap {rc}" + (f" + cap {cap}" if cap else ""),
                             **metrics(union_keys(rchans, n, cap))})
        rows = [{k: (round(float(v), 4) if not isinstance(v, str) else v) for k, v in r.items()} for r in rows]
        for r in rows:
            print(r, flush=True)
        n_subset = {nm: int(m.sum()) for nm, m in subsets.items()}
        lines = [f"\n## Run {dt.datetime.now():%Y-%m-%d %H:%M} — config={path} qfrac={a.qfrac} dense={a.dense} "
                 f"BER_THREADS={os.environ['BER_THREADS']} (exploration/p3/blk_dense.py)\n",
                 f"- normalize: `{json.dumps(cfg['normalize'])}`",
                 "- blockers: " + "; ".join(f"`{s['name']} {json.dumps(s.get('params') or {})}`" for s in specs),
                 f"- records {n:,} (S1 {int(is_s1.sum()):,}, query S1 {len(q_rows):,}, right {n - int(is_s1.sum()):,});"
                 f" query GT pairs {len(gt_keys):,}; subset sizes {n_subset}",
                 "- share of reverse-only pairs per blocker (query S1): "
                 + ", ".join(f"{nm}={sh:.3f}" for nm, sh in zip(names, rev_share)),
                 "- blocking time summed over groups (s) / max peak RSS (GB) per blocker: "
                 + ", ".join(f"{nm}={t:.0f}s/{p:.1f}GB" for nm, t, p in zip(names, secs, peaks))
                 + (f" — at dense=True this is the full-train blocking time (test ≈ x{1.73 / 2.21:.2f})" if a.dense
                    else " — qfrac < 1 without --dense: forward work only for the queries"),
                 "",
                 be.md_table(rows)]
        sections.append("\n".join(lines))
        print(sections[-1], flush=True)
        if a.misses and path == a.config[0]:
            keys = union_keys(chans2, n, cfg_cap)
            missed = gt_keys[~np.isin(gt_keys, keys)]
            rng = np.random.default_rng(0)
            show = rng.choice(missed, size=min(a.misses, len(missed)), replace=False)
            ml = [f"\n### Missed pairs sample ({len(missed):,} missed at cap {cfg_cap}) — {path}\n"]
            for key in show:
                i, j = key // n, key % n
                ml.append(f"- `{name[i]}` | `{addr[i]}` [{country[i]}]  <>  `{name[j]}` | `{addr[j]}` [{ids[j][:2]}]")
            sections.append("\n".join(ml))
            print(sections[-1], flush=True)
    out = Path(a.out)
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# Dense blocking evaluation on full train — generated by exploration/p3/blk_dense.py\n",
                       encoding="utf-8")
    with open(out, "a", encoding="utf-8") as f:
        f.write("\n".join(sections) + "\n")
    print(f"eval {time.time() - t0:.0f}s; new blocker runs: {json.dumps({k: round(v) for k, v in run_log.items()})}")


if __name__ == "__main__":
    main()
