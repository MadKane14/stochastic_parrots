"""T02 (p1): recall / size / runtime of the p1_idf_overlap blocker on the FULL labelled train set.

Why full train: the dev split val_v1@0.1 keeps only 10% of the distractors, so top-k competition there is
~10x easier than in test. Full train (2.2M S1 x 10.3M S2/S3) has the same density as test (1.7M x 10M), so its
recall and runtime are the projection for test. No model is fitted, no labels are used by the blocker;
labels are only used here to score the candidate sets.

Usage (from repo root, env as in CLAUDE.md):
    nice -n 10 python exploration/p1/blk_eval.py --kmax 40 --max_df 20000 [--frac 1.0]
Writes exploration/p1/blk_eval_out.md (appends a section per invocation).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code/business_entity_resolution/src"))
os.environ.setdefault("BER_THREADS", "8")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, os.environ["BER_THREADS"])

from ber.blocking.p1_compound import CompoundKeyBlocker  # noqa: E402
from ber.blocking.p1_idf_overlap import IdfOverlapBlocker  # noqa: E402
from ber.config import env  # noqa: E402
from ber.data import load_part, unit_hash  # noqa: E402
from ber.normalize.basic import BasicNormalizer  # noqa: E402

CHANNELS = {"name": ["name_norm"], "addr": ["addr_norm"], "both": ["name_norm", "addr_norm"]}
LEGAL = ["inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation", "co", "company", "plc", "llp"]


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def load_normalized(frac: float) -> tuple[pd.DataFrame, dict]:
    art = Path(env("BER_ARTIFACTS"))
    ds = load_part(env("BER_DATA_DIR"), "train", cache_dir=art / "cache")
    cache = art / "cache" / "p1_norm_train_full.parquet"
    if cache.exists():
        recs = pd.read_parquet(cache)
    else:
        t = time.time()
        recs = BasicNormalizer(LEGAL).transform(ds.records)[["entity_id", "source", "country", "name_norm", "addr_norm"]]
        recs.to_parquet(cache)
        print(f"normalized {len(recs):,} records in {time.time() - t:.0f}s", flush=True)
    gt = ds.gt
    if frac < 1:   # same subsampling logic as the harness: frac of S1 + their matches + frac of distractors
        s1_keep = {s for s in gt if unit_hash(s, 11) < frac}
        matched = set().union(*gt.values())
        keep_right = {c for s in s1_keep for c in gt[s]}
        is_right = recs["source"] != 1
        rid = recs["entity_id"]
        mask = rid.isin(s1_keep) | rid.isin(keep_right) | (is_right & ~rid.isin(matched)
                                                           & rid.map(lambda r: unit_hash(r, 11) < frac))
        recs = recs[mask].reset_index(drop=True)
        gt = {s: gt[s] for s in s1_keep}
    return recs.reset_index(drop=True), gt


def md_table(rows: list[dict]) -> str:
    cols = list(rows[0])
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows])


def miss_analysis(recs, gt_keys, chan, ks, n, n_show=40, seed=0):
    """Profile true pairs missed by the union of channels at the given k's."""
    keys = np.unique(np.concatenate([chan[ch][0][chan[ch][1] < k] for ch, k in ks.items()]))
    missed = gt_keys[~np.isin(gt_keys, keys)]
    s1r, rr = missed // n, missed % n
    L, R = recs.iloc[s1r].reset_index(drop=True), recs.iloc[rr].reset_index(drop=True)
    tok = lambda s: set(s.split())
    digits = lambda s: {t for t in s.split() if t.isdigit()}
    prof = pd.DataFrame({
        "right_addr_empty": R["addr_norm"].str.len() == 0,
        "right_name_nonascii": ~R["name_norm"].map(str.isascii),
        "right_addr_nonascii": ~R["addr_norm"].map(str.isascii),
        "share_name_tok": [bool(tok(a) & tok(b)) for a, b in zip(L["name_norm"], R["name_norm"])],
        "share_addr_tok": [bool(tok(a) & tok(b)) for a, b in zip(L["addr_norm"], R["addr_norm"])],
        "share_number": [bool(digits(a) & digits(b)) for a, b in zip(L["addr_norm"], R["addr_norm"])],
        "name_exact": (L["name_norm"] == R["name_norm"]).to_numpy(),
    })
    print(f"missed {len(missed):,} of {len(gt_keys):,} true pairs at {ks}")
    print("share of missed pairs with property:\n" + prof.mean().round(4).to_string())
    print("by country:\n" + L["country"].value_counts(normalize=True).round(3).to_string())
    idx = np.random.default_rng(seed).choice(len(missed), size=min(n_show, len(missed)), replace=False)
    for i in idx:
        print(f"- `{L.name_norm[i]}` | `{L.addr_norm[i]}` [{L.country[i]}]  <>  `{R.name_norm[i]}` | `{R.addr_norm[i]}` [{R.entity_id[i][:2]}]")


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmax", type=int, default=40)
    ap.add_argument("--max_df", type=int, default=20000)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--channels", default="name,addr,both")
    ap.add_argument("--qfrac", type=float, default=1.0,
                    help="keep all right records but only this fraction of S1 queries (unbiased for forward top-k)")
    ap.add_argument("--krev", type=int, default=3)
    ap.add_argument("--num_variants", action="store_true")
    ap.add_argument("--max_df_unigram", type=int, default=0)
    ap.add_argument("--save", default="", help="save per-channel (key, rank) to $BER_ARTIFACTS/p1_blk/<save>.npz")
    ap.add_argument("--load", default="", help="skip blocking; load a --save file")
    ap.add_argument("--misses", default="", help="combo like name=20,addr=20,both=20: print miss analysis")
    a = ap.parse_args()
    t0 = time.time()
    recs, gt = load_normalized(a.frac)
    if a.qfrac < 1:
        is_s1 = recs["source"] == 1
        keep_s1 = recs["entity_id"].map(lambda e: unit_hash(e, 5) < a.qfrac) & is_s1
        recs = recs[keep_s1 | ~is_s1].reset_index(drop=True)
        kept = set(recs.loc[recs["source"] == 1, "entity_id"])
        gt = {s: v for s, v in gt.items() if s in kept}
    print(f"loaded {len(recs):,} records, {len(gt):,} S1  ({time.time() - t0:.0f}s, peak {peak_gb():.1f} GB)", flush=True)

    n = len(recs)
    row_of = pd.Series(np.arange(n), index=recs["entity_id"].to_numpy())
    s1_rows = np.flatnonzero((recs["source"] == 1).to_numpy())
    g_s1 = [s for s in gt for _ in gt[s]]
    g_c = [c for s in gt for c in gt[s]]
    gt_keys = np.unique(row_of[g_s1].to_numpy() * n + row_of[g_c].to_numpy())
    truth_n = np.zeros(n, np.int32)
    np.add.at(truth_n, row_of[list(gt)].to_numpy(), [len(v) for v in gt.values()])
    country = recs["country"].to_numpy()

    # ---- run each channel once at k = kmax, keep (key, rank) ----
    chan = {}
    timings = {}
    store = Path(env("BER_ARTIFACTS")) / "p1_blk"
    if a.load:
        z = np.load(store / f"{a.load}.npz")
        chan = {k[4:]: (z[k], z["rank" + k[3:]]) for k in z.files if k.startswith("key_")}
        timings = {k: float(v) for k, v in zip(z["t_names"], z["t_vals"])}
    for name in ([] if a.load else a.channels.split(",")):
        t = time.time()
        if name == "cmp":
            b = CompoundKeyBlocker(k=a.kmax, k_rev=0, max_df=a.max_df, num_variants=a.num_variants,
                                   max_df_unigram=a.max_df_unigram or None)
        elif name == "cmprev":
            b = CompoundKeyBlocker(k=0, k_rev=a.krev, max_df=a.max_df, num_variants=a.num_variants,
                                   max_df_unigram=a.max_df_unigram or None)
        else:
            b = IdfOverlapBlocker(fields=CHANNELS[name], k=a.kmax, max_df=a.max_df)
        r, c, v = b.candidate_rows(recs)
        order = np.lexsort((-v, r))
        r, c = r[order], c[order]
        start = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
        rank = np.arange(len(r)) - np.repeat(start, np.diff(np.r_[start, len(r)]))
        if name == "cmprev":   # reverse top-k is per right record: all of it is in, whatever the forward k
            rank[:] = 0
        chan[name] = (r * n + c, rank.astype(np.int16))
        timings[name] = time.time() - t
        print(f"channel {name}: {len(r):,} pairs in {timings[name]:.0f}s (peak {peak_gb():.1f} GB)", flush=True)

    if a.save:
        store.mkdir(parents=True, exist_ok=True)
        np.savez(store / f"{a.save}.npz", **{f"key_{c}": v[0] for c, v in chan.items()},
                 **{f"rank_{c}": v[1] for c, v in chan.items()},
                 t_names=np.array(list(timings)), t_vals=np.array(list(timings.values())))
    if a.misses:
        miss_analysis(recs, gt_keys, chan, dict((kv.split("=")[0], int(kv.split("=")[1])) for kv in a.misses.split(",")), n)
        return

    def metrics(ks: dict) -> dict:
        keys = np.unique(np.concatenate([chan[ch][0][chan[ch][1] < k] for ch, k in ks.items()]))
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

    combos = []
    for k in (5, 10, 15, 20, 30, 40, 50):
        if k > a.kmax:
            continue
        for ch in chan:
            combos.append({ch: k})
        if len(chan) > 1:
            combos.append({ch: k for ch in chan})
    if {"cmp", "cmprev"} <= set(chan):
        combos = [c for c in combos if "cmprev" not in c or len(c) > 1]
        combos += [{"cmp": k, "cmprev": 1} for k in (5, 10, 15, 20, 30, 40, 50) if k <= a.kmax]
    if {"name", "addr", "both"} <= set(chan):
        for kn, ka, kb in ((10, 10, 20), (15, 15, 20), (10, 20, 20), (20, 20, 10), (20, 20, 30), (15, 25, 25)):
            if max(kn, ka, kb) <= a.kmax:
                combos.append({"name": kn, "addr": ka, "both": kb})
    rows = []
    for ks in combos:
        m = metrics(ks)
        rows.append({"channels(k)": " + ".join(f"{c}@{k}" for c, k in ks.items()), **{k: round(float(v), 4) for k, v in m.items()}})
        print(rows[-1], flush=True)

    s1_n = len(s1_rows)
    right_n = n - s1_n
    lines = [f"\n## Run {dt.datetime.now():%Y-%m-%d %H:%M} — channels={a.channels} frac={a.frac} qfrac={a.qfrac} "
             f"kmax={a.kmax} krev={a.krev} max_df={a.max_df} num_variants={a.num_variants} max_df_unigram={a.max_df_unigram} "
             f"BER_THREADS={os.environ['BER_THREADS']}\n",
             f"- records {n:,} (S1 {s1_n:,}, right {right_n:,}); GT pairs {len(gt_keys):,}",
             f"- blocking time per channel at k={a.kmax} (s): " + ", ".join(f"{k}={v:.0f}" for k, v in timings.items())
             + f"; total {sum(timings.values()):.0f}s; peak RSS {peak_gb():.1f} GB",
             f"- test projection (1.73M S1 x 9.97M right, same scale): ~{sum(timings.values()) * 1.73e6 / max(1, s1_n):.0f}s "
             "for blocking the test set (the harness's test mode also blocks full train for fitting)\n",
             md_table(rows)]
    out = REPO / "exploration/p1/blk_eval_out.md"
    if not out.exists():
        out.write_text("# Blocker evaluation on full train (p1, T02) — generated by exploration/p1/blk_eval.py\n")
    with open(out, "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
