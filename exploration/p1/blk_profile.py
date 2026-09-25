"""T02 (p1): where does p1_compound spend its time? Per-family index cost (sum over indexed features of
df_S1 * df_right = number of posting hits of the sparse product) and a timed sample chunk (product vs top-k).
Runs on the largest country-string group of the full train set.

    nice -n 10 python exploration/p1/blk_profile.py [--max_df 5000] [--max_df_unigram 1000] [--num_variants]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code/business_entity_resolution/src"))
from ber.blocking.p1_compound import _binary_csr, compound_features, tokenize  # noqa: E402
from ber.blocking.p1_idf_overlap import topk_rows  # noqa: E402
from ber.config import env  # noqa: E402

FAMILIES = ["U_name", "U_addr", "P_name", "B_addr", "NW", "NN", "NV_word", "NV_name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_df", type=int, default=5000)
    ap.add_argument("--max_df_unigram", type=int, default=0)
    ap.add_argument("--num_variants", action="store_true")
    a = ap.parse_args()
    recs = pd.read_parquet(Path(env("BER_ARTIFACTS")) / "cache" / "p1_norm_train_full.parquet")
    name_tok, addr_tok = tokenize(recs["name_norm"]), tokenize(recs["addr_norm"])
    g = recs["country"].value_counts().index[0]
    in_g = (recs["country"] == g).to_numpy()
    is_s1 = (recs["source"] == 1).to_numpy()
    i1, i2 = np.flatnonzero(in_g & is_s1), np.flatnonzero(in_g & ~is_s1)
    r, keys = compound_features(name_tok, addr_tok, np.r_[i1, i2], 8, 20, 4, a.num_variants)
    col, uniq = pd.factorize(keys)
    fam_of_col = (np.asarray(uniq) >> 58).astype(int)
    X = _binary_csr(r, col, len(i1) + len(i2), len(uniq))
    X1, X2 = X[: len(i1)], X[len(i1):]
    df1 = np.asarray(X1.sum(0)).ravel()
    df2 = np.asarray(X2.sum(0)).ravel()
    df = df1 + df2
    cap = np.where(fam_of_col <= 1, a.max_df_unigram or a.max_df, a.max_df)
    idx = (df > 0) & (df <= cap)
    hits = df1 * df2 * idx
    print(f"group {g}: {len(i1):,} S1 x {len(i2):,} right; total hits {hits.sum():.3e}")
    rows = []
    for f in range(len(FAMILIES)):
        m = fam_of_col == f
        if not m.any():
            continue
        rows.append({"family": FAMILIES[f], "features": int(m.sum()), "nnz_per_rec": round(float(df[m].sum()) / X.shape[0], 1),
                     "indexed": int((m & idx).sum()), "hits": f"{hits[m].sum():.2e}",
                     "hits_share": round(float(hits[m].sum() / hits.sum()), 3),
                     "hits_from_df>1000": round(float(hits[m & (df > 1000)].sum() / max(1, hits[m].sum())), 3)})
    print(pd.DataFrame(rows).to_string(index=False))
    # timed sample chunk of 5000 S1 rows (unweighted binary product: same sparsity as the real one)
    keep = sp.diags(idx.astype(np.float32))
    A = (X1[:5000] @ keep).tocsr()
    BT = (X2 @ keep).T.tocsr()
    t = time.time(); C = A @ BT; tp = time.time() - t
    t = time.time(); topk_rows(C, 50, 0.0); tk = time.time() - t
    print(f"sample chunk 5000 S1 rows: product {tp:.1f}s, result nnz {C.nnz:.3e} ({C.nnz / 5000:.0f}/row), top-k {tk:.1f}s")
    # parallel scaling: the same 24 chunks of 1000 rows with 1, 4 and 12 forked workers
    import multiprocessing as mp
    from ber.blocking import p1_idf_overlap as blk
    A24 = (X1[:24000] @ keep).tocsr()
    blk._W.update(A=A24, BT=BT, k=50, min_score=0.0)
    bounds = [(s, s + 1000) for s in range(0, 24000, 1000)]
    for w in (1, 4, 12):
        t = time.time()
        with mp.get_context("fork").Pool(w) as pool:
            pool.map(blk._chunk, bounds, chunksize=1)
        print(f"24 chunks x 1000 rows, {w:2d} workers: {time.time() - t:.1f}s wall", flush=True)


if __name__ == "__main__":
    main()
