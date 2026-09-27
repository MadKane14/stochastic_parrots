"""Quick ablation on cached dense-CV features (p1, Sun 27 Sep): train LightGBM on the first N rows, evaluate pair-level
logloss / AUC / AP on the next M rows (different S1 entities: rows are grouped by blocking output), per variant.
Indicative only — the decision metric stays dense CV macro F0.5."""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

art = Path("/scratch2/madhur.kankane/ber_artifacts/dense_cv")
N, M = 4_000_000, 2_000_000
stems = ["base_c5c7a065621efecc", "feat_p1_match_b6515fad19131118", "feat_p1_extra_62dd598f314af756",
         "feat_p1_e5_e4be97d99003c15c", "feat_p1_noise_3e501c769538748d", "feat_p1_ambig_0b3c41c9f6596b99",
         "feat_p1_mnoise_b0857f985709ee1c"]


def head(stem, n):
    f = pq.ParquetFile(art / f"{stem}.parquet")
    out, k = [], 0
    for g in range(f.num_row_groups):
        t = f.read_row_group(g).to_pandas(); out.append(t); k += len(t)
        if k >= n:
            break
    return pd.concat(out, ignore_index=True).iloc[:n]


base = head(stems[0], N + M)
y = base["label"].to_numpy()
s1 = base["s1_id"].to_numpy()
X = pd.concat([base[[c for c in base.columns if c.startswith("blk_")]].astype(np.float32)] +
              [head(s, N + M).astype(np.float32) for s in stems[1:]], axis=1)
X = X[[c for c in X.columns if not c.startswith("__")]]
tr = np.arange(N); te = np.arange(N, N + M)
tr = tr[~np.isin(s1[tr], s1[te])]                     # no S1 entity on both sides
variants = {
    "T18 (all, no mnoise)": [r"^f_p1m_"],
    "T27 (mnoise, -core_n)": [r"^f_p1_core_n_"],
    "T28 (T27 - supervised p1_noise)": [r"^f_p1_core_n_", r"^f_p1n_(n|a)_", r"^f_p1n_total_unexpl", r"^f_p1n_rec_"],
    "T28b (T28 - p1_noise unsup too)": [r"^f_p1_core_n_", r"^f_p1n_"],
}
for name, drops in variants.items():
    cols = [c for c in X.columns if not any(re.search(r, c) for r in drops)]
    m = lgb.train({"objective": "binary", "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 100,
                   "feature_fraction": 0.8, "verbose": -1, "num_threads": 4, "seed": 0},
                  lgb.Dataset(X.iloc[tr][cols].to_numpy(np.float32), y[tr]), 400)
    p = m.predict(X.iloc[te][cols].to_numpy(np.float32))
    print(f"{name:34s} feats {len(cols):3d}  logloss {log_loss(y[te], np.clip(p, 1e-7, 1 - 1e-7)):.5f}  "
          f"AUC {roc_auc_score(y[te], p):.6f}  AP {average_precision_score(y[te], p):.6f}", flush=True)
