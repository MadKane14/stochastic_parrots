"""Gain importance of feature groups on a contiguous slice of the cached dense-CV matrix (p1, Sun 27 Sep): does the
matcher use p1_mnoise when the label-learned p1_noise rates are also present?

    python exploration/p1/feat_importance.py <base parquet stem> <feat parquet stem> ... [--rows N] [--drop regex ...]
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

art = Path("/scratch2/madhur.kankane/ber_artifacts/dense_cv")
args = sys.argv[1:]
rows = int(args[args.index("--rows") + 1]) if "--rows" in args else 4_000_000
drops = args[args.index("--drop") + 1:] if "--drop" in args else []
stems = [a for a in args if not a.startswith("--") and not a.isdigit() and a not in drops]


def head(stem):
    f = pq.ParquetFile(art / f"{stem}.parquet")
    out, n = [], 0
    for g in range(f.num_row_groups):
        t = f.read_row_group(g).to_pandas()
        out.append(t); n += len(t)
        if n >= rows:
            break
    return pd.concat(out, ignore_index=True).iloc[:rows]


base = head(stems[0])
y = base["label"].to_numpy()
X = [base[[c for c in base.columns if c.startswith("blk_")]].astype(np.float32)]
for s in stems[1:]:
    X.append(head(s).astype(np.float32))
X = pd.concat(X, axis=1)
X = X[[c for c in X.columns if not c.startswith("__") and not any(re.search(r, c) for r in drops)]]
print("rows", len(X), "features", X.shape[1], "pos rate", y.mean().round(4), flush=True)
m = lgb.train({"objective": "binary", "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 100,
               "feature_fraction": 0.8, "verbose": -1, "num_threads": 4, "seed": 0},
              lgb.Dataset(X.to_numpy(np.float32), y, feature_name=list(X.columns)), 300)
imp = pd.Series(m.feature_importance("gain"), index=X.columns)
imp = imp / imp.sum()
grp = imp.groupby(imp.index.map(lambda c: re.sub(r"^(f_p1[a-z]?_|blk_\d+_|f_p1e5?_|f_[a-z0-9]+_).*", r"\1", c))).sum()
print(grp.sort_values(ascending=False).round(4).to_string())
print("\ntop 25:\n" + imp.sort_values(ascending=False).head(25).round(4).to_string())
print("\np1_mnoise features:\n" + imp[imp.index.str.startswith("f_p1m_")].sort_values(ascending=False).round(5).to_string())
