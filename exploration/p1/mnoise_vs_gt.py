"""Are p1_mnoise's mined copies clean? On TRAIN (labelled): precision of the mined likely-copy pairs, and mined vs
label-based add rates per word (p1, Sun 27 Sep). Uses the dense-CV candidate pairs (base parquet) and GT labels."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src"))
from ber.config import env, load_config
from ber.data import load_part
from ber.pipeline import Pipeline
from ber.features.p1_mnoise import MinedNoiseFeatures
from ber.features.p1_noise import _field_matrix

REPO = Path(__file__).resolve().parents[2]
art = Path(env("BER_ARTIFACTS"))
cfg = load_config(str(REPO / "code/business_entity_resolution/configs/best.yaml"), [])
pipe = Pipeline(cfg, art)
full = load_part(env("BER_DATA_DIR"), "train", cache_dir=art / "cache")
pipe.normalizer.fit(full)
recs = pipe._normalize(full).records
b = pd.read_parquet(art / "dense_cv" / "base_c5c7a065621efecc.parquet", columns=["s1_id", "ri", "label"])
li, ri, lab = b["s1_id"].to_numpy().astype(np.int64), b["ri"].to_numpy().astype(np.int64), b["label"].to_numpy()
fs = MinedNoiseFeatures()
m = fs._mine(recs, li, ri)
codes, groups = pd.factorize(recs["country"].astype(str).to_numpy())
for k, g in enumerate(groups):
    s = m[codes[li[m]] == k]
    print(f"{g}: mined {len(s)}, precision {lab[s].mean():.4f}, share of true pairs mined {lab[s].sum() / lab[codes[li] == k].sum():.3f}")
B, vocab = _field_matrix(recs, "name_norm")
B = B.tocsr()
mined = fs._rates(B, li[m], ri[m], codes[li[m]], len(groups))
t = np.flatnonzero(lab == 1)
gt = fs._rates(B, li[t], ri[t], codes[li[t]], len(groups))
fp = m[lab[m] == 0]
BR, BL = B[ri[fp]], B[li[fp]]
only_r = np.asarray((BR - BR.multiply(BL)).sum(0)).ravel()
rows = []
for k, g in enumerate(groups):
    top = np.argsort(-mined["add"][k])
    for j in [j for j in top if mined["add"][k, j] > 0.3][:15]:
        rows.append({"group": g, "word": vocab[j], "mined_add": round(float(mined["add"][k, j]), 3),
                     "gt_add": round(float(gt["add"][k, j]), 3), "in_mined_FP_right_only": int(only_r[j])})
print(pd.DataFrame(rows).to_string(index=False))
