"""Same-address sibling classes in dense-CV OOF decisions (train, labelled): precision / recall per class for two runs
(p1, Sun 27 Sep). Classes as in sibling_probe.py (same numbers + street tsr >= 90; name tsr >= 90 / 60-90 / < 60).

    python exploration/p1/sibling_oof.py <run A> <run B> [thr]
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ber.config import env, load_config
from ber.data import load_part
from ber.pipeline import Pipeline
from dense_decide import owned

REPO = Path(__file__).resolve().parents[2]
art = Path(env("BER_ARTIFACTS"))
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
cfg = load_config(str(REPO / "code/business_entity_resolution/configs/best.yaml"), [])
pipe = Pipeline(cfg, art)
full = load_part(env("BER_DATA_DIR"), "train", cache_dir=art / "cache")
pipe.normalizer.fit(full)
tr = pipe._normalize(full).records
DIG = re.compile(r"\d+")
a = tr["addr_norm"].astype(str)
num = a.map(lambda s: "|".join(sorted(set(DIG.findall(s)))))
code, has = pd.factorize(num)[0], num.ne("").to_numpy()
street = a.str.replace(r"\d+", " ", regex=True).str.split().str.join(" ").to_numpy(object)
name = tr["name_norm"].astype(str).to_numpy(object)
ctry = tr["country"].to_numpy()
cls_of = {}
for run in sys.argv[1:3]:
    o = pd.read_parquet(art / "dense_cv" / "oof" / f"{run}.parquet")
    li, ri = o["li"].to_numpy().astype(np.int64), o["ri"].to_numpy().astype(np.int64)
    lab, p = o["label"].to_numpy(), o["oof"].to_numpy().astype(np.float64)
    if "cls" not in cls_of:
        c = np.flatnonzero(has[li] & has[ri] & (code[li] == code[ri]))
        st = process.cpdist(street[li[c]], street[ri[c]], scorer=fuzz.token_set_ratio, workers=4)
        c = c[st >= 90]
        nm = process.cpdist(name[li[c]], name[ri[c]], scorer=fuzz.token_set_ratio, workers=4)
        cls = np.full(len(li), "other", dtype=object)
        cls[c] = np.where(nm >= 90, "same", np.where(nm >= 60, "partial", "renamed"))
        cls_of["cls"], cls_of["key"] = cls, (li[:1000].sum(), ri[:1000].sum())
    assert cls_of["key"] == (li[:1000].sum(), ri[:1000].sum()), "runs must share the pair order"
    cls = cls_of["cls"]
    own = owned(li, ri, p)
    pred = np.zeros(len(li), bool); pred[own[p[own] >= thr]] = True
    rows = []
    for k in ("same", "partial", "renamed", "other"):
        for cc in sorted(set(ctry)):
            m = (cls == k) & (ctry[li] == cc)
            tp = (pred & m & (lab == 1)).sum(); fp = (pred & m & (lab == 0)).sum(); fn = (~pred & m & (lab == 1)).sum()
            rows.append({"class": k, "country": cc, "pairs": int(m.sum()), "true": int((m & (lab == 1)).sum()),
                         "TP": int(tp), "FP": int(fp), "FN": int(fn),
                         "precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4)})
    print(f"\n## {run} thr {thr}\n" + pd.DataFrame(rows).to_string(index=False), flush=True)
