"""Same-address candidates with different names ("siblings" vs renamed copies): how often are they true in train, and
how common are they per S1 in test by country? (p1, Sun 27 Sep; label-free on test)

Records normalized by best.yaml's normalizer (pipeline cache). A pair is same-address when both addresses have numbers,
the number sets are equal, and token_set_ratio of the number-free addresses >= 90. Name classes by token_set_ratio of
name_norm: same (>= 90), partial (60-90), renamed (< 60).
Train: all dense-CV candidates of best.yaml's blocker (base parquet), with labels. Test: candidate_pairs.tsv + matches of
a submission (argv[1]).
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src"))
from ber.config import env, load_config
from ber.data import load_part
from ber.pipeline import Pipeline

REPO = Path(__file__).resolve().parents[2]
art = Path(env("BER_ARTIFACTS"))
sub = Path(sys.argv[1])
base = sys.argv[2] if len(sys.argv) > 2 else "base_c5c7a065621efecc"
cfg = load_config(str(REPO / "code/business_entity_resolution/configs/best.yaml"), [])
pipe = Pipeline(cfg, art)
full = load_part(env("BER_DATA_DIR"), "train", cache_dir=art / "cache")
pipe.normalizer.fit(full)
DIG = re.compile(r"\d+")


def keys(recs):
    a = recs["addr_norm"].astype(str)
    num = a.map(lambda s: "|".join(sorted(set(DIG.findall(s)))))
    street = a.str.replace(r"\d+", " ", regex=True).str.split().str.join(" ")
    return pd.factorize(num)[0], num.ne("").to_numpy(), street.to_numpy(object), recs["name_norm"].astype(str).to_numpy(object)


def classify(li, ri, K):
    code, has, street, name = K
    cand = np.flatnonzero(has[li] & has[ri] & (code[li] == code[ri]))
    st = process.cpdist(street[li[cand]], street[ri[cand]], scorer=fuzz.token_set_ratio, workers=4)
    cand = cand[st >= 90]
    nm = process.cpdist(name[li[cand]], name[ri[cand]], scorer=fuzz.token_set_ratio, workers=4)
    cls = np.where(nm >= 90, "same", np.where(nm >= 60, "partial", "renamed"))
    return cand, cls


rows = []
tr = pipe._normalize(full).records
K = keys(tr)
b = pd.read_parquet(art / "dense_cv" / f"{base}.parquet", columns=["s1_id", "ri", "label"])
li, ri, lab = b["s1_id"].to_numpy().astype(np.int64), b["ri"].to_numpy().astype(np.int64), b["label"].to_numpy()
del b
cand, cls = classify(li, ri, K)
ctry = tr["country"].to_numpy()
n_s1 = pd.Series(ctry[(tr["source"] == 1).to_numpy()]).value_counts()
t = pd.DataFrame({"country": ctry[li[cand]], "cls": cls, "true": lab[cand]})
for (c, k), g in t.groupby(["country", "cls"]):
    rows.append({"part": "train", "country": c, "class": k, "pairs_per_s1": round(len(g) / n_s1[c], 4),
                 "true_rate": round(g["true"].mean(), 4), "assigned_rate": None})
del tr, K, li, ri, lab, t
te_ds = load_part(env("BER_DATA_DIR"), "test", cache_dir=art / "cache")
te = pipe._normalize(te_ds).records
K = keys(te)
idx = pd.Index(te["entity_id"].to_numpy())
cp = pd.read_csv(sub / "candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
cp.columns = ["s1", "ids"]
x = cp.assign(r=cp["ids"].str.split(",")).explode("r")
x = x[x["r"].fillna("") != ""]
li, ri = idx.get_indexer(x["s1"]), idx.get_indexer(x["r"])
mr = pd.read_csv(sub / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
mr.columns = ["s1", "ids"]
y = mr.assign(r=mr["ids"].str.split(",")).explode("r")
y = y[y["r"].fillna("") != ""]
assigned = set(zip(idx.get_indexer(y["s1"]), idx.get_indexer(y["r"])))
cand, cls = classify(li, ri, K)
ctry = te["country"].to_numpy()
n_s1 = pd.Series(ctry[(te["source"] == 1).to_numpy()]).value_counts()
asg = np.array([(a, b_) in assigned for a, b_ in zip(li[cand], ri[cand])])
t = pd.DataFrame({"country": ctry[li[cand]], "cls": cls, "asg": asg})
for (c, k), g in t.groupby(["country", "cls"]):
    rows.append({"part": "test", "country": c, "class": k, "pairs_per_s1": round(len(g) / n_s1[c], 4),
                 "true_rate": None, "assigned_rate": round(g["asg"].mean(), 4)})
out = pd.DataFrame(rows)
print(out.to_string(index=False))
