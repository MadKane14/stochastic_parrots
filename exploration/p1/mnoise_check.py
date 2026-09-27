"""Sanity check of p1_mnoise's mined per-country word rates on TEST (label-free): prints add/drop rates of chosen words
per country group, from the candidate pairs of a submission (argv[1]) (p1, Sun 27 Sep)."""
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
pipe.normalizer.fit(load_part(env("BER_DATA_DIR"), "train", cache_dir=art / "cache"))
recs = pipe._normalize(load_part(env("BER_DATA_DIR"), "test", cache_dir=art / "cache")).records
idx = pd.Index(recs["entity_id"].to_numpy())
cp = pd.read_csv(Path(sys.argv[1]) / "candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
cp.columns = ["s1", "ids"]
x = cp.assign(r=cp["ids"].str.split(",")).explode("r")
x = x[x["r"].fillna("") != ""]
li, ri = idx.get_indexer(x["s1"]), idx.get_indexer(x["r"])
fs = MinedNoiseFeatures()
m = fs._mine(recs, li, ri)
codes, groups = pd.factorize(recs["country"].astype(str).to_numpy())
print("mined pairs per group:", pd.Series(groups[codes[li[m]]]).value_counts().to_dict(), "of", len(li))
B, vocab = _field_matrix(recs, "name_norm")
rates = fs._rates(B.tocsr(), li[m], ri[m], codes[li[m]], len(groups))
pos = {t: i for i, t in enumerate(vocab)}
words = ["sarl", "eurl", "sas", "sasu", "sa", "sci", "france", "holding", "groupe", "participations", "developpement",
         "club", "comite", "musique", "ecole", "amicale", "maison", "centre", "culture", "lycee", "pharmacie",
         "llc", "inc", "co", "corp", "group", "center", "partners", "limited", "private", "pvt", "ltd", "services"]
rows = []
for w in words:
    if w in pos:
        rows.append({"word": w, **{f"{g}_add": round(float(rates['add'][k, pos[w]]), 3) for k, g in enumerate(groups)},
                     **{f"{g}_drop": round(float(rates['drop'][k, pos[w]]), 3) for k, g in enumerate(groups)}})
print(pd.DataFrame(rows).to_string(index=False))
