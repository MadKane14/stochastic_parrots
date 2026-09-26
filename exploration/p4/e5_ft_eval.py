"""T12 (p4): intrinsic check of fine-tuned vs pretrained E5 on REAL blocker candidates of a held-out target side.

Usage (repo root):  python exploration/p4/e5_ft_eval.py --pairs <run_dir>/scored_pairs.parquet --label val_v1@0.1 \
                        --models intfloat/multilingual-e5-small /path/ft_val_v1 --out exploration/p4/e5_ft_eval_out.md
`scored_pairs.parquet` = the harness's target-side candidates with labels (s1_id, cand_id, label). For each model and
field (name, full): ROC-AUC / average precision of the cosine for true vs false candidate pairs, the share of S1
entities (with >= 1 true candidate) whose top-cosine candidate is a true match, and the same AUC restricted to pairs
whose S2/S3 name is non-Latin. Proxy only; the decision metric is dense CV / LOCO (D9).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "business_entity_resolution" / "src"))
from ber.data import load_part  # noqa: E402
from ber.features.p4_e5 import _Encoder, field_texts  # noqa: E402


def _nonlatin(s: str) -> bool:
    return any(ch.isalpha() and not ch.isascii() and not unicodedata.name(ch, "").startswith("LATIN") for ch in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--fields", nargs="+", default=["name", "full"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    pairs = pd.read_parquet(a.pairs, columns=["s1_id", "cand_id", "label"])
    full = load_part(os.environ["BER_DATA_DIR"], "train", cache_dir=Path(os.environ["BER_ARTIFACTS"]) / "cache")
    ids = pd.unique(np.concatenate([pairs["s1_id"].to_numpy(), pairs["cand_id"].to_numpy()]))
    recs = full.lookup(ids, ["business_name", "business_address"])
    del full
    index = pd.Index(ids)
    li, ri = index.get_indexer(pairs["s1_id"]), index.get_indexer(pairs["cand_id"])
    y = pairs["label"].to_numpy()
    nonlat = np.array([_nonlatin(s) for s in recs["business_name"].to_numpy()])[ri]
    s1 = pairs["s1_id"].to_numpy()
    has_pos = pd.Series(y).groupby(s1).transform("max").to_numpy() > 0
    res = []
    for m in a.models:
        enc = _Encoder(m, 64, 512, None)
        for field in a.fields:
            texts = field_texts(recs, field)
            codes, uniq = pd.factorize(pd.Series(texts, dtype=object))
            U = enc.encode([t if t else " " for t in uniq]).astype(np.float32)
            E = U[codes]
            empty = np.asarray([not t for t in texts])
            cos = np.einsum("ij,ij->i", E[li], E[ri])
            ok = ~(empty[li] | empty[ri])
            c = np.where(ok, cos, -1.0)
            df = pd.DataFrame({"s1": s1, "c": c, "y": y})[has_pos]
            top = df.sort_values(["s1", "c"], ascending=[True, False]).groupby("s1").head(1)
            r = {"model": m, "field": field, "n_pairs": int(ok.sum()), "pos": int(y[ok].sum()),
                 "auc": round(roc_auc_score(y[ok], cos[ok]), 5), "ap": round(average_precision_score(y[ok], cos[ok]), 5),
                 "top1_true": round(float(top["y"].mean()), 5)}
            nl = ok & nonlat
            if y[nl].min() != y[nl].max():
                r.update({"auc_nonlatin": round(roc_auc_score(y[nl], cos[nl]), 5), "n_nonlatin": int(nl.sum())})
            print(json.dumps(r), flush=True)
            res.append(r)
        del enc
    lines = [f"\n## {a.label} — {dt.datetime.now():%Y-%m-%d %H:%M} — pairs `{a.pairs}`",
             "| model | field | pairs (pos) | ROC-AUC | AP | top-1 true | AUC non-Latin name (n) |", "|---|---|---|---|---|---|---|"]
    for r in res:
        lines.append(f"| `{r['model']}` | {r['field']} | {r['n_pairs']:,} ({r['pos']:,}) | {r['auc']} | {r['ap']} | "
                     f"{r['top1_true']} | {r.get('auc_nonlatin', '–')} ({r.get('n_nonlatin', 0):,}) |")
    with open(a.out, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
