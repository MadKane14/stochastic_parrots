"""T12: embed every unique text of the train/test records once into the p4_e5 cache (same keys, same format).

Usage (repo root, GPU box):
    python exploration/p4/e5_embed_all.py --parts train,test --fields name,full --out-json exploration/p4/e5_embed_all.json
    # one field per GPU in parallel: CUDA_VISIBLE_DEVICES=0 ... --fields name &  CUDA_VISIBLE_DEVICES=1 ... --fields full
Writes shards to $BER_ARTIFACTS/emb/<model>/<field>_ml<max_length>/ (or --cache-dir), skipping texts already cached,
so p4_e5 / dense_cv later find everything in the cache. Reports unique-text counts, time and texts/s per part x field.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "business_entity_resolution" / "src"))
from ber.data import load_part  # noqa: E402
from ber.features.p4_e5 import E5Cosine, _Cache, _Encoder, _hash64, field_texts, write_shard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="train,test")
    ap.add_argument("--fields", default="name,full")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=1_000_000, help="texts per shard")
    ap.add_argument("--cache-dir", default=None, help="default: $BER_ARTIFACTS/emb")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    if a.cache_dir:
        os.environ["BER_ARTIFACTS"] = str(Path(a.cache_dir).parent)
    art = Path(os.environ.get("BER_ARTIFACTS", REPO / "artifacts"))
    fs = E5Cosine(model=a.model, max_length=a.max_length)
    enc = None
    report = {"model": a.model, "max_length": a.max_length, "batch_size": a.batch_size, "runs": []}
    for part in a.parts.split(","):
        t = time.time()
        recs = load_part(os.environ.get("BER_DATA_DIR", REPO / "data" / "dataset"), part, cache_dir=art / "cache").records
        print(f"{part}: {len(recs):,} records loaded in {time.time() - t:.0f}s", flush=True)
        for field in a.fields.split(","):
            texts = field_texts(recs, field)
            uniq = pd.unique(texts[texts != ""])
            keys = _hash64(uniq)
            root = fs._cache_root(field)
            todo = np.flatnonzero(_Cache([root]).find(keys) < 0)
            print(f"{part}/{field}: {len(texts):,} texts, {len(uniq):,} unique non-empty, {len(todo):,} to embed", flush=True)
            if enc is None and len(todo):
                enc = _Encoder(a.model, a.max_length, a.batch_size, None)
            t0 = time.time()
            for i in range(0, len(todo), a.chunk):
                sel = todo[i:i + a.chunk]
                write_shard(root, keys[sel], enc.encode(list(uniq[sel])))
                done = i + len(sel)
                print(f"  {done:,}/{len(todo):,}  {done / (time.time() - t0):,.0f}/s", flush=True)
            s = time.time() - t0
            run = {"part": part, "field": field, "records": int(len(texts)), "unique_nonempty": int(len(uniq)),
                   "embedded": int(len(todo)), "seconds": round(s, 1),
                   "texts_per_s": round(len(todo) / s, 1) if len(todo) and s > 0 else None,
                   "gpu": (__import__("torch").cuda.get_device_name(0) if enc is not None and enc.device.startswith("cuda") else None),
                   "cache_bytes": int(sum(p.stat().st_size for p in root.glob("shard_*.npz")))}
            report["runs"].append(run)
            print(json.dumps(run), flush=True)
        del recs
    if enc is not None:
        report["params"] = enc.n_params
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
