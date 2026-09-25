"""T06: E5 embedding throughput on a sample of real record texts -> projected time for all train + test records.

Usage (repo root, GPU box):  python exploration/p4/e5_throughput.py --n 200000 --out exploration/p4/e5_throughput_out.md
Samples records uniformly from every train/test source file (all countries), embeds the name / address /
name+address texts with the same encoder code as the p4_e5 FeatureSet, and reports texts/s per field.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "business_entity_resolution" / "src"))
from ber.features.p4_e5 import _FIELDS, _clean, _Encoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000, help="records sampled in total")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dd = Path(os.environ.get("BER_DATA_DIR", REPO / "data" / "dataset"))
    files = sorted(dd.glob("*/*_source*.tsv"))
    total = 0
    frames = []
    rng = np.random.default_rng(0)
    for p in files:
        df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                         usecols=["business_name", "business_address", "country"])
        total += len(df)
        frames.append(df)
    per = a.n / total
    sample = pd.concat([f.iloc[rng.random(len(f)) < per] for f in frames], ignore_index=True)
    enc = _Encoder(a.model, a.max_length, a.batch_size, None)
    import torch
    lines = [f"## e5_throughput — {dt.datetime.now():%Y-%m-%d %H:%M}",
             f"model `{a.model}`, params {enc.n_params:,}, device {enc.device} "
             f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}), torch {torch.__version__}, "
             f"batch {a.batch_size}, max_length {a.max_length}",
             f"records in all train+test source files: {total:,}; sample: {len(sample):,}", "",
             "| field | texts | seconds | texts/s | projected hours for all records |", "|---|---|---|---|---|"]
    enc.encode(["warm up"] * 64)
    for field, cols in _FIELDS.items():
        texts = [", ".join(p for p in ps if p) for ps in zip(*[sample[c].map(_clean) for c in cols])]
        texts = [t for t in texts if t]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        enc.encode(texts)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        s = time.time() - t0
        rate = len(texts) / s
        lines.append(f"| {field} | {len(texts):,} | {s:.1f} | {rate:,.0f} | {total / rate / 3600:.2f} |")
    if torch.cuda.is_available():
        lines.append(f"\npeak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    lines.append("(projection = all records, no dedup; unique texts are fewer, so this is an upper bound)\n")
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
