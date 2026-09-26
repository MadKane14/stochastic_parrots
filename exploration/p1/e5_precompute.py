"""Precompute p1_e5 embedding matrices for the full train and test datasets (GPU), so later dense-CV / test runs
only read the memory-mapped files. Same code path as the feature set (p1_e5.matrix).
    CUDA_VISIBLE_DEVICES=0 python exploration/p1/e5_precompute.py [--fields name full addr]
"""
import argparse
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code/business_entity_resolution/src"))
from ber.config import env  # noqa: E402
from ber.data import load_part  # noqa: E402
from ber.features.p1_e5 import E5MemmapCosine  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", nargs="+", default=["name", "full", "addr"])
    ap.add_argument("--parts", nargs="+", default=["train", "test"])
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    fs = E5MemmapCosine()
    for part in a.parts:
        ds = load_part(env("BER_DATA_DIR"), part, cache_dir=Path(env("BER_ARTIFACTS")) / "cache")
        for field in a.fields:
            t = time.time()
            m = fs.matrix(ds, field)
            print(f"{part} {field}: {m.shape} in {time.time() - t:.0f}s", flush=True)
