"""T18 (p3): precompute p1_e5 embedding matrices for ONE country-string group of train — exactly the dataset that
exploration/p2/dense_cv_group.py builds (p2's group_loader) — so the dense-CV runs on a CPU machine only read them.

p1_e5 keys its memory-mapped matrices by the dataset fingerprint (sorted entity ids) and field, so the normalized
group dataset inside dense_cv_group finds these files. The p1_e5 spec (model, max_length, ...) is taken from the
config, so the cache directory matches what the config's feature set looks up.

    CUDA_VISIBLE_DEVICES=0 python exploration/p3/e5_group.py -c code/business_entity_resolution/configs/exp/p1/T13_sub4.yaml \
        --group India --field name
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code/business_entity_resolution/src"))
_spec = importlib.util.spec_from_file_location("p2_dense_cv_group", REPO / "exploration/p2/dense_cv_group.py")
dcg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dcg)

from ber.config import env, load_config  # noqa: E402
from ber.registry import build  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="config whose p1_e5 spec is used")
    ap.add_argument("--group", required=True)
    ap.add_argument("--field", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    cfg = load_config(a.config)
    spec = next(f for f in cfg["features"] if f["name"] == "p1_e5")
    fs = build("features", spec)
    ds = dcg.group_loader(a.group)(env("BER_DATA_DIR"), "train", cache_dir=Path(cfg["paths"]["artifacts_dir"]) / "cache")
    t = time.time()
    m = fs.matrix(ds, a.field)
    print(f"{ds.name} {a.field}: {m.shape} {m.dtype} in {time.time() - t:.0f}s -> "
          f"{fs._root() / (ds.fingerprint() + '_' + a.field + '.npy')}", flush=True)
