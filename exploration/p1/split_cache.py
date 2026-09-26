"""One-off (p1): split an old combined dense-CV feature cache (pairs_<key>.parquet) into the per-feature-set layout
of dense_cv.py (base_<key> + feat_<name>_<key>), so already-computed features are reused.
    python exploration/p1/split_cache.py <config> <old pairs_*.parquet name>
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code/business_entity_resolution/src"))
from ber.config import env, load_config  # noqa: E402
from ber.data import load_part  # noqa: E402
from ber.pipeline import Pipeline, _component_fp  # noqa: E402

PREFIX = {"p1_match": "f_p1_", "p1_extra": "f_p1x_", "p1_e5": "f_p1_e5_"}

cfg = load_config(sys.argv[1])
art = Path(env("BER_ARTIFACTS"))
full = load_part(env("BER_DATA_DIR"), "train", cache_dir=art / "cache")
pipe = Pipeline(cfg, art)
tfp = full.fingerprint() + full.name
base_key = hashlib.sha256("|".join([tfp, *[_component_fp(b) for b in pipe.blockers],
                                    _component_fp(pipe.normalizer)]).encode()).hexdigest()[:16]
old = pd.read_parquet(art / "dense_cv" / sys.argv[2])
base = [c for c in old.columns if c in ("s1_id", "ri", "label") or c.startswith("blk_")]
old[base].to_parquet(art / "dense_cv" / f"base_{base_key}.parquet")
for fs in pipe.featuresets:
    cols = [c for c in old.columns if c.startswith(PREFIX[fs.registry_name])
            and not (fs.registry_name == "p1_match" and (c.startswith("f_p1x_") or c.startswith("f_p1_e5_")))]
    if not cols:
        continue
    fkey = hashlib.sha256(f"{base_key}|{_component_fp(fs)}".encode()).hexdigest()[:16]
    old[cols].to_parquet(art / "dense_cv" / f"feat_{fs.registry_name}_{fkey}.parquet")
    print(fs.registry_name, len(cols), "columns ->", f"feat_{fs.registry_name}_{fkey}.parquet")
