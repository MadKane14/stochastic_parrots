"""Rebuild the pretrained multilingual-e5-small memmaps (train + test, fields name/full) used by p1_e5 (p1, Sun 27 Sep).
Same code path as the pipeline (E5MemmapCosine.matrix), so file names/fingerprints match what a run looks up.
Model: intfloat/multilingual-e5-small, licence MIT, params 117,653,760 (offline, HF cache)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/business_entity_resolution/src"))
from ber.config import env
from ber.data import load_part
from ber.features.p1_e5 import E5MemmapCosine

fs = E5MemmapCosine(model="intfloat/multilingual-e5-small", fields=["name", "full"])
for part in ("test", "train"):
    ds = load_part(env("BER_DATA_DIR"), part, cache_dir=Path(env("BER_ARTIFACTS")) / "cache")
    for f in ("name", "full"):
        m = fs.matrix(ds, f)
        print(part, f, ds.fingerprint(), m.shape, flush=True)
