"""T32 (p4): reproducibility of the packaged src/e5_finetune.py.

  python exploration/p4/ft_repro.py compare A_dir B_dir --out out.json   # sha256, max |Δweight|, max |Δcos| on 1,000 test pairs
  python exploration/p4/ft_repro.py det -- <e5_finetune.py args>         # same run with deterministic torch settings
The 1,000 pairs: TEST S1 record text vs a random TEST S2/S3 record text ("name, address", as the features embed them),
fixed seed; cosines use the same encoder code as p4_e5/p1_e5 (`_Encoder`).
"""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "code" / "business_entity_resolution"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compare(a: str, b: str, out: str):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    sys.path.insert(0, str(PKG / "src"))
    from ber.data import load_part
    from ber.features.p4_e5 import _Encoder, field_texts
    fa, fb = Path(a) / "model.safetensors", Path(b) / "model.safetensors"
    res = {"sha256_a": sha256(fa), "sha256_b": sha256(fb)}
    res["identical_file"] = res["sha256_a"] == res["sha256_b"]
    wa, wb = load_file(str(fa)), load_file(str(fb))
    res["max_abs_weight_diff"] = max(float((wa[k].float() - wb[k].float()).abs().max()) for k in wa)
    test = load_part(os.environ["BER_DATA_DIR"], "test", cache_dir=Path(os.environ["BER_ARTIFACTS"]) / "cache").records
    rng = np.random.default_rng(0)
    s1 = np.flatnonzero((test["source"] == 1).to_numpy()); rt = np.flatnonzero((test["source"] != 1).to_numpy())
    i, j = rng.choice(s1, 1000, replace=False), rng.choice(rt, 1000, replace=False)
    txt = field_texts(test.iloc[np.r_[i, j]].reset_index(drop=True), "full")
    txt = [t if t else " " for t in txt]
    cos = {}
    for k, d in (("a", a), ("b", b)):
        E = _Encoder(d, 64, 256, None).encode(txt).astype(np.float32)
        cos[k] = np.einsum("ij,ij->i", E[:1000], E[1000:])
        del E; torch.cuda.empty_cache()
    res["max_abs_cos_diff_1000_test_pairs"] = float(np.abs(cos["a"] - cos["b"]).max())
    res["mean_abs_cos_diff_1000_test_pairs"] = float(np.abs(cos["a"] - cos["b"]).mean())
    for k, d in (("a", a), ("b", b)):
        st = json.loads((Path(d) / "p4_finetune_stats.json").read_text())
        res[f"train_seconds_{k}"] = st["train_seconds"]; res[f"steps_{k}"] = st["steps"]
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def det(args: list[str]):
    """Run src/e5_finetune.py with deterministic torch settings (the proposed fix), without editing the package."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)   # warn (not raise) on ops without a deterministic kernel
    torch.backends.cudnn.benchmark = False
    sys.argv = [str(PKG / "src" / "e5_finetune.py"), *args]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3], sys.argv[sys.argv.index("--out") + 1])
    elif sys.argv[1] == "det":
        det(sys.argv[sys.argv.index("--") + 1:])
