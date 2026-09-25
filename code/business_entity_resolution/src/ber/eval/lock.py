"""FROZEN. Lead-only: write planning/eval_lock.json (eval code hash + split fingerprints).

Run once when the real data lands (and again only if the lead deliberately changes the eval):
    python src/run.py lock
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from ber.data import load_part
from ber.eval.harness import DEV_SEED, LOCK_PATH, eval_code_hash
from ber.eval.splits import make_split, split_fingerprint, subsample

DEV_FRACS = (0.1,)   # locked dev subsamples, used as e.g. --split val_v1@0.1


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_lock(data_dir: str | Path, cache_dir: str | Path | None = None) -> dict:
    data_dir = Path(data_dir)
    full = load_part(data_dir, "train", cache_dir=cache_dir)
    split_names = ["val_v1"] + [f"xc_{c}" for c in sorted(full.s1["country"].unique())]
    splits = {}
    for name in split_names:
        tr0, va0 = make_split(full, name)
        for frac in (1.0, *DEV_FRACS):
            tr, va = (tr0, va0) if frac == 1.0 else (subsample(tr0, frac, seed=DEV_SEED), subsample(va0, frac, seed=DEV_SEED))
            key = name if frac == 1.0 else f"{name}@{frac}"
            splits[key] = {"fingerprint": split_fingerprint(tr, va), "n_s1_train": len(tr.s1), "n_s1_val": len(va.s1),
                           "n_right_train": len(tr.right), "n_right_val": len(va.right),
                           "val_singleton_rate": round(sum(1 for v in va.gt.values() if not v) / max(1, len(va.gt)), 4),
                           "conflicting_right_records": tr0.meta["conflicting_right_records"]}
    lock = {
        "eval_code_sha256": eval_code_hash(),
        "locked_at": dt.datetime.now().isoformat(timespec="seconds"),
        "data_files_sha256": {p.name: _file_sha(p) for p in sorted((data_dir / "train").glob("*.tsv"))},
        "splits": splits,
    }
    LOCK_PATH.write_text(json.dumps(lock, indent=1), encoding="utf-8")
    return lock
