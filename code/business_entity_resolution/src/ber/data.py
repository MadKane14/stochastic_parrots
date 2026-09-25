"""Data loading and the Dataset container.

Files (official layout, BER_DATA_DIR points at the `dataset/` folder):
    <data_dir>/train/train_source{1,2,3}.tsv, <data_dir>/train/train_ground_truth.tsv
    <data_dir>/test/test_source{1,2,3}.tsv
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


@dataclass
class Dataset:
    """All records of one evaluation unit (a split side, or the test set).

    records: DataFrame with entity_id, source (1/2/3), business_name, business_address,
             country (+ normalized columns after the normalize stage).
    gt:      {s1_id: frozenset(matched S2/S3 ids)} for every S1 id, or None (test).
             Only contains ids present in `records`.
    """

    name: str
    records: pd.DataFrame
    gt: dict[str, frozenset] | None = None
    meta: dict = field(default_factory=dict)

    @property
    def s1(self) -> pd.DataFrame:
        return self.records[self.records["source"] == 1]

    @property
    def right(self) -> pd.DataFrame:
        return self.records[self.records["source"] != 1]

    @property
    def s1_ids(self) -> list[str]:
        return self.s1["entity_id"].tolist()

    def with_records(self, records: pd.DataFrame) -> "Dataset":
        return replace(self, records=records)

    def lookup(self, ids, columns) -> pd.DataFrame:
        """Rows of `records` for `ids` (in order), restricted to `columns`."""
        idx = self.records.set_index("entity_id")
        return idx.loc[list(ids), list(columns)].reset_index(drop=True)

    def labels(self, pairs: pd.DataFrame) -> np.ndarray:
        if self.gt is None:
            raise ValueError(f"dataset {self.name} has no ground truth")
        gt = self.gt
        return np.fromiter(
            (c in gt.get(s, ()) for s, c in zip(pairs["s1_id"], pairs["cand_id"])),
            dtype=np.int8, count=len(pairs),
        )

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for eid in sorted(self.records["entity_id"]):
            h.update(eid.encode())
            h.update(b"\n")
        return h.hexdigest()[:16]


def _read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    # Parsing sanity check: quotes inside fields can silently merge lines.
    with open(path, "rb") as f:
        n_lines = sum(1 for line in f if line.strip())
    if n_lines - 1 != len(df):
        warnings.warn(f"{path}: {n_lines - 1} non-empty data lines but parsed {len(df)} rows "
                      "(quoting issue?) — report this to the lead.")
    return df


def load_part(data_dir: str | Path, part: str, cache_dir: str | Path | None = None) -> Dataset:
    """Load `train` or `test` into one Dataset (all three sources concatenated).

    The raw TSVs are ~2.4 GB; with `cache_dir` the parsed tables are cached as parquet
    (keyed by file names/sizes/mtimes), which makes repeated runs load in seconds.
    """
    d = Path(data_dir) / part
    if cache_dir is not None:
        files = sorted(d.glob("*.tsv"))
        sig = hashlib.sha256("|".join(f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}" for p in files).encode()).hexdigest()[:16]
        rec_p, gt_p = Path(cache_dir) / f"raw_{part}_{sig}.parquet", Path(cache_dir) / f"raw_{part}_{sig}_gt.parquet"
        if rec_p.exists():
            records = pd.read_parquet(rec_p)
            gt = None
            if gt_p.exists():
                g = pd.read_parquet(gt_p)
                gt = {s: frozenset(x.split(",")) if x else frozenset() for s, x in zip(g["s1"], g["m"])}
            return Dataset(name=part, records=records, gt=gt, meta={"data_dir": str(data_dir)})
        ds = load_part(data_dir, part)
        rec_p.parent.mkdir(parents=True, exist_ok=True)
        if ds.gt is not None:  # gt first: records file existing == cache complete
            pd.DataFrame({"s1": list(ds.gt), "m": [",".join(sorted(v)) for v in ds.gt.values()]}).to_parquet(gt_p)
        tmp = rec_p.with_suffix(".tmp")
        ds.records.to_parquet(tmp)
        tmp.replace(rec_p)
        return ds
    frames = []
    for src in (1, 2, 3):
        df = _read_tsv(d / f"{part}_source{src}.tsv")
        missing = set(RAW_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{part}_source{src}.tsv missing columns {missing}")
        df = df[RAW_COLUMNS].copy()
        df.insert(1, "source", src)
        frames.append(df)
    records = pd.concat(frames, ignore_index=True)
    for col in ("business_name", "business_address", "country"):
        records[col] = records[col].fillna("").astype(str)
    if records["entity_id"].duplicated().any():
        raise ValueError(f"duplicate entity_ids in {part}")

    gt = None
    gt_path = d / f"{part}_ground_truth.tsv"
    if gt_path.exists():
        g = _read_tsv(gt_path)
        gt = {}
        for s1, lst in zip(g["source1_entity_id"], g["matched_entity_ids"]):
            ids = [x.strip() for x in str(lst).split(",") if x.strip()]
            gt[s1] = frozenset(ids)
        for s1 in records.loc[records["source"] == 1, "entity_id"]:
            gt.setdefault(s1, frozenset())
    return Dataset(name=part, records=records, gt=gt, meta={"data_dir": str(data_dir)})


def unit_hash(key: str, seed: int) -> float:
    """Deterministic uniform [0,1) hash — identical on every machine/OS."""
    h = hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:12], 16) / float(16**12)
