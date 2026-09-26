"""Multilingual E5 cosine features at full scale (p1): p4's p4_e5, re-engineered for ~12M records / ~75M pairs.

# model: intfloat/multilingual-e5-small, licence: MIT, params: 117,653,760 (p4 T06 report), offline after download
Same encoder, texts and cosines as p4_e5 (p4 T06: "query: " + cleaned raw text, mean pooling, L2-normalised, fp16,
max 64 tokens; fields name / addr / full). Difference: embeddings are stored as ONE memory-mapped matrix per
(dataset, field), aligned to the dataset's record rows ($BER_ARTIFACTS/emb_mm/<model>/<dataset fp>_<field>.npy),
and cosines are computed in chunks from that file — so RAM stays small instead of holding every vector (p4's shard
cache loads all shards into memory: ~18 GB per field at our scale). Each unique text is embedded once per dataset.
Features: f_p1_e5_<field>_cos, and per field the gap to the best cosine and the rank within the S1 entity and
within the S2/S3 record. Pretrained weights only; nothing fitted on the target → not transductive.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ber.data import Dataset
from ber.features.p4_e5 import _FIELDS, _clean, _Encoder
from ber.interfaces import FeatureSet
from ber.registry import register

log = logging.getLogger(__name__)


def _field_texts(recs: pd.DataFrame, field: str) -> np.ndarray:
    parts = [recs[c].map(_clean) for c in _FIELDS[field]]
    if len(parts) == 1:
        return parts[0].to_numpy(dtype=object)
    return np.array([", ".join(p for p in ps if p) for ps in zip(*parts)], dtype=object)


def _rank_gap(gid: np.ndarray, v: np.ndarray):
    """Rank (1 = best) and gap to the best value of v within groups gid; NaN-safe (NaN ranks last)."""
    vv = np.where(np.isnan(v), -np.inf, v)
    order = np.lexsort((-vv, gid))
    g, s = gid[order], vv[order]
    first = np.r_[True, g[1:] != g[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(g)), 0))
    rank = np.empty(len(v), np.float32); rank[order] = np.arange(len(g)) - start + 1
    with np.errstate(invalid="ignore"):                       # -inf - -inf for all-NaN groups -> NaN below
        gap = np.empty(len(v), np.float32); gap[order] = s[start] - s
    gap[~np.isfinite(gap)] = np.nan
    return rank, gap


@register("features", "p1_e5")
class E5MemmapCosine(FeatureSet):
    """E5 cosine features backed by per-dataset memory-mapped embedding matrices; see module docstring."""

    def __init__(self, model: str = "intfloat/multilingual-e5-small", fields: list[str] | None = None,
                 max_length: int = 64, batch_size: int = 512, device: str | None = None,
                 chunk: int = 2_000_000, embed_chunk: int = 1_000_000):
        self.model_name, self.fields = model, fields or ["name", "addr", "full"]
        self.max_length, self.batch_size, self.device = max_length, batch_size, device
        self.chunk, self.embed_chunk = int(chunk), int(embed_chunk)
        self._enc = None

    def _root(self) -> Path:
        art = os.environ.get("BER_ARTIFACTS") or str(Path(__file__).resolve().parents[5] / "artifacts")
        return Path(art) / "emb_mm" / f"{self.model_name.replace('/', '__')}_ml{self.max_length}"

    def matrix(self, ds: Dataset, field: str) -> np.ndarray:
        """(n_records, dim) fp16 memmap of `field` embeddings aligned to ds.records rows (computed once)."""
        path = self._root() / f"{ds.fingerprint()}_{field}.npy"
        if path.exists():
            return np.load(path, mmap_mode="r")
        if self._enc is None:
            self._enc = _Encoder(self.model_name, self.max_length, self.batch_size, self.device)
        texts = _field_texts(ds.records, field)
        codes, uniq = pd.factorize(texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        U = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(uniq), self._enc.dim))
        t0 = time.time()
        for s in range(0, len(uniq), self.embed_chunk):
            batch = [t if t else " " for t in uniq[s:s + self.embed_chunk]]
            U[s:s + len(batch)] = self._enc.encode(batch)
            log.info("p1_e5 %s: %d/%d unique texts embedded (%.0f/s)", field, s + len(batch), len(uniq),
                     (s + len(batch)) / max(time.time() - t0, 1e-9))
        M = np.lib.format.open_memmap(path.with_suffix(".rows.npy"), mode="w+", dtype=np.float16,
                                      shape=(len(texts), self._enc.dim))
        for s in range(0, len(texts), self.embed_chunk):
            M[s:s + self.embed_chunk] = U[codes[s:s + self.embed_chunk]]
            empty = np.flatnonzero(np.asarray([not t for t in texts[s:s + self.embed_chunk]]))
            M[s + empty] = 0                                    # empty text → zero vector (cosine NaN later)
        M.flush(); del M, U
        os.replace(path.with_suffix(".rows.npy"), path)
        tmp.unlink()
        return np.load(path, mmap_mode="r")

    def transform(self, ds: Dataset, pairs: pd.DataFrame) -> pd.DataFrame:
        index = pd.Index(ds.records["entity_id"].to_numpy())
        li = index.get_indexer(pairs["s1_id"])
        ri = index.get_indexer(pairs["cand_id"])
        out = {}
        s1u = np.unique(li)                       # S1 rows repeat ~30x: read each once
        order = np.argsort(ri, kind="stable")     # right rows read in sorted order (sequential memmap access)
        for field in self.fields:
            M = self.matrix(ds, field)
            E1 = np.asarray(M[s1u])               # (n_S1, dim) fp16, ~1.7 GB on full train
            cos = np.empty(len(pairs), np.float32)
            for s in range(0, len(pairs), self.chunk):
                o = order[s:s + self.chunk]
                ru, inv = np.unique(ri[o], return_inverse=True)
                b = np.asarray(M[ru], dtype=np.float32)[inv]
                a = E1[np.searchsorted(s1u, li[o])].astype(np.float32)
                c = np.einsum("ij,ij->i", a, b)
                zero = (np.abs(a).sum(1) == 0) | (np.abs(b).sum(1) == 0)
                c[zero] = np.nan
                cos[o] = c
            del E1
            out[f"f_p1_e5_{field}_cos"] = cos
            for side, gid in (("s1", li), ("rec", ri)):
                rank, gap = _rank_gap(gid, cos)
                out[f"f_p1_e5_{field}_{side}_rank"] = rank
                out[f"f_p1_e5_{field}_{side}_gap"] = gap
        return pd.DataFrame({k: v.astype(np.float16) for k, v in out.items()}, index=pairs.index)
