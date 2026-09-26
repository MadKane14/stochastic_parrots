"""p4 T06/T12: multilingual E5 cosine similarities (name, address, name+address) as GBDT features.

# model: intfloat/multilingual-e5-small, licence: MIT, params: 117,653,760 (counted at load)
Pretrained weights only (no fine-tuning), run locally/offline after the first download (set HF_HOME).
Each unique text is embedded once ("query: " + text, mean pooling, L2-normalised, stored fp16) and cached
under $BER_ARTIFACTS/emb/<model>/<field>_ml<max_length>/shard_*.npz, keyed by a 64-bit hash of the text.
Extra read-only caches (e.g. a precomputed full-train embedding) can be listed in BER_EMB_READONLY
(os.pathsep-separated folders that contain <model>/<field>_ml<max_length>/).
Memory: one field's vectors are held at a time; cosines are computed chunk-wise straight from the cache rows,
so ~77M dense-CV pairs need about (unique texts x 768 B) + a few GB.
Nothing is fitted on the target records -> not transductive.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ber.data import Dataset
from ber.interfaces import FeatureSet
from ber.registry import register

# field name -> raw columns joined with ", "
_FIELDS = {"name": ("business_name",), "addr": ("business_address",), "full": ("business_name", "business_address")}


def _clean(s: str) -> str:
    return " ".join(str(s).split())


def field_texts(recs: pd.DataFrame, field: str) -> np.ndarray:
    """Text of `field` for every row of `recs` ('' when empty) — shared by the FeatureSet and the precompute script."""
    parts = [recs[c].map(_clean).to_numpy(dtype=object) for c in _FIELDS[field]]
    if len(parts) == 1:
        return parts[0]
    return np.array([", ".join(p for p in ps if p) for ps in zip(*parts)], dtype=object)


def _hash64(texts) -> np.ndarray:
    """Stable 64-bit text keys (identical on every machine)."""
    return np.fromiter((int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "little", signed=True)
                        for t in texts), dtype=np.int64, count=len(texts))


class _Encoder:
    """Mean-pooled, L2-normalised sentence embeddings with a HF encoder (fp16 on GPU)."""

    def __init__(self, model_name: str, max_length: int, batch_size: int, device: str | None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.dim = self.model.config.hidden_size
        self.max_length, self.batch_size = max_length, batch_size
        print(f"[p4_e5] {model_name} on {self.device} ({dtype}), params={self.n_params:,}, dim={self.dim}", flush=True)

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        out = np.empty((len(texts), self.dim), dtype=np.float16)
        order = np.argsort([-len(t) for t in texts], kind="stable")   # length-sorted batches: little padding
        with torch.inference_mode():
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                b = self.tok(["query: " + texts[j] for j in idx], padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt").to(self.device)
                h = self.model(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1).float() / m.sum(1).float().clamp(min=1.0)
                out[idx] = torch.nn.functional.normalize(v, dim=-1).half().cpu().numpy()
        return out


def write_shard(root: Path, keys: np.ndarray, vecs: np.ndarray) -> Path:
    """Append one shard (atomic rename) to a cache folder."""
    root.mkdir(parents=True, exist_ok=True)
    n = len(list(root.glob("shard_*.npz")))
    tmp = root / f"shard_{n:05d}.tmp.npz"
    np.savez(tmp, keys=keys, vecs=vecs)
    out = root / f"shard_{n:05d}.npz"
    tmp.replace(out)
    return out


class _Cache:
    """Read-only view of all shards in `roots`: keys (int64 text hashes) -> row of one fp16 matrix.
    Keys are sorted through an index array, so the vectors are never copied a second time."""

    def __init__(self, roots: list[Path]):
        shards = [p for r in roots for p in sorted(r.glob("shard_*.npz"))]
        sizes, dim = [], 0
        for p in shards:
            with np.load(p) as z:
                sizes.append(len(z["keys"]))
                dim = z["vecs"].shape[1]
        n = int(sum(sizes))
        self.keys = np.empty(n, np.int64)
        self.vecs = np.empty((n, dim), np.float16) if n else None
        o = 0
        for p, s in zip(shards, sizes):
            with np.load(p) as z:
                self.keys[o:o + s] = z["keys"]
                self.vecs[o:o + s] = z["vecs"]
            o += s
        self.order = np.argsort(self.keys, kind="stable")
        self.sorted_keys = self.keys[self.order]

    def find(self, keys: np.ndarray) -> np.ndarray:
        """Row in `vecs` for each key, -1 if missing."""
        if len(self.sorted_keys) == 0:
            return np.full(len(keys), -1, np.int64)
        pos = np.searchsorted(self.sorted_keys, keys).clip(0, len(self.sorted_keys) - 1)
        return np.where(self.sorted_keys[pos] == keys, self.order[pos], -1)


def _group_feats(v: np.ndarray, gid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gap to the best value and rank ('min' method) of v within each group; NaN where v is NaN."""
    codes = pd.factorize(gid)[0]
    w = np.where(np.isnan(v), -np.inf, v)
    order = np.lexsort((-w, codes))
    gc, vs = codes[order], w[order]
    first = np.r_[True, gc[1:] != gc[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(gc)), 0))
    new_val = np.r_[True, (gc[1:] != gc[:-1]) | (vs[1:] != vs[:-1])]
    run_start = np.maximum.accumulate(np.where(new_val, np.arange(len(gc)), 0))
    rank = np.empty(len(v), np.float32)
    rank[order] = run_start - start + 1
    gap = np.empty(len(v), np.float32)
    with np.errstate(invalid="ignore"):
        gap[order] = vs[start] - vs
    nan = np.isnan(v)
    rank[nan], gap[nan] = np.nan, np.nan
    gap[~np.isfinite(gap)] = np.nan
    return gap, rank


@register("features", "p4_e5")
class E5Cosine(FeatureSet):
    """Cosine of E5 embeddings per candidate pair, for each field in `fields` (name / addr / full).
    Empty text on either side -> NaN. Optional `group_feats`: gap to the best cosine and rank of the cosine
    within the S1 entity (`_gap`, `_rank`) and, with `group_sides: [s1, rec]`, also within the S2/S3 record."""

    def __init__(self, model: str = "intfloat/multilingual-e5-small", fields: list[str] | None = None,
                 max_length: int = 64, batch_size: int = 512, device: str | None = None,
                 group_feats: bool = False, group_sides: list[str] | None = None, cache: bool = True):
        self.model_name, self.fields = model, fields or ["name", "addr", "full"]
        self.max_length, self.batch_size, self.device = max_length, batch_size, device
        self.group_feats, self.group_sides = group_feats, group_sides or ["s1"]
        self.use_cache = cache
        self._enc = None
        self.stats = {"n_embedded": 0, "embed_s": 0.0}

    def _encoder(self) -> _Encoder:
        if self._enc is None:
            self._enc = _Encoder(self.model_name, self.max_length, self.batch_size, self.device)
        return self._enc

    def _sub(self, field: str) -> str:
        return f"{self.model_name.replace('/', '__')}/{field}_ml{self.max_length}"

    def _cache_root(self, field: str) -> Path:
        art = os.environ.get("BER_ARTIFACTS") or str(Path(__file__).resolve().parents[5] / "artifacts")
        return Path(art) / "emb" / self._sub(field)

    def _readonly_roots(self, field: str) -> list[Path]:
        extra = [p for p in os.environ.get("BER_EMB_READONLY", "").split(os.pathsep) if p]
        return [Path(p) / self._sub(field) for p in extra]

    def _rows(self, texts: np.ndarray, field: str) -> tuple[np.ndarray, np.ndarray]:
        """Cache row for each text (-1 where the text is empty) and the cache's vector matrix."""
        inv, uniq = pd.factorize(pd.Series(texts, dtype=object))
        uniq = np.asarray(uniq, dtype=object)
        keys = _hash64(uniq)
        root = self._cache_root(field)
        roots = [root, *self._readonly_roots(field)] if self.use_cache else []
        cache = _Cache(roots)
        urows = cache.find(keys)
        miss = np.flatnonzero((urows < 0) & (uniq != ""))
        if len(miss):
            t0 = time.time()
            new = self._encoder().encode(list(uniq[miss]))
            dt = time.time() - t0
            self.stats["n_embedded"] += len(miss)
            self.stats["embed_s"] += dt
            print(f"[p4_e5] {field}: embedded {len(miss):,} new texts in {dt:.1f}s ({len(miss) / max(dt, 1e-9):,.0f}/s); "
                  f"{len(uniq) - len(miss):,} from cache", flush=True)
            if self.use_cache:
                write_shard(root, keys[miss], new)
                del new
                cache = _Cache(roots)
                urows = cache.find(keys)
            else:
                cache.vecs = new
                urows = np.full(len(uniq), -1, np.int64)
                urows[miss] = np.arange(len(miss))
        else:
            print(f"[p4_e5] {field}: all {len(uniq):,} texts from cache", flush=True)
        urows = np.where(uniq == "", -1, urows)
        return urows[inv], cache.vecs

    def transform(self, ds: Dataset, pairs: pd.DataFrame) -> pd.DataFrame:
        ids = pd.unique(np.concatenate([pairs["s1_id"].to_numpy(), pairs["cand_id"].to_numpy()]))
        recs = ds.lookup(ids, ["business_name", "business_address"])
        index = pd.Index(ids)
        a, b = index.get_indexer(pairs["s1_id"]), index.get_indexer(pairs["cand_id"])
        f = {}
        for field in self.fields:
            rows, vecs = self._rows(field_texts(recs, field), field)
            ra, rb = rows[a], rows[b]
            ok = (ra >= 0) & (rb >= 0)
            cos = np.full(len(pairs), np.nan, dtype=np.float32)
            step = 2_000_000
            for i in range(0, len(pairs), step):
                sl = slice(i, i + step)
                m = ok[sl]
                va = vecs[ra[sl][m]].astype(np.float32)
                vb = vecs[rb[sl][m]].astype(np.float32)
                c = cos[sl]
                c[m] = np.einsum("ij,ij->i", va, vb)
            del vecs, rows, ra, rb, ok
            f[f"f_p4_e5_{field}_cos"] = cos
            if self.group_feats:
                sides = {"s1": pairs["s1_id"].to_numpy(), "rec": pairs["cand_id"].to_numpy()}
                for side in self.group_sides:
                    gap, rank = _group_feats(cos, sides[side])
                    sfx = "" if side == "s1" else "_rec"
                    f[f"f_p4_e5_{field}{sfx}_gap"] = gap
                    f[f"f_p4_e5_{field}{sfx}_rank"] = rank
        return pd.DataFrame(f, index=pairs.index)
