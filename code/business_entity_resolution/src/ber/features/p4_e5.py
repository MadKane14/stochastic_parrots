"""p4 T06: multilingual E5 cosine similarities (name, address, name+address) as GBDT features.

# model: intfloat/multilingual-e5-small, licence: MIT, params: ~118M (logged at load time)
Pretrained weights only (no fine-tuning), run locally/offline after the first download (set HF_HOME).
Each unique text is embedded once ("query: " + text, mean pooling, L2-normalised, stored fp16) and cached
under $BER_ARTIFACTS/emb/<model>/<field>/shard_*.npz, keyed by a 64-bit hash of the text.
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


def _hash64(texts: list[str]) -> np.ndarray:
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


class _Cache:
    """Append-only shard store: keys (int64 text hashes) -> fp16 vectors."""

    def __init__(self, root: Path):
        self.root = root
        keys, vecs = [], []
        for p in sorted(root.glob("shard_*.npz")):
            z = np.load(p)
            keys.append(z["keys"])
            vecs.append(z["vecs"])
        self.keys = np.concatenate(keys) if keys else np.zeros(0, np.int64)
        self.vecs = np.concatenate(vecs) if vecs else None
        self._sort()

    def _sort(self):
        o = np.argsort(self.keys, kind="stable")
        self.keys = self.keys[o]
        if self.vecs is not None:
            self.vecs = self.vecs[o]

    def find(self, keys: np.ndarray) -> np.ndarray:
        """Row index in the cache for each key, -1 if missing."""
        if len(self.keys) == 0:
            return np.full(len(keys), -1)
        pos = np.searchsorted(self.keys, keys).clip(0, len(self.keys) - 1)
        return np.where(self.keys[pos] == keys, pos, -1)

    def add(self, keys: np.ndarray, vecs: np.ndarray):
        self.root.mkdir(parents=True, exist_ok=True)
        n = len(list(self.root.glob("shard_*.npz")))
        tmp = self.root / f"shard_{n:05d}.tmp.npz"
        np.savez(tmp, keys=keys, vecs=vecs)
        tmp.replace(self.root / f"shard_{n:05d}.npz")
        self.keys = np.concatenate([self.keys, keys])
        self.vecs = vecs if self.vecs is None else np.concatenate([self.vecs, vecs])
        self._sort()


@register("features", "p4_e5")
class E5Cosine(FeatureSet):
    """Cosine of E5 embeddings per candidate pair, for each field in `fields` (name / addr / full).
    Empty text on either side -> NaN. Optional `group_feats`: gap to the best cosine within the S1 entity."""

    def __init__(self, model: str = "intfloat/multilingual-e5-small", fields: list[str] | None = None,
                 max_length: int = 64, batch_size: int = 512, device: str | None = None,
                 group_feats: bool = False, cache: bool = True):
        self.model_name, self.fields = model, fields or ["name", "addr", "full"]
        self.max_length, self.batch_size, self.device = max_length, batch_size, device
        self.group_feats, self.use_cache = group_feats, cache
        self._enc = None
        self.stats = {"n_embedded": 0, "embed_s": 0.0}

    def _encoder(self) -> _Encoder:
        if self._enc is None:
            self._enc = _Encoder(self.model_name, self.max_length, self.batch_size, self.device)
        return self._enc

    def _cache_root(self, field: str) -> Path:
        art = os.environ.get("BER_ARTIFACTS") or str(Path(__file__).resolve().parents[5] / "artifacts")
        return Path(art) / "emb" / self.model_name.replace("/", "__") / f"{field}_ml{self.max_length}"

    def _embed_field(self, texts: list[str], field: str) -> tuple[np.ndarray, np.ndarray]:
        """Vectors for `texts` (zeros where empty) and a validity mask."""
        valid = np.array([bool(t) for t in texts])
        uniq = sorted({t for t in texts if t})
        if not uniq:
            return np.zeros((len(texts), 1), dtype=np.float16), valid
        keys = _hash64(uniq)
        cache = _Cache(self._cache_root(field)) if self.use_cache else None
        rows = cache.find(keys) if cache else np.full(len(uniq), -1)
        miss = np.flatnonzero(rows < 0)
        if len(miss):
            t0 = time.time()
            new = self._encoder().encode([uniq[i] for i in miss])
            dt = time.time() - t0
            self.stats["n_embedded"] += len(miss)
            self.stats["embed_s"] += dt
            print(f"[p4_e5] {field}: embedded {len(miss):,} new texts in {dt:.1f}s ({len(miss) / max(dt, 1e-9):,.0f}/s); "
                  f"{len(uniq) - len(miss):,} from cache", flush=True)
            if cache is not None:
                cache.add(keys[miss], new)
                rows = cache.find(keys)
        if cache is not None:
            uvecs = cache.vecs[rows]
        else:
            uvecs = new
        pos = {t: i for i, t in enumerate(uniq)}
        vecs = np.zeros((len(texts), uvecs.shape[1] if len(uvecs) else 1), dtype=np.float16)
        idx = np.array([pos.get(t, -1) for t in texts])
        vecs[valid] = uvecs[idx[valid]]
        return vecs, valid

    def transform(self, ds: Dataset, pairs: pd.DataFrame) -> pd.DataFrame:
        ids = pd.unique(np.concatenate([pairs["s1_id"].to_numpy(), pairs["cand_id"].to_numpy()]))
        recs = ds.lookup(ids, ["business_name", "business_address"])
        index = pd.Index(ids)
        a, b = index.get_indexer(pairs["s1_id"]), index.get_indexer(pairs["cand_id"])
        f = {}
        for field in self.fields:
            cols = _FIELDS[field]
            parts = [recs[c].map(_clean) for c in cols]
            if len(parts) == 1:
                texts = parts[0].tolist()
            else:
                texts = [", ".join(p for p in ps if p) for ps in zip(*parts)]
            vecs, valid = self._embed_field(texts, field)
            cos = np.empty(len(pairs), dtype=np.float32)
            step = 500_000
            for i in range(0, len(pairs), step):
                va, vb = vecs[a[i:i + step]].astype(np.float32), vecs[b[i:i + step]].astype(np.float32)
                cos[i:i + step] = np.einsum("ij,ij->i", va, vb)
            cos[~(valid[a] & valid[b])] = np.nan
            name = f"f_p4_e5_{field}_cos"
            f[name] = cos
            if self.group_feats:
                s = pd.Series(cos, index=pairs.index)
                g = s.groupby(pairs["s1_id"])
                f[f"f_p4_e5_{field}_gap"] = (g.transform("max") - s).to_numpy(np.float32)
                f[f"f_p4_e5_{field}_rank"] = g.rank(ascending=False, method="min").to_numpy(np.float32)
        return pd.DataFrame(f, index=pairs.index)
