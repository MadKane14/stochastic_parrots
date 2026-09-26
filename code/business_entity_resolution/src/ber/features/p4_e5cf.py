"""p4 T16: cross-fitted fine-tuned E5 cosines — honest features when the embedder was fine-tuned on train pairs.

# model: intfloat/multilingual-e5-small fine-tuned twice by exploration/p4/e5_finetune.py (--split full+fold2:0 / :1),
# licence: MIT (base), params: 117,653,760 each; fine-tuned only on the provided train pairs; offline.
S1 entities are split into 2 folds by a hash of their entity id (`cf_fold`). Model k is fine-tuned only on the true
pairs of fold-k S1 entities. A candidate pair of an S1 in fold f is embedded with model 1-f, i.e. a model that never
saw that S1's pairs — on train (GBDT fit, dense CV) and, identically, on val/test, so the feature has the same
meaning everywhere. Embeddings come from p1_e5's per-dataset memory-mapped matrices (one set per model).
Features: f_p4_e5cf_<field>_cos and rank / gap within the S1 entity and within the S2/S3 record (as p1_e5).
Nothing is fitted on the target records -> not transductive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ber.data import Dataset, unit_hash
from ber.features.p1_e5 import E5MemmapCosine, _rank_gap
from ber.interfaces import FeatureSet
from ber.registry import register

CF_SEED = 777


def cf_fold(s1_ids) -> np.ndarray:
    """Cross-fitting fold (0/1) of each S1 entity id — stable on every machine and in every dataset."""
    return np.fromiter((int(unit_hash(str(s), CF_SEED) * 2) for s in s1_ids), dtype=np.int8, count=len(s1_ids))


@register("features", "p4_e5cf")
class E5CrossFit(FeatureSet):
    """`models: [m0, m1]` with m_k fine-tuned on fold-k S1 pairs; pairs of fold-f S1 use model m_(1-f)."""

    def __init__(self, models: list[str], fields: list[str] | None = None, max_length: int = 64,
                 batch_size: int = 512, device: str | None = None, chunk: int = 2_000_000):
        if len(models) != 2:
            raise ValueError("p4_e5cf needs exactly 2 models (fold 0, fold 1)")
        self.fields = fields or ["name", "full"]
        self.chunk = int(chunk)
        self.encs = [E5MemmapCosine(model=m, fields=self.fields, max_length=max_length, batch_size=batch_size,
                                    device=device, chunk=chunk) for m in models]

    def transform(self, ds: Dataset, pairs: pd.DataFrame) -> pd.DataFrame:
        index = pd.Index(ds.records["entity_id"].to_numpy())
        li = index.get_indexer(pairs["s1_id"])
        ri = index.get_indexer(pairs["cand_id"])
        s1u, s1c = np.unique(pairs["s1_id"].to_numpy(), return_inverse=True)
        use = 1 - cf_fold(s1u)[s1c]                       # model that never saw this S1's pairs
        out = {}
        for field in self.fields:
            cos = np.full(len(pairs), np.nan, np.float32)
            for k, enc in enumerate(self.encs):
                sel = np.flatnonzero(use == k)
                if not len(sel):
                    continue
                M = enc.matrix(ds, field)
                for s in range(0, len(sel), self.chunk):
                    idx = sel[s:s + self.chunk]
                    a = np.asarray(M[li[idx]], dtype=np.float32)
                    b = np.asarray(M[ri[idx]], dtype=np.float32)
                    c = np.einsum("ij,ij->i", a, b)
                    c[(np.abs(a).sum(1) == 0) | (np.abs(b).sum(1) == 0)] = np.nan
                    cos[idx] = c
                del M
            out[f"f_p4_e5cf_{field}_cos"] = cos
            for side, gid in (("s1", li), ("rec", ri)):
                rank, gap = _rank_gap(gid, cos)
                out[f"f_p4_e5cf_{field}_{side}_rank"] = rank
                out[f"f_p4_e5cf_{field}_{side}_gap"] = gap
        return pd.DataFrame({k: v.astype(np.float16) for k, v in out.items()}, index=pairs.index)
