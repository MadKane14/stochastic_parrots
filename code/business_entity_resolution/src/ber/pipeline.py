"""Pipeline orchestrator: builds components from config and runs the stages.

Config sections (see configs/base.yaml):
    normalize: {name, params}
    blocking:  {blockers: [{name, params}, ...], max_candidates: int|null}
    features:  [{name, params}, ...]
    scorer:    {name, params}
    decider:   {name, params}
    prune:     optional {scorer: {name, params}, features: [column names], min_score: float, max_per_s1: int}
               cheap stage-1 model fitted on the training candidates; at prediction time, per S1 keep candidates with
               score >= min_score (at most max_per_s1): the final scorer/decider — and candidate_pairs.tsv — see only
               the survivors. The final scorer is trained on all training candidates.
    cache:     true|false   (cache normalized records / candidates / features under artifacts)
"""
from __future__ import annotations

import hashlib
import inspect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ber import schema
from ber.data import Dataset
from ber.interfaces import feature_columns
from ber.registry import build


def _component_fp(obj) -> str:
    """Fingerprint = spec + source code of the component's module (stale cache protection)."""
    try:
        src = Path(inspect.getfile(type(obj))).read_bytes().replace(b"\r\n", b"\n")
    except (TypeError, OSError):
        src = b""
    h = hashlib.sha256(json.dumps(obj.spec, sort_keys=True, default=str).encode() + src)
    return h.hexdigest()[:12]


@dataclass
class PipelineOutput:
    pairs: pd.DataFrame          # candidates with blk_*, f_*, score (and label if gt known)
    matches: pd.DataFrame        # s1_id, cand_id
    timings: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, cfg: dict, artifacts_dir: str | Path | None = None, use_cache: bool | None = None):
        self.cfg = cfg
        self.normalizer = build("normalizer", cfg["normalize"])
        self.blockers = [build("blocker", s) for s in cfg["blocking"]["blockers"]]
        self.max_candidates = cfg["blocking"].get("max_candidates")
        self.featuresets = [build("features", s) for s in (cfg.get("features") or [])]
        self.scorer = build("scorer", cfg["scorer"])
        self.decider = build("decider", cfg["decider"])
        self.prune_cfg = cfg.get("prune")
        self.pruner = build("scorer", self.prune_cfg["scorer"]) if self.prune_cfg else None
        self.use_cache = cfg.get("cache", True) if use_cache is None else use_cache
        self.cache_dir = Path(artifacts_dir) / "cache" if artifacts_dir else None
        self.timings: dict[str, float] = {}
        self._train_fp = "unfitted"   # cache keys include the training set (stateful components)

    # ---------------- caching helpers ----------------
    def _cached(self, kind: str, key_parts: list[str], fn):
        if not (self.use_cache and self.cache_dir):
            return fn()
        key = hashlib.sha256("|".join([self._train_fp, *key_parts]).encode()).hexdigest()[:20]
        path = self.cache_dir / f"{kind}_{key}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        out = fn()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        out.to_parquet(tmp)
        tmp.replace(path)
        return out

    def _tick(self, name: str, t0: float):
        self.timings[name] = round(self.timings.get(name, 0.0) + time.time() - t0, 2)

    # ---------------- stages ----------------
    def _normalize(self, ds: Dataset) -> Dataset:
        t0 = time.time()
        n = len(ds.records)
        recs = self._cached("norm", [ds.fingerprint(), ds.name, _component_fp(self.normalizer)],
                            lambda: self.normalizer.transform(ds.records))
        schema.check_records(recs, n, f"normalize:{self.normalizer.registry_name}")
        self._tick("normalize", t0)
        return ds.with_records(recs.reset_index(drop=True))

    def _block(self, ds: Dataset) -> pd.DataFrame:
        t0 = time.time()
        s1_ids = set(ds.s1_ids)
        right_ids = set(ds.right["entity_id"])
        merged = None
        for i, b in enumerate(self.blockers):
            key = [ds.fingerprint(), ds.name, _component_fp(self.normalizer), _component_fp(b)]
            c = self._cached("cands", key, lambda b=b: b.candidates(ds))
            schema.check_candidates(c, s1_ids, right_ids, f"block:{b.registry_name}")
            c = c[["s1_id", "cand_id", "score"]].copy()
            c[f"blk_{i}_rank"] = c.groupby("s1_id")["score"].rank(ascending=False, method="first")
            c = c.rename(columns={"score": f"blk_{i}_score"})
            merged = c if merged is None else merged.merge(c, on=["s1_id", "cand_id"], how="outer")
        if merged is None:
            merged = pd.DataFrame(columns=["s1_id", "cand_id"])
        if self.max_candidates:
            rank_cols = [c for c in merged.columns if c.endswith("_rank")]
            merged["_best"] = merged[rank_cols].min(axis=1)
            merged = (merged.sort_values(["s1_id", "_best"]).groupby("s1_id").head(int(self.max_candidates))
                      .drop(columns="_best"))
        merged = merged.sort_values(["s1_id", "cand_id"]).reset_index(drop=True)
        self._tick("block", t0)
        return merged

    def _features(self, ds: Dataset, pairs: pd.DataFrame) -> pd.DataFrame:
        frames = []   # feature sets only read pairs' id columns: collect, then concatenate ONCE (one copy, not one per set)
        for fs in self.featuresets:
            t0 = time.time()
            f = fs.transform(ds, pairs)
            schema.check_features(f, pairs, f"features:{fs.registry_name}")
            frames.append(f)
            self._tick(f"features:{fs.registry_name}", t0)
        return pd.concat([pairs, *frames], axis=1) if frames else pairs

    def _prune_view(self, pairs: pd.DataFrame) -> pd.DataFrame:
        cols = ["s1_id", "cand_id"] + [c for c in ("label",) if c in pairs.columns] + list(self.prune_cfg["features"])
        return pairs[cols]

    def _prune(self, ds: Dataset, pairs: pd.DataFrame) -> pd.DataFrame:
        """Keep, per S1, candidates with stage-1 score >= min_score, best first, at most max_per_s1."""
        t0 = time.time()
        q = np.asarray(self.pruner.score(ds, self._prune_view(pairs)), dtype=float)
        keep = pd.DataFrame({"s1_id": pairs["s1_id"].to_numpy(), "q": q})
        keep = keep[keep["q"] >= float(self.prune_cfg.get("min_score", 0.0))]
        cap = self.prune_cfg.get("max_per_s1")
        if cap:
            keep = keep.sort_values(["s1_id", "q"], ascending=[True, False]).groupby("s1_id").head(int(cap))
        out = pairs.loc[np.sort(keep.index.to_numpy())].reset_index(drop=True)
        self._tick("prune", t0)
        return out

    # ---------------- public API ----------------
    def fit(self, train: Dataset) -> "Pipeline":
        self._train_fp = train.fingerprint() + train.name
        self.normalizer.fit(train)
        train = self._normalize(train)
        for b in self.blockers:
            b.fit(train)
        pairs = self._block(train)
        pairs["label"] = train.labels(pairs)
        for fs in self.featuresets:
            fs.fit(train, pairs)
        pairs = self._features(train, pairs)
        if self.pruner is not None:   # fitted on all train candidates; applied in predict() only — the final
            t0 = time.time()           # scorer keeps training on all candidates (in-sample pruning would strip
            self.pruner.fit(train, self._prune_view(pairs))   # its hard negatives)
            self._tick("prune_fit", t0)
        t0 = time.time()
        self.scorer.fit(train, pairs)
        self._tick("scorer_fit", t0)
        self.decider.fit(train, pairs)
        self.train_pairs_stats = {"n_pairs": int(len(pairs)), "pos_rate": float(pairs["label"].mean()) if len(pairs) else 0.0,
                                  "n_features": len(feature_columns(pairs))}
        return self

    def predict(self, ds: Dataset) -> PipelineOutput:
        ds = self._normalize(ds)
        pairs = self._block(ds)
        pairs = self._features(ds, pairs)
        if self.pruner is not None and len(pairs):
            pairs = self._prune(ds, pairs)
        t0 = time.time()
        scores = self.scorer.score(ds, pairs) if len(pairs) else np.zeros(0)
        schema.check_scores(scores, len(pairs), f"score:{self.scorer.registry_name}")
        pairs["score"] = scores
        self._tick("score", t0)
        matches = self.decider.decide(ds, pairs) if len(pairs) else pd.DataFrame(columns=["s1_id", "cand_id"])
        matches = matches[["s1_id", "cand_id"]].drop_duplicates().reset_index(drop=True)
        schema.check_matches(matches, pairs, f"decide:{self.decider.registry_name}")
        if ds.gt is not None:
            pairs["label"] = ds.labels(pairs)
        return PipelineOutput(pairs=pairs, matches=matches, timings=dict(self.timings))

    def transductive_components(self) -> list[str]:
        comps = [self.normalizer, *self.blockers, *self.featuresets, self.scorer, self.decider]
        return [c.registry_name for c in comps if getattr(c, "transductive", False)]
