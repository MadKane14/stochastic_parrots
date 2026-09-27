"""Combine several scorers (e.g. a GBDT and a neural cross-encoder) by weighted mean or minimum."""
from __future__ import annotations

import numpy as np

from ber.interfaces import Scorer
from ber.registry import build, register


@register("scorer", "weighted_mean")
class WeightedMean(Scorer):
    def __init__(self, members: list[dict], weights: list[float] | None = None, combine: str = "mean", **shared):
        """members: scorer specs {name, params}; combine: "mean" (weighted) or "min" (every member must agree; weights
        ignored); shared: default params for every member (e.g. max_iter / learning_rate inherited from a parent
        config) — a member's own params take precedence."""
        if combine not in ("mean", "min"):
            raise ValueError(f"combine must be 'mean' or 'min', got {combine!r}")
        self.combine = combine
        self.members = [build("scorer", {**m, "params": {**shared, **(m.get("params") or {})}}) for m in members]
        w = np.asarray(weights or [1.0] * len(members), dtype=float)
        self.weights = w / w.sum()

    def fit(self, train, pairs):
        for m in self.members:
            m.fit(train, pairs)
        return self

    def score(self, ds, pairs):
        scores = [np.asarray(m.score(ds, pairs)) for m in self.members]
        if self.combine == "min":
            return np.clip(np.minimum.reduce(scores), 0, 1)
        return np.clip(sum(w * s for w, s in zip(self.weights, scores)), 0, 1)
