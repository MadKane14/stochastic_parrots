"""Combine several scorers (e.g. a GBDT and a neural cross-encoder) by weighted mean."""
from __future__ import annotations

import numpy as np

from ber.interfaces import Scorer
from ber.registry import build, register


@register("scorer", "weighted_mean")
class WeightedMean(Scorer):
    def __init__(self, members: list[dict], weights: list[float] | None = None):
        self.members = [build("scorer", m) for m in members]
        w = np.asarray(weights or [1.0] * len(members), dtype=float)
        self.weights = w / w.sum()

    def fit(self, train, pairs):
        for m in self.members:
            m.fit(train, pairs)
        return self

    def score(self, ds, pairs):
        return np.clip(sum(w * np.asarray(m.score(ds, pairs)) for w, m in zip(self.weights, self.members)), 0, 1)
