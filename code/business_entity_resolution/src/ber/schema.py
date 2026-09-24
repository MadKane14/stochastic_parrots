"""Runtime checks of the stage contracts (see interfaces.py). Called by the pipeline after
each stage so a broken component fails loudly at the stage that broke it."""
from __future__ import annotations

import numpy as np
import pandas as pd


class ContractError(AssertionError):
    pass


def _req(cond: bool, msg: str):
    if not cond:
        raise ContractError(msg)


def check_records(records: pd.DataFrame, n_expected: int, where: str):
    for c in ("entity_id", "source", "business_name", "business_address", "country", "name_norm", "addr_norm"):
        _req(c in records.columns, f"[{where}] records missing column {c!r}")
    _req(len(records) == n_expected, f"[{where}] normalizer changed row count {n_expected} -> {len(records)}")


def check_candidates(pairs: pd.DataFrame, s1_ids: set, right_ids: set, where: str):
    for c in ("s1_id", "cand_id", "score"):
        _req(c in pairs.columns, f"[{where}] candidates missing column {c!r}")
    _req(not pairs.duplicated(["s1_id", "cand_id"]).any(), f"[{where}] duplicate (s1_id, cand_id) pairs")
    _req(pairs["s1_id"].isin(s1_ids).all(), f"[{where}] s1_id not a Source 1 id of this dataset")
    _req(pairs["cand_id"].isin(right_ids).all(), f"[{where}] cand_id not an S2/S3 id of this dataset")


def check_features(feats: pd.DataFrame, pairs: pd.DataFrame, where: str):
    _req(len(feats) == len(pairs) and feats.index.equals(pairs.index), f"[{where}] features not aligned with pairs")
    bad = [c for c in feats.columns if not c.startswith("f_")]
    _req(not bad, f"[{where}] feature columns must start with 'f_': {bad[:5]}")


def check_scores(scores, n: int, where: str):
    s = np.asarray(scores, dtype=float)
    _req(s.shape == (n,), f"[{where}] scores shape {s.shape} != ({n},)")
    _req(np.isfinite(s).all() and (s >= 0).all() and (s <= 1).all(), f"[{where}] scores must be finite in [0,1]")


def check_matches(matches: pd.DataFrame, pairs: pd.DataFrame, where: str):
    _req(list(matches.columns[:2]) == ["s1_id", "cand_id"], f"[{where}] matches must have columns s1_id, cand_id")
    key = set(zip(pairs["s1_id"], pairs["cand_id"]))
    _req(all(k in key for k in zip(matches["s1_id"], matches["cand_id"])), f"[{where}] match not among candidates")
    _req(not matches.duplicated(["s1_id", "cand_id"]).any(), f"[{where}] duplicate matches")
