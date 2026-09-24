"""FROZEN. Deterministic train/validation splits of the labelled training data.

Splits are grouped by Source 1 entity: an S2/S3 record that matches a validation S1 entity
lives in the validation pool (never in the training pool). Unmatched S2/S3 records are
distributed with the same ratio, so the validation pool has realistic distractors.

  val_v1       20% of S1 entities, stratified by (country, #matches capped at 3). THE main split.
  xc_<C>       hold out every S1 entity whose country == C (e.g. xc_India = train on the other
               countries, evaluate on India). Proxy for the unseen country (France) in test.
  full         no split — train on all labelled data (used for test predictions).

All assignment uses sha1 hashes of entity ids -> identical on every machine / OS.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from ber.data import Dataset, unit_hash

SPLIT_SEED = 20260925
VAL_FRAC = 0.2


def _materialize(full: Dataset, val_s1: set[str], val_right_unmatched: set[str], name: str):
    gt = full.gt
    s1_all = set(full.s1_ids)
    right_to_s1: dict[str, list[str]] = defaultdict(list)
    for s, lst in gt.items():
        for c in lst:
            right_to_s1[c].append(s)

    val_ids = set(val_s1) | set(val_right_unmatched)
    conflicts = 0
    for c, owners in right_to_s1.items():
        sides = {o in val_s1 for o in owners}
        if True in sides:
            val_ids.add(c)      # matched to at least one validation entity -> validation pool
        if len(sides) > 1:
            conflicts += 1
    recs = full.records
    is_val = recs["entity_id"].isin(val_ids)
    train_recs, val_recs = recs[~is_val].reset_index(drop=True), recs[is_val].reset_index(drop=True)
    train_ids, val_ids_final = set(train_recs["entity_id"]), set(val_recs["entity_id"])
    train_gt = {s: frozenset(x for x in gt[s] if x in train_ids) for s in s1_all - val_s1}
    val_gt = {s: frozenset(x for x in gt[s] if x in val_ids_final) for s in val_s1}
    meta = {"split": name, "conflicting_right_records": conflicts}
    return (Dataset(f"{name}:train", train_recs, train_gt, dict(meta)),
            Dataset(f"{name}:val", val_recs, val_gt, dict(meta)))


def _unmatched_right(full: Dataset) -> list[str]:
    matched = set().union(*full.gt.values()) if full.gt else set()
    return [r for r in full.right["entity_id"] if r not in matched]


def split_val(full: Dataset, frac: float = VAL_FRAC, seed: int = SPLIT_SEED, name: str = "val_v1"):
    s1 = full.s1[["entity_id", "country"]]
    strata = defaultdict(list)
    for eid, country in zip(s1["entity_id"], s1["country"]):
        strata[(country, min(len(full.gt[eid]), 3))].append(eid)
    val_s1 = set()
    for ids in strata.values():
        ids = sorted(ids, key=lambda e: unit_hash(e, seed))
        val_s1.update(ids[: int(round(frac * len(ids)))])
    val_unmatched = {r for r in _unmatched_right(full) if unit_hash(r, seed) < frac}
    return _materialize(full, val_s1, val_unmatched, name)


def split_country(full: Dataset, country: str, name: str):
    s1 = full.s1
    val_s1 = set(s1.loc[s1["country"] == country, "entity_id"])
    if not val_s1:
        raise ValueError(f"no Source 1 entities with country == {country!r}; available: {sorted(s1['country'].unique())}")
    right = full.right.set_index("entity_id")["country"]
    val_unmatched = {r for r in _unmatched_right(full) if right[r] == country}
    return _materialize(full, val_s1, val_unmatched, name)


def make_split(full: Dataset, name: str):
    """Return (train_ds, val_ds) for a split name."""
    if name == "val_v1":
        return split_val(full)
    if name.startswith("xc_"):
        return split_country(full, name[3:], name)
    raise ValueError(f"unknown split {name!r} (use val_v1 or xc_<Country>)")


def subsample(ds: Dataset, frac: float, seed: int = 7) -> Dataset:
    """Smoke-test sample: frac of S1 entities + their matched records + frac of the other records."""
    if frac >= 1:
        return ds
    s1_keep = {s for s in ds.s1_ids if unit_hash(s, seed) < frac}
    keep = set(s1_keep)
    matched_any = set()
    if ds.gt is not None:
        for s, lst in ds.gt.items():
            matched_any |= lst
            if s in s1_keep:
                keep |= lst
    for r in ds.right["entity_id"]:
        if r not in matched_any and unit_hash(r, seed) < frac:
            keep.add(r)
    if ds.gt is None:  # test: keep a random frac of right records
        keep |= {r for r in ds.right["entity_id"] if unit_hash(r, seed) < frac}
    recs = ds.records[ds.records["entity_id"].isin(keep)].reset_index(drop=True)
    gt = None if ds.gt is None else {s: frozenset(x for x in ds.gt[s] if x in keep) for s in s1_keep}
    meta = dict(ds.meta, sample_frac=frac)
    return Dataset(ds.name + f"@{frac}", recs, gt, meta)


def split_fingerprint(train: Dataset, val: Dataset) -> str:
    import hashlib
    h = hashlib.sha256()
    for eid in sorted(val.records["entity_id"]):
        h.update(eid.encode() + b"\n")
    h.update(b"|")
    for eid in sorted(train.records["entity_id"]):
        h.update(eid.encode() + b"\n")
    return h.hexdigest()[:16]
