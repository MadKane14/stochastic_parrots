"""FROZEN. Official metric (macro F0.5 per Source 1 entity, singletons included) + diagnostics.

Per-entity rule (problem statement):
    truth empty & pred empty        -> 1.0
    truth empty & pred non-empty    -> 0.0
    truth non-empty & pred empty    -> 0.0
    otherwise P = tp/|pred|, R = tp/|truth|, F = (1+b^2)PR / (b^2 P + R)  (0 if tp = 0)
Final score = mean over ALL Source 1 entities in the evaluation set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BETA = 0.5


def entity_fbeta(pred: set, truth: set, beta: float = BETA) -> float:
    if not truth and not pred:
        return 1.0
    if not truth or not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def pairs_to_sets(pairs: pd.DataFrame, s1_ids) -> dict[str, set]:
    out = {s: set() for s in s1_ids}
    for s, c in zip(pairs["s1_id"], pairs["cand_id"]):
        if s in out:
            out[s].add(c)
    return out


def per_entity_scores(pred: dict[str, set], gt: dict[str, frozenset], s1_ids) -> np.ndarray:
    return np.array([entity_fbeta(pred.get(s, set()), set(gt[s])) for s in s1_ids], dtype=float)


def macro_fbeta(pred: dict[str, set], gt: dict[str, frozenset], s1_ids) -> float:
    return float(per_entity_scores(pred, gt, s1_ids).mean()) if len(s1_ids) else float("nan")


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return [round(float(np.quantile(means, 0.025)), 5), round(float(np.quantile(means, 0.975)), 5)]


def _auc(y: np.ndarray, s: np.ndarray) -> tuple[float | None, float | None]:
    if len(y) == 0 or y.min() == y.max():
        return None, None
    from sklearn.metrics import average_precision_score, roc_auc_score
    return round(float(roc_auc_score(y, s)), 5), round(float(average_precision_score(y, s)), 5)


def evaluate(s1: pd.DataFrame, gt: dict[str, frozenset], pairs: pd.DataFrame, matches: pd.DataFrame,
             n_right: int) -> dict:
    """s1: Source 1 records (entity_id, country); pairs: candidates (+score); matches: final pairs."""
    s1_ids = s1["entity_id"].tolist()
    pred = pairs_to_sets(matches, s1_ids)
    cands = pairs_to_sets(pairs, s1_ids)
    f = per_entity_scores(pred, gt, s1_ids)
    is_single = np.array([len(gt[s]) == 0 for s in s1_ids])
    pred_empty = np.array([len(pred[s]) == 0 for s in s1_ids])
    n_true = sum(len(gt[s]) for s in s1_ids)
    tp = sum(len(pred[s] & gt[s]) for s in s1_ids)
    n_pred = sum(len(pred[s]) for s in s1_ids)

    m: dict = {
        "f05": round(float(f.mean()), 5),
        "f05_ci95": bootstrap_ci(f),
        "n_s1": len(s1_ids),
        "singleton_rate": round(float(is_single.mean()), 4),
        "singleton_acc": round(float(pred_empty[is_single].mean()), 4) if is_single.any() else None,
        "f05_nonsingleton": round(float(f[~is_single].mean()), 5) if (~is_single).any() else None,
        "pred_empty_rate": round(float(pred_empty.mean()), 4),
        "pair_precision": round(tp / n_pred, 5) if n_pred else None,
        "pair_recall": round(tp / n_true, 5) if n_true else None,
        "n_pred_pairs": int(n_pred),
    }
    countries = s1["country"].tolist()
    m["by_country"] = {
        c: {"n": int(sum(1 for x in countries if x == c)),
            "f05": round(float(f[[x == c for x in countries]].mean()), 5)}
        for c in sorted(set(countries))
    }

    # ---- blocking quality (recall ceiling) ----
    n_c = np.array([len(cands[s]) for s in s1_ids])
    cand_tp = sum(len(cands[s] & gt[s]) for s in s1_ids)
    oracle = per_entity_scores({s: cands[s] & gt[s] for s in s1_ids}, gt, s1_ids)
    by_src = {}
    for src in ("S2", "S3"):
        tot = sum(1 for s in s1_ids for x in gt[s] if x.startswith(src))
        hit = sum(1 for s in s1_ids for x in (cands[s] & gt[s]) if x.startswith(src))
        by_src[src] = round(hit / tot, 5) if tot else None
    m["blocking"] = {
        "pair_recall": round(cand_tp / n_true, 5) if n_true else None,
        "pair_recall_by_source": by_src,
        "entity_full_recall": round(float(np.mean([gt[s] <= cands[s] for s in s1_ids if gt[s]])), 5) if n_true else None,
        "oracle_f05": round(float(oracle.mean()), 5),
        "cands_per_s1_mean": round(float(n_c.mean()), 2) if len(n_c) else 0,
        "cands_per_s1_median": float(np.median(n_c)) if len(n_c) else 0,
        "cands_per_s1_p95": float(np.quantile(n_c, 0.95)) if len(n_c) else 0,
        "cands_per_s1_max": int(n_c.max()) if len(n_c) else 0,
        "n_pairs": int(n_c.sum()),
        "reduction_ratio": round(1 - n_c.sum() / max(1, len(s1_ids) * n_right), 6),
        "s1_without_candidates": int((n_c == 0).sum()),
    }

    # ---- scorer quality on candidate pairs + diagnostic threshold curve ----
    if "score" in pairs.columns and len(pairs):
        y = np.fromiter((c in gt.get(s, ()) for s, c in zip(pairs["s1_id"], pairs["cand_id"])), dtype=np.int8,
                        count=len(pairs))
        sc = pairs["score"].to_numpy(dtype=float)
        auc, ap = _auc(y, sc)
        m["scorer"] = {"roc_auc": auc, "avg_precision": ap}
        curve = {}
        for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            keep = pairs.loc[sc >= t, ["s1_id", "cand_id"]]
            curve[str(t)] = round(macro_fbeta(pairs_to_sets(keep, s1_ids), gt, s1_ids), 5)
        m["diag_plain_threshold_curve"] = curve  # DIAGNOSTIC ONLY (optimistic if you pick t from it)
    return m
