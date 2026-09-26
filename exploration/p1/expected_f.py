"""Expected macro F0.5 of a run's predictions if its own probabilities were the truth (p1).

For each S1, candidate labels are drawn as Bernoulli(score) (20 draws) and the run's matches are scored with the
F0.5 rule. On in-distribution validation this tracked the true score (val_v1@0.1: 0.990 vs 0.9917); it ignores
true pairs the blocker never proposed and is optimistic on unseen countries (xc_India@0.1: 0.975 vs 0.957).
    python exploration/p1/expected_f.py <run_id> [<run_id> ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("/scratch2/madhur.kankane/ber_artifacts")


def expected_f(rid: str, country: pd.Series, draws: int = 20) -> dict:
    sp = pd.read_parquet(ART / f"runs/{rid}/scored_pairs.parquet", columns=["s1_id", "cand_id", "score"])
    m = pd.read_csv(ART / f"runs/{rid}/matches.tsv", sep="\t", dtype=str)
    pr = pd.MultiIndex.from_frame(sp[["s1_id", "cand_id"]]).isin(pd.MultiIndex.from_frame(m[["s1_id", "cand_id"]]))
    codes, uniq = pd.factorize(sp["s1_id"])
    n, p = len(uniq), sp["score"].to_numpy()
    npred = np.bincount(codes, weights=pr, minlength=n)
    rng, F = np.random.default_rng(0), np.zeros(n)
    for _ in range(draws):
        t = rng.random(len(p)) < p
        tp = np.bincount(codes, weights=t & pr, minlength=n)
        nt = np.bincount(codes, weights=t, minlength=n)
        P, R = np.where(npred > 0, tp / np.maximum(npred, 1), 0), np.where(nt > 0, tp / np.maximum(nt, 1), 0)
        F += np.where((nt == 0) & (npred == 0), 1.0, np.where(tp > 0, 1.25 * P * R / np.maximum(.25 * P + R, 1e-9), 0)) / draws
    c = pd.Series(uniq).map(country).to_numpy()
    return {**{k: round(float(F[c == k].mean()), 4) for k in sorted(set(c))}, "all": round(float(F.mean()), 4)}


if __name__ == "__main__":
    test = pd.read_parquet(sorted(ART.glob("cache/raw_test_*.parquet"))[0], columns=["entity_id", "country"])
    country = test.set_index("entity_id")["country"]
    for rid in sys.argv[1:]:
        print(rid, expected_f(rid, country), flush=True)
