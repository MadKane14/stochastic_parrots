"""Blend two test-mode runs (p1): average their pair probabilities, apply the same decision rule, write + validate.

    python exploration/p1/blend_runs.py <run_id_a> <run_id_b> [--threshold 0.7] [--out-name blend_<a>_<b>] [--combine min]

Reads $BER_ARTIFACTS/runs/<run>/scored_pairs.parquet of both runs (s1_id, cand_id, score). Candidate set = union of
both runs' candidates (identical when both used the same pruner); a pair scored by one run only keeps that score.
Decision = p1_assign (one owner per S2/S3 record, then threshold). Writes both TSVs to
$BER_ARTIFACTS/submissions/<out-name>/ and runs our validator + the organizers' validator.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code/business_entity_resolution/src"))
from ber import submission  # noqa: E402
from ber.config import env  # noqa: E402
from ber.data import load_part  # noqa: E402
from ber.deciders.p1_assign import AssignDecider  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--combine", choices=["mean", "min"], default="mean", help="min = both runs must pass the threshold")
    a = ap.parse_args()
    art = Path(env("BER_ARTIFACTS"))
    cols = ["s1_id", "cand_id", "score"]
    A = pd.read_parquet(art / "runs" / a.run_a / "scored_pairs.parquet", columns=cols)
    B = pd.read_parquet(art / "runs" / a.run_b / "scored_pairs.parquet", columns=cols)
    m = A.merge(B, on=["s1_id", "cand_id"], how="outer", suffixes=("_a", "_b"))
    both = m["score_a"].notna() & m["score_b"].notna()
    print(f"pairs: a {len(A):,}  b {len(B):,}  union {len(m):,}  both {both.sum():,}  "
          f"corr {m.loc[both, ['score_a', 'score_b']].corr().iloc[0, 1]:.4f}", flush=True)
    m["score"] = m[["score_a", "score_b"]].mean(axis=1) if a.combine == "mean" else m[["score_a", "score_b"]].min(axis=1)
    matches = AssignDecider(threshold=a.threshold).decide(None, m[cols])
    for name, s in (("a", "score_a"), ("b", "score_b")):
        mm = AssignDecider(threshold=a.threshold).decide(None, m.dropna(subset=[s])[["s1_id", "cand_id", s]]
                                                        .rename(columns={s: "score"}))
        k = set(zip(matches.s1_id, matches.cand_id))
        agree = sum((x, y) in k for x, y in zip(mm.s1_id, mm.cand_id))
        print(f"matches {name}: {len(mm):,}; shared with blend: {agree:,}", flush=True)
    print(f"blend matches: {len(matches):,}", flush=True)
    test = load_part(env("BER_DATA_DIR"), "test", cache_dir=art / "cache")
    out = art / "submissions" / (a.out_name or f"blend_{a.run_a[:15]}_{a.run_b[:15]}")
    submission.write_submission(matches, m[cols], test.s1_ids, out)
    issues = submission.validate(out, test.s1_ids, set(test.right["entity_id"]))
    print("our validator issues:", issues or "none")
    print("official:", (submission.run_official_validator(out, Path(env("BER_DATA_DIR"))) or "n/a").splitlines()[0])
    s1c = test.s1.set_index("entity_id")["country"]
    per = matches.groupby("s1_id").size().reindex(test.s1_ids, fill_value=0)
    print((pd.DataFrame({"c": s1c.reindex(per.index).to_numpy(), "n": per.to_numpy()})
           .groupby("c")["n"].agg(pred_empty=lambda x: round(float((x == 0).mean()), 4),
                                  matches_per_s1=lambda x: round(float(x.mean()), 3))).to_string())
    print("written:", out)


if __name__ == "__main__":
    main()
