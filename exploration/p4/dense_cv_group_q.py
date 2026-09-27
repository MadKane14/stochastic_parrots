"""p4: p2's dense_cv_group.py on a HASH SAMPLE of the group's S1 entities (all S2/S3 records of the group kept), so the
full stack fits a 31 GB Kaggle machine. The India full stack (p1_match + p1_extra + p1_e5 + p1_noise, ~29.4M pairs)
peaks at 28–29 GB after the features and is killed when they are concatenated (T19, T20 on Kaggle, Sat night).

Kept: every S2/S3 record of the group -> forward candidate density per kept S1 is unchanged; S1 entities with
unit_hash(s1_id, 5) < qfrac (same hash as blk_eval's --qfrac). Caveat (as for blk_eval --qfrac): the reverse top-k per
S2/S3 record now chooses among fewer S1, so reverse candidates are slightly more generous; IDF statistics see fewer
S1 documents. Identical for the two configs of a comparison -> use only the difference between runs of this script.
    python exploration/p4/dense_cv_group_q.py -c <cfg.yaml> --group India --qfrac 0.5 [--tag x]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code/business_entity_resolution/src"))
_spec = importlib.util.spec_from_file_location("p2_dense_cv_group", REPO / "exploration/p2/dense_cv_group.py")
dcg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dcg)

from ber.data import Dataset, unit_hash  # noqa: E402


def main():
    argv = sys.argv[1:]
    q = 0.5
    if "--qfrac" in argv:
        i = argv.index("--qfrac")
        q = float(argv[i + 1])
        del argv[i:i + 2]
    group_loader = dcg.group_loader

    def group_loader_q(group: str):
        base = group_loader(group)

        def load(data_dir, part, cache_dir=None):
            ds = base(data_dir, part, cache_dir)
            recs = ds.records
            is_s1 = (recs["source"] == 1).to_numpy()
            keep = ~is_s1 | recs["entity_id"].map(lambda e: unit_hash(e, 5) < q).to_numpy()
            recs = recs[keep].reset_index(drop=True)
            kept = set(recs.loc[recs["source"] == 1, "entity_id"])
            gt = {s: v for s, v in ds.gt.items() if s in kept}
            print(f"[dense_cv_group_q] {group} qfrac {q}: {len(kept):,} of {int(is_s1.sum()):,} S1 kept, "
                  f"{int((~is_s1).sum()):,} S2/S3 records (all)", flush=True)
            return Dataset(f"{ds.name}@q{q}", recs, gt, dict(ds.meta, qfrac=q))
        return load

    dcg.group_loader = group_loader_q                  # p2's main() looks the name up at call time
    tag_i = argv.index("--tag") + 1 if "--tag" in argv else None
    if tag_i is not None:
        argv[tag_i] = f"{argv[tag_i]}_q{q}"
    else:
        argv += ["--tag", f"q{q}"]
    sys.argv = [sys.argv[0], *argv]
    dcg.main()


if __name__ == "__main__":
    main()
