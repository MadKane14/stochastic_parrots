#!/usr/bin/env python
"""Build <team>_submission.zip in the exact layout the problem statement requires. Stdlib only.

    python scripts/package_submission.py --run-dir <BER_ARTIFACTS>/submissions/<run_id> --team stochastic_parrots

Zips the two TSVs straight from --run-dir (no copy into the repo: outputs are large and $HOME may be a small NFS
quota); the zip goes to --out, default $BER_ARTIFACTS/<team>_submission.zip (repo root if BER_ARTIFACTS is unset):
    output/ (2 TSVs), code/business_entity_resolution/ (src, README.md, requirements.txt, configs, tests),
    Documentation_template.md
"""
import argparse
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="folder with matching_results.tsv + candidate_pairs.tsv")
    ap.add_argument("--team", default="stochastic_parrots")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    doc = ROOT / "Documentation_template.md"
    problems = []
    if not doc.exists():
        problems.append("Documentation_template.md missing at repo root")
    pkg = ROOT / "code" / "business_entity_resolution"
    for need in ("src", "README.md", "requirements.txt"):
        if not (pkg / need).exists():
            problems.append(f"code/business_entity_resolution/{need} missing")
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if not (run_dir / f).exists():
            problems.append(f"{run_dir / f} missing")
    if problems:
        raise SystemExit("\n".join(problems))
    zdir = Path(os.environ["BER_ARTIFACTS"]) if os.environ.get("BER_ARTIFACTS") else ROOT
    zpath = Path(a.out or zdir / f"{a.team}_submission.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(run_dir / f, f"output/{f}")
        for p in sorted(pkg.rglob("*")):
            if p.is_file() and not SKIP_PARTS & set(p.parts):
                z.write(p, p.relative_to(ROOT).as_posix())
        if doc.exists():
            z.write(doc, "Documentation_template.md")
    print(f"wrote {zpath}")
    for p in problems:
        print("WARNING:", p)
    print("Now: unzip into a clean folder, pip install -r requirements.txt in a fresh env, and re-run the README steps.")


if __name__ == "__main__":
    main()
