"""Write and validate the two submission TSVs (matching_results.tsv, candidate_pairs.tsv)."""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _lists(pairs: pd.DataFrame, s1_ids: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {s: [] for s in s1_ids}
    seen: dict[str, set] = {s: set() for s in s1_ids}
    for s, c in zip(pairs["s1_id"], pairs["cand_id"]):
        if c not in seen[s]:
            seen[s].add(c)
            out[s].append(c)
    return out


def write_tsv(pairs: pd.DataFrame, s1_ids: list[str], path: Path, col: str):
    lists = _lists(pairs, s1_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(["source1_entity_id", col])
        for s in s1_ids:
            w.writerow([s, ",".join(lists[s])])


def write_submission(matches: pd.DataFrame, candidates: pd.DataFrame, s1_ids: list[str], out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    write_tsv(matches, s1_ids, out_dir / "matching_results.tsv", "matched_entity_ids")
    write_tsv(candidates, s1_ids, out_dir / "candidate_pairs.tsv", "candidate_entity_ids")
    return {"matching": str(out_dir / "matching_results.tsv"), "candidate": str(out_dir / "candidate_pairs.tsv")}


def validate(out_dir: Path, s1_ids: list[str], right_ids: set[str]) -> list[str]:
    """Our own check of every rule in the problem statement. Returns a list of issues."""
    issues = []
    files = {"matching_results.tsv": "matched_entity_ids", "candidate_pairs.tsv": "candidate_entity_ids"}
    parsed = {}
    for fname, col in files.items():
        df = pd.read_csv(Path(out_dir) / fname, sep="\t", dtype=str, keep_default_na=False)
        if list(df.columns) != ["source1_entity_id", col]:
            issues.append(f"{fname}: bad header {list(df.columns)}")
            continue
        if df["source1_entity_id"].duplicated().any():
            issues.append(f"{fname}: duplicate source1_entity_id rows")
        missing = set(s1_ids) - set(df["source1_entity_id"])
        extra = set(df["source1_entity_id"]) - set(s1_ids)
        if missing:
            issues.append(f"{fname}: {len(missing)} Source 1 entities missing")
        if extra:
            issues.append(f"{fname}: {len(extra)} unknown source1 ids")
        lists = {}
        for s, lst in zip(df["source1_entity_id"], df[col]):
            ids = [x for x in lst.split(",") if x] if lst else []
            if len(ids) != len(set(ids)):
                issues.append(f"{fname}: duplicate ids in list of {s}")
            bad = [x for x in ids if x not in right_ids]
            if bad:
                issues.append(f"{fname}: {s} lists ids not in test S2/S3: {bad[:3]}")
            lists[s] = set(ids)
        parsed[fname] = lists
    if len(parsed) == 2:
        m, c = parsed["matching_results.tsv"], parsed["candidate_pairs.tsv"]
        n_bad = sum(1 for s in m if not m[s] <= c.get(s, set()))
        if n_bad:
            issues.append(f"{n_bad} entities have matches that are not in candidate_pairs.tsv")
    return issues[:50]


def run_official_validator(out_dir: Path, data_dir: Path) -> str | None:
    """Run the organizers' utils/validate_submission.py if it ships next to the dataset."""
    for cand in (Path(data_dir).parent / "utils" / "validate_submission.py",
                 Path(data_dir) / "utils" / "validate_submission.py"):
        if cand.exists():
            r = subprocess.run(
                [sys.executable, str(cand), "--matching", str(Path(out_dir) / "matching_results.tsv"),
                 "--candidate", str(Path(out_dir) / "candidate_pairs.tsv"), "--test-dir", str(Path(data_dir) / "test")],
                capture_output=True, text=True,
            )
            return ("PASS" if r.returncode == 0 else "FAIL") + "\n" + (r.stdout + r.stderr)[-3000:]
    return None
