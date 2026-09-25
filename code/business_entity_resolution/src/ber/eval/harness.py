"""FROZEN. Runs one experiment and writes exactly one machine-generated results JSON.

    results/<member>/<run_id>.json          (committed; the ONLY source of numbers for planning)
    results/<member>/smoke/<run_id>.json    (gitignored smoke runs)
    $BER_ARTIFACTS/runs/<run_id>/           (scored pairs parquet, matches, code diff — not in git)
    $BER_ARTIFACTS/submissions/<run_id>/    (test mode: the two submission TSVs)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from ber.config import PKG_ROOT, REPO_ROOT, config_hash, env
from ber.data import Dataset, load_part
from ber.eval.metrics import evaluate
from ber.eval.splits import make_split, split_fingerprint, subsample

EVAL_DIR = Path(__file__).resolve().parent
LOCK_PATH = REPO_ROOT / "planning" / "eval_lock.json"
SMOKE_FRAC = 0.01        # smoke = 1% of entities (~22k S1 on the real train set): plumbing check only
HARNESS_VERSION = "2"
DEV_SEED = 11             # seed of the @frac dev subsamples (different from the smoke seed)


def eval_code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(EVAL_DIR.glob("*.py")):
        h.update(p.name.encode() + b"\0" + p.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:16]


def load_lock() -> dict | None:
    try:
        return json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def git_info() -> dict:
    code_rel = PKG_ROOT.relative_to(REPO_ROOT).as_posix()
    dirty_files = [l[3:] for l in _git("status", "--porcelain", "--", code_rel).splitlines() if l.strip()]
    diff = _git("diff", "HEAD", "--", code_rel)
    return {"commit": _git("rev-parse", "HEAD")[:12] or None, "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or None,
            "dirty": bool(dirty_files), "dirty_files": dirty_files[:20],
            "diff_sha": hashlib.sha256(diff.encode()).hexdigest()[:12] if diff else None, "_diff": diff}


def machine_info() -> dict:
    gpu = None
    if shutil.which("nvidia-smi"):
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                                 text=True, timeout=10).stdout.strip().replace("\n", "; ")
        except Exception:
            gpu = "unknown"
    return {"host": platform.node(), "python": sys.version.split()[0], "platform": platform.platform(), "gpu": gpu}


def _results_dir(member: str, smoke: bool) -> Path:
    base = Path(os.environ.get("BER_RESULTS_DIR") or REPO_ROOT / "results") / member
    return base / "smoke" if smoke else base


def _write(record: dict, member: str, smoke: bool) -> Path:
    out = _results_dir(member, smoke) / f"{record['run_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return out


def parse_split(spec: str) -> tuple[str, float]:
    """'val_v1' -> ('val_v1', 1.0); 'val_v1@0.1' -> ('val_v1', 0.1): a fixed, deterministic dev subsample
    (same fraction of S1 entities + their matches + unmatched records on BOTH sides of the split)."""
    name, _, frac = spec.partition("@")
    return name, float(frac) if frac else 1.0


def run_experiment(cfg: dict, split: str, mode: str = "val", smoke: bool = False, task: str = "adhoc",
                   note: str = "", raise_errors: bool = True, use_cache: bool | None = None) -> dict:
    """mode='val': fit on split train side, evaluate on split val side.
    mode='test': fit on all labelled data, predict the test set, write submission files.
    split may carry a dev-sample suffix: 'val_v1@0.1' (evidence-grade, locked fingerprint)."""
    from ber.pipeline import Pipeline
    from ber import submission

    t_start = time.time()
    member = env("BER_MEMBER")
    data_dir = Path(cfg["paths"]["data_dir"])
    artifacts = Path(cfg["paths"]["artifacts_dir"])
    cfg_name = Path(cfg["_meta"]["config_path"]).stem
    split_name = "test" if mode == "test" else split
    base_split, frac = ("test", 1.0) if mode == "test" else parse_split(split)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{member}_{task}_{cfg_name}_{split_name}_{config_hash(cfg)[:6]}" + ("_smoke" if smoke else "")
    git = git_info()
    lock = load_lock()
    ehash = eval_code_hash()
    record = {
        "run_id": run_id, "harness_version": HARNESS_VERSION, "status": "running", "member": member, "task": task,
        "note": note, "mode": mode, "split": split_name, "smoke": smoke, "sample_frac": SMOKE_FRAC if smoke else frac,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "config_path": cfg["_meta"]["config_path"], "overrides": cfg["_meta"]["overrides"], "config_hash": config_hash(cfg),
        "config": {k: v for k, v in cfg.items() if k != "_meta"},
        "git": {k: v for k, v in git.items() if not k.startswith("_")},
        "eval": {"eval_code_sha256": ehash, "locked_sha256": (lock or {}).get("eval_code_sha256"),
                 "eval_hash_ok": bool(lock) and lock.get("eval_code_sha256") == ehash},
        "machine": machine_info(),
    }
    try:
        full = load_part(data_dir, "train", cache_dir=artifacts / "cache")
        if mode == "test":
            train, target = full, load_part(data_dir, "test", cache_dir=artifacts / "cache")
        else:
            train, target = make_split(full, base_split)
            if frac < 1 and not smoke:   # smoke replaces the dev fraction with SMOKE_FRAC
                train, target = subsample(train, frac, seed=DEV_SEED), subsample(target, frac, seed=DEV_SEED)
            fp = split_fingerprint(train, target)
            locked_fp = ((lock or {}).get("splits") or {}).get(split, {}).get("fingerprint")
            record["eval"].update({"split_fingerprint": fp, "split_fingerprint_ok": locked_fp == fp if locked_fp else None})
        record["sizes"] = {"train_s1": len(train.s1), "train_right": len(train.right),
                           "target_s1": len(target.s1), "target_right": len(target.right)}
        if smoke:
            train, target = subsample(train, SMOKE_FRAC), subsample(target, SMOKE_FRAC)

        pipe = Pipeline(cfg, artifacts, use_cache=use_cache)
        pipe.fit(train)
        out = pipe.predict(target)
        record["train_pairs"] = pipe.train_pairs_stats
        record["transductive_components"] = pipe.transductive_components()
        record["timings_s"] = out.timings

        run_dir = artifacts / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        keep = [c for c in ("s1_id", "cand_id", "score", "label") if c in out.pairs.columns]
        out.pairs[keep].to_parquet(run_dir / "scored_pairs.parquet")
        out.matches.to_csv(run_dir / "matches.tsv", sep="\t", index=False)
        if git.get("_diff"):
            (run_dir / "code.diff").write_text(git["_diff"], encoding="utf-8")
        record["artifacts_dir"] = str(run_dir)

        if mode == "test":
            sub_dir = artifacts / "submissions" / run_id
            submission.write_submission(out.matches, out.pairs, target.s1_ids, sub_dir)
            issues = submission.validate(sub_dir, target.s1_ids, set(target.right["entity_id"]))
            record["submission"] = {"dir": str(sub_dir), "our_validator_issues": issues,
                                    "official_validator": submission.run_official_validator(sub_dir, data_dir)}
            s1 = target.s1
            pred_n = out.matches.groupby("s1_id").size()
            cand_n = out.pairs.groupby("s1_id").size()
            by_c = {}
            for c, grp in s1.groupby("country"):
                ids = grp["entity_id"]
                by_c[c] = {"n": int(len(ids)), "pred_empty_rate": round(float((pred_n.reindex(ids).fillna(0) == 0).mean()), 4),
                           "matches_per_s1": round(float(pred_n.reindex(ids).fillna(0).mean()), 3),
                           "cands_per_s1": round(float(cand_n.reindex(ids).fillna(0).mean()), 2)}
            record["test_stats"] = {"n_s1": len(s1), "by_country": by_c}
        else:
            record["metrics"] = evaluate(target.s1, target.gt, out.pairs, out.matches, n_right=len(target.right))
        record["status"] = "ok"
    except Exception as e:  # failed runs are recorded too, so overnight failures are visible
        record["status"] = "failed"
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()[-4000:]
        if raise_errors:
            record["runtime_s"] = round(time.time() - t_start, 1)
            path = _write(record, member, smoke)
            print(f"[harness] FAILED — wrote {path}", file=sys.stderr)
            raise
    record["runtime_s"] = round(time.time() - t_start, 1)
    path = _write(record, member, smoke)
    record["_path"] = str(path)
    return record
