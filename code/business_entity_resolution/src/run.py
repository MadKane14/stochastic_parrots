#!/usr/bin/env python
"""Single entry point for the Business Entity Resolution pipeline.

Run from code/business_entity_resolution/:

  python src/run.py check                                   # environment + data sanity
  python src/run.py run   -c configs/base.yaml --smoke      # 1% sample, plumbing only: ALWAYS first
  python src/run.py run   -c configs/base.yaml --task T03   # dev run on the default split (val_v1@0.1 = 10% sample)
  python src/run.py run   -c configs/base.yaml --split val_v1,xc_India@0.1   # full val / country hold-out
  python src/run.py test  -c configs/best.yaml              # train on all labels, write submission TSVs
  python src/run.py sweep -c configs/exp/p3/x.yaml --grid "blocking.blockers.0.params.k=10|20|40" --task T03
  python src/run.py lock                                    # LEAD ONLY: freeze eval (planning/eval_lock.json)

Every run/test/sweep job writes one results JSON under <repo>/results/<BER_MEMBER>/.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ber.config import REPO_ROOT, env, load_config  # noqa: E402


def _summary(rec: dict) -> str:
    if rec["status"] != "ok":
        return f"[{rec['status'].upper()}] {rec['run_id']}: {rec.get('error')}"
    if rec["mode"] == "test":
        sub = rec["submission"]
        return (f"[TEST] {rec['run_id']}\n  submission: {sub['dir']}\n  our validator issues: {sub['our_validator_issues'] or 'none'}"
                f"\n  official validator: {(sub['official_validator'] or 'not found').splitlines()[0]}"
                f"\n  test stats: {json.dumps(rec['test_stats']['by_country'])}")
    m, b = rec["metrics"], rec["metrics"]["blocking"]
    flags = []
    if not rec["eval"]["eval_hash_ok"]:
        flags.append("EVAL_NOT_LOCKED_OR_MODIFIED")
    if rec["git"]["dirty"]:
        flags.append("DIRTY_CODE")
    return (f"[{rec['split']}{' SMOKE' if rec['smoke'] else ''}] {rec['run_id']}\n"
            f"  F0.5={m['f05']:.4f} CI95={m['f05_ci95']}  singleton_acc={m['singleton_acc']}  "
            f"pairP={m['pair_precision']} pairR={m['pair_recall']}\n"
            f"  blocking: oracle_F0.5={b['oracle_f05']:.4f} pair_recall={b['pair_recall']} "
            f"cands/s1={b['cands_per_s1_mean']}  scorer={m.get('scorer')}\n"
            f"  by_country={json.dumps(m['by_country'])}  runtime={rec['runtime_s']}s {' '.join(flags)}")


def _autocommit(paths: list[str], msg: str, push: bool):
    try:
        subprocess.run(["git", "add", "--", *paths], cwd=REPO_ROOT, check=True)
        subprocess.run(["git", "commit", "-m", msg, "--", *paths], cwd=REPO_ROOT, check=False, capture_output=True)
        if push:
            subprocess.run(["git", "push", "origin", "HEAD"], cwd=REPO_ROOT, check=False, capture_output=True, timeout=120)
    except Exception as e:  # never kill a sweep because of git
        print(f"[autocommit] skipped: {e}", file=sys.stderr)


def cmd_run(a, mode: str):
    from ber.eval.harness import run_experiment
    cfg = load_config(a.config, a.set)
    splits = ["test"] if mode == "test" else (a.split or cfg.get("eval", {}).get("split", "val_v1")).split(",")
    if not a.smoke and env("BER_MEMBER") == "anon":
        sys.exit("Set BER_MEMBER to your handle (p1..p4) before full runs.")
    for sp in splits:
        rec = run_experiment(cfg, sp, mode=mode, smoke=a.smoke, task=a.task, note=a.note, use_cache=not a.no_cache)
        print(_summary(rec))
        print(f"  results: {rec['_path']}")


def cmd_sweep(a):
    from ber.eval.harness import run_experiment
    if env("BER_MEMBER") == "anon":
        sys.exit("Set BER_MEMBER to your handle (p1..p4) before sweeps.")
    grid = []
    for g in a.grid:
        key, _, vals = g.partition("=")
        grid.append([f"{key}={v}" for v in vals.split("|")])
    combos = [list(c) for c in itertools.product(*grid)] or [[]]
    splits = (a.split or load_config(a.config).get("eval", {}).get("split", "val_v1@0.1")).split(",")
    print(f"[sweep] {len(combos)} configs x {len(splits)} splits")
    if not a.skip_smoke:
        rec = run_experiment(load_config(a.config, a.set + combos[0]), splits[0], smoke=True, task=a.task,
                             note="sweep smoke", raise_errors=False)
        print(_summary(rec))
        if rec["status"] != "ok":
            sys.exit("[sweep] smoke test failed — fix before launching the full sweep.")
    for i, combo in enumerate(combos):
        for sp in splits:
            rec = run_experiment(load_config(a.config, a.set + combo), sp, task=a.task,
                                 note=(a.note + " " + " ".join(combo)).strip(), raise_errors=False)
            print(f"({i + 1}/{len(combos)}) " + _summary(rec), flush=True)
            if a.autocommit:
                rel = Path(rec["_path"]).resolve().relative_to(REPO_ROOT).as_posix()
                _autocommit([rel], f"results: {rec['run_id']} [{rec['status']}]", push=a.push)


def cmd_lock(a):
    from ber.eval.lock import write_lock
    cfg = load_config(a.config)
    lock = write_lock(cfg["paths"]["data_dir"], cache_dir=Path(cfg["paths"]["artifacts_dir"]) / "cache")
    print(json.dumps(lock, indent=1))
    print("\nWrote planning/eval_lock.json — commit it to main.")


def cmd_check(a):
    import importlib
    import platform
    print(f"python {platform.python_version()}  ({sys.executable})")
    req = Path(__file__).resolve().parents[1] / "requirements.txt"
    for line in req.read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" not in line:
            continue
        pkg, want = line.split("==")
        mod = {"scikit-learn": "sklearn", "PyYAML": "yaml"}.get(pkg, pkg.replace("-", "_"))
        try:
            got = importlib.import_module(mod).__version__
            print(f"  {'OK ' if got == want else 'MISMATCH'} {pkg} {got} (pinned {want})")
        except Exception as e:
            print(f"  MISSING {pkg}: {e}")
    cfg = load_config(a.config)
    for k in ("BER_MEMBER", "BER_DATA_DIR", "BER_ARTIFACTS"):
        print(f"{k} = {env(k)}")
    dd = Path(cfg["paths"]["data_dir"])
    for part in ("train", "test"):
        files = sorted(p.name for p in (dd / part).glob("*.tsv")) if (dd / part).exists() else []
        print(f"data/{part}: {files or 'NOT FOUND'}")
    from ber.registry import available
    print("components:", json.dumps(available()))
    from ber.eval.harness import eval_code_hash, load_lock, machine_info
    lock = load_lock()
    print(f"eval hash {eval_code_hash()}  lock: {(lock or {}).get('eval_code_sha256', 'NO LOCK')}")
    print("machine:", machine_info())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    default_cfg = str(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
    for name in ("run", "test", "sweep", "lock", "check"):
        p = sub.add_parser(name)
        p.add_argument("-c", "--config", default=default_cfg)
        p.add_argument("--set", action="append", default=[], help="override, e.g. --set scorer.params.max_iter=500")
        p.add_argument("--task", default="adhoc", help="task id from planning/TASKS.md, e.g. T03")
        p.add_argument("--note", default="")
        p.add_argument("--split", default=None, help="comma-separated: val_v1@0.1 (dev, default), val_v1 (full), xc_India@0.1, xc_US ...")
        p.add_argument("--smoke", action="store_true", help="1%% sample of every split, plumbing check only (results go to results/<you>/smoke/)")
        p.add_argument("--no-cache", action="store_true")
        if name == "sweep":
            p.add_argument("--grid", action="append", default=[], help='"dotted.key=v1|v2|v3" (repeatable)')
            p.add_argument("--skip-smoke", action="store_true")
            p.add_argument("--autocommit", action="store_true", help="git-commit each results JSON as it lands")
            p.add_argument("--push", action="store_true", help="with --autocommit: push to your branch")
    a = ap.parse_args()
    {"run": lambda: cmd_run(a, "val"), "test": lambda: cmd_run(a, "test"), "sweep": lambda: cmd_sweep(a),
     "lock": lambda: cmd_lock(a), "check": lambda: cmd_check(a)}[a.cmd]()


if __name__ == "__main__":
    main()
