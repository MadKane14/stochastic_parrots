"""Kaggle batch kernel: p2's T04 evaluations run on p4's Kaggle (CPU, 4 cores / ~30 GB, internet for pip only).
Code = p2's commit on branch w/p2 (from the p4.bundle dataset), untouched, so results carry p2's commit hash.
MODE "blk": full-density blocking evals (exploration/p2/blk_eval_p2.py, --qfrac 0.1): best.yaml, then T04 A, D, C, B.
MODE "harness": harness runs of T04 A, D and best.yaml on val_v1@0.1 + xc_India@0.1 (BER_MEMBER=p4, --task T04).
Outputs in /kaggle/working: blk_eval_out.md + blk_eval_json/, results/p4/*.json, logs/*.log, summary.txt.
"""
import glob
import os
import subprocess
import time
from pathlib import Path

MODE = "harness"   # set per kernel folder: "blk" | "harness"

W = Path("/kaggle/working")
LOGS = W / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
REPO = Path("/tmp/repo")
summary = []


def sh(cmd, log, cwd=None, env=None):
    t0 = time.time()
    with open(LOGS / f"{log}.log", "a") as f:
        f.write(f"$ {cmd}\n")
        f.flush()
        rc = subprocess.run(cmd, shell=True, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    line = f"{time.strftime('%H:%M:%S')} rc={rc} {time.time() - t0:7.0f}s  {log}: {cmd}"
    print(line, flush=True)
    summary.append(line)
    (W / "summary.txt").write_text("\n".join(summary) + "\n")
    return rc


def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    if not hits:
        raise SystemExit(f"input not found: {pattern}")
    return hits[0]


bundle = find("p4.bundle")
data_dir = str(Path(find("train_source1.tsv")).parents[1])
sh(f"git clone -b w/p2 {bundle} {REPO} && git -C {REPO} log -1 --oneline", "setup")
pkg = REPO / "code" / "business_entity_resolution"
sh("nproc; free -g; df -h /tmp /kaggle/working; python --version", "setup")
sh(f"pip install -q -r {pkg}/requirements.txt", "pip")
env = dict(os.environ, BER_MEMBER="p4", BER_DATA_DIR=data_dir, BER_ARTIFACTS="/tmp/ber_artifacts",
           BER_RESULTS_DIR=str(W / "results"), BER_THREADS="4", PYTHONUNBUFFERED="1")
sh("python src/run.py check", "check", cwd=pkg, env=env)
sh("python exploration/p2/test_translit.py", "check", cwd=REPO, env=env)
cfgs = {"base": "configs/best.yaml", "T04_a": "configs/exp/p2/T04_a.yaml", "T04_d": "configs/exp/p2/T04_d.yaml",
        "T04_c": "configs/exp/p2/T04_c.yaml", "T04_b": "configs/exp/p2/T04_b.yaml"}

if MODE == "blk":
    for label in ["base", "T04_a", "T04_d", "T04_c", "T04_b"]:
        sh(f"python exploration/p2/blk_eval_p2.py --config code/business_entity_resolution/{cfgs[label]} --qfrac 0.1 "
           f"--label {label} --out {W}/blk_eval_out.md --json {W}/blk_eval_json/{label}_q0.1.json",
           f"blk_{label}", cwd=REPO, env=env)
        sh("rm -f /tmp/ber_artifacts/cache/p2blk_norm_*", "cleanup")   # one normalized copy of train at a time
elif MODE == "harness":
    note = "p2 T04 run on p4 Kaggle (planner/lead: p4 compute for p2)"
    smoke_ok = sh(f"python src/run.py run -c {cfgs['T04_a']} --smoke --task T04", "smoke", cwd=pkg, env=env) == 0
    if smoke_ok:
        for label in ["T04_a", "T04_d", "base"]:
            sh(f'python src/run.py run -c {cfgs[label]} --task T04 --split val_v1@0.1,xc_India@0.1 --note "{note}"',
               f"run_{label}", cwd=pkg, env=env)
sh("free -g; du -sh /tmp/ber_artifacts/* 2>/dev/null; ls -R /kaggle/working", "done")
