"""Kaggle batch kernel for p3 (CPU: 4 cores, 30 GB, no swap). Inputs (private datasets):
  ber-student-resource : data/dataset/{train,test}/*.tsv (the organizers' student_resource contents)
  ber-p3-code          : p3.bundle (git bundle of branch w/p3 -> results JSONs carry a clean commit hash)
Outputs in /kaggle/working: results/p3/*.json, logs/*.log, blk_eval_out.md, summary.txt.
Stages are listed in STAGES (edit before `kaggle kernels push`); every stage runs even if an earlier one failed
(except setup); failures are listed in summary.txt. Adapted from exploration/p4/kaggle/ber_p4_t06.py.
"""
import glob
import os
import subprocess
import time
from pathlib import Path

STAGES = ["t01", "blk", "prof", "val"]
BLK_CONFIGS = ["configs/best.yaml", "configs/exp/p3/T05_cg_name.yaml", "configs/exp/p3/T05_cg_addr.yaml",
               "configs/exp/p3/T05_cg_core.yaml"]
BLK_QFRAC = "0.1"
VAL_CONFIGS = ["configs/exp/p3/T05_cg_addr.yaml"]

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


bundle = find("p3.bundle")
data_dir = str(Path(find("train_source1.tsv")).parents[1])
sh(f"git clone -b w/p3 {bundle} {REPO} && git -C {REPO} log -1 --oneline", "setup")
pkg = REPO / "code" / "business_entity_resolution"
sh("nproc; free -g; cat /sys/fs/cgroup/memory.max 2>/dev/null; df -h /tmp /kaggle/working; python --version; "
   "lscpu | grep -E 'Model name|^CPU\\(s\\)'", "setup")
sh(f"pip install -q -r {pkg}/requirements.txt", "pip")
sh("pip freeze | grep -iE '^(numpy|pandas|scipy|scikit-learn|rapidfuzz|lightgbm|pyarrow|pyyaml)=='", "pip")

env = dict(os.environ, BER_MEMBER="p3", BER_DATA_DIR=data_dir, BER_ARTIFACTS="/tmp/ber_artifacts",
           BER_RESULTS_DIR=str(W / "results"), BER_THREADS="4", PYTHONUNBUFFERED="1")
run = "python src/run.py"
if "t01" in STAGES:
    sh(f"{run} check", "t01_check", cwd=pkg, env=env)
    sh("python tests/test_metrics.py", "t01_check", cwd=pkg, env=env)
    sh(f"{run} run --smoke --task T01", "t01_smoke", cwd=pkg, env=env)
    sh(f"{run} run -c configs/exp/p3/T05_cg_addr.yaml --smoke --task T05", "t05_smoke", cwd=pkg, env=env)
    sh(f"cp -r {W}/results/p3/smoke {W}/smoke_results 2>/dev/null; du -sh /tmp/ber_artifacts", "t01_smoke")
if "blk" in STAGES:
    for cfg in BLK_CONFIGS:
        sh(f"python exploration/p3/blk_union.py --config code/business_entity_resolution/{cfg} --qfrac {BLK_QFRAC} "
           f"--caps 40,50,60 --out {W}/blk_eval_out.md", "blk", cwd=REPO, env=env)
    sh("ls -la /tmp/ber_artifacts/p3_blk; free -g", "blk")
if "prof" in STAGES:
    sh(f"python exploration/p3/gram_df_profile.py --config code/business_entity_resolution/{BLK_CONFIGS[1]} "
       f"--out {W}/gram_df_profile_out.md", "prof", cwd=REPO, env=env)
if "misses" in STAGES:
    sh(f"python exploration/p3/blk_union.py --config code/business_entity_resolution/{BLK_CONFIGS[-1]} "
       f"--qfrac {BLK_QFRAC} --misses 50 --out {W}/blk_misses_out.md", "misses", cwd=REPO, env=env)
if "val" in STAGES:
    for cfg in VAL_CONFIGS:
        sh(f"{run} run -c {cfg} --task T05", "val", cwd=pkg, env=env)
sh("du -sh /tmp/ber_artifacts/*; ls -R /kaggle/working/results", "done")
