"""Kaggle batch kernel for p4 T06 (GPU + internet). Inputs (private datasets):
  ber-data-p4  : train/*.tsv, test/*.tsv, utils/validate_submission.py
  ber-p4-code  : p4.bundle  (git bundle of branch w/p4 -> results JSONs carry a clean commit hash)
Outputs in /kaggle/working: results/p4/*.json, logs/*.log, e5_throughput_out.md, summary.txt.
Every step runs even if an earlier one failed (except the setup); failures are listed in summary.txt.
"""
import glob
import os
import subprocess
import time
from pathlib import Path

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
sh(f"git clone -b w/p4 {bundle} {REPO} && git -C {REPO} log -1 --oneline", "setup")
pkg = REPO / "code" / "business_entity_resolution"
sh("nvidia-smi; nproc; free -g; df -h /tmp /kaggle/working; python --version", "setup")
if sh(f"pip install -q -r {pkg}/requirements-gpu.txt", "pip") != 0:
    sh(f"pip install -q -r {pkg}/requirements.txt transformers==4.57.1", "pip")   # keep the image's torch
# the image's torchvision/torchaudio are built for its own torch and break `import transformers` under the pinned
# torch (RuntimeError: operator torchvision::nms does not exist) -> remove them (the pipeline uses neither)
sh("pip uninstall -y -q torchvision torchaudio", "pip")
sh("pip freeze | grep -iE '^(numpy|pandas|scipy|scikit-learn|lightgbm|pyarrow|torch|transformers|tokenizers)=='", "pip")

env = dict(os.environ, BER_MEMBER="p4", BER_DATA_DIR=data_dir, BER_ARTIFACTS="/tmp/ber_artifacts",
           BER_RESULTS_DIR=str(W / "results"), BER_THREADS="4", TOKENIZERS_PARALLELISM="true",
           HF_HOME="/tmp/hf", CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
run = "python src/run.py"
sh(f"{run} check", "check", cwd=pkg, env=env)
sh("python tests/test_metrics.py", "check", cwd=pkg, env=env)
smoke_ok = sh(f"{run} run -c configs/exp/p4/T06_e5.yaml --smoke --task T06", "smoke", cwd=pkg, env=env) == 0
env_off = dict(env, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")   # weights are cached now: run offline
if os.environ.get("P4_RUN_BASE", "0") == "1":   # base runs of commit 7b02ab4 done in kernel v1
    sh(f"{run} run -c configs/exp/p4/T06_base.yaml --task T06 --split val_v1@0.1,xc_India@0.1", "T06_base", cwd=pkg, env=env_off)
if smoke_ok:
    sh(f"{run} run -c configs/exp/p4/T06_e5.yaml --task T06 --split val_v1@0.1,xc_India@0.1", "T06_e5", cwd=pkg, env=env_off)
sh(f"python exploration/p4/e5_throughput.py --n 200000 --out {W}/e5_throughput_out.md", "throughput", cwd=REPO, env=env_off)
if smoke_ok:
    sh(f"{run} run -c configs/exp/p4/T06_e5g.yaml --task T06 --split val_v1@0.1,xc_India@0.1", "T06_e5g", cwd=pkg, env=env_off)
sh("du -sh /tmp/ber_artifacts/*; ls -R /kaggle/working/results", "done")
