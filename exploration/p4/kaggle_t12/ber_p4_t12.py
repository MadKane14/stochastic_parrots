"""Kaggle batch kernel for p4 T12 (2x T4, 4 cores, ~31 GB; internet for pip + one-time HF download).
Dense CV itself needs 41-61 GB (p1's runs) and cannot run here, so this kernel does the parts that fit:
  1. e5_embed_all.py on train + test, one field per GPU in parallel -> unique-text counts + total embedding time
  2. plumbing: smoke + val_v1@0.1 of T12_e5 (p4_e5 on top of p1's T08_sub features) and T08_sub, same commit
     (val_v1@0.1 is plumbing only since round 3 — not evidence)
Outputs in /kaggle/working: e5_embed_all_{name,full}.json, results/p4/*.json, logs/*.log, summary.txt.
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
sh("nvidia-smi -L; nproc; free -g; python --version", "setup")
sh(f"pip install -q -r {pkg}/requirements-gpu.txt", "pip")
sh("pip uninstall -y -q torchvision torchaudio", "pip")   # image's torchvision breaks transformers under torch 2.8.0
env = dict(os.environ, BER_MEMBER="p4", BER_DATA_DIR=data_dir, BER_ARTIFACTS="/tmp/ber_artifacts",
           BER_RESULTS_DIR=str(W / "results"), BER_THREADS="4", TOKENIZERS_PARALLELISM="true",
           HF_HOME="/tmp/hf", CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
sh("python src/run.py check", "check", cwd=pkg, env=env)
# raw-data parquet caches once (the parallel embedders must not race on them) + model download once
sh("python -c \"import sys; sys.path.insert(0,'src'); from ber.data import load_part; import os; "
   "[load_part(os.environ['BER_DATA_DIR'], p, cache_dir='/tmp/ber_artifacts/cache') for p in ('train','test')]\"",
   "rawcache", cwd=pkg, env=env)
sh("python -c \"from transformers import AutoModel, AutoTokenizer; m='intfloat/multilingual-e5-small'; "
   "AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)\"", "hf", env=env)
off = dict(env, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

# 1) embed all unique train + test texts: name on GPU 0, name+address on GPU 1, in parallel
t0 = time.time()
procs = []
for gpu, field in ((0, "name"), (1, "full")):
    f = open(LOGS / f"embed_{field}.log", "w")
    procs.append((field, subprocess.Popen(
        f"python exploration/p4/e5_embed_all.py --parts train,test --fields {field} "
        f"--out-json {W}/e5_embed_all_{field}.json", shell=True, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
        env=dict(off, CUDA_VISIBLE_DEVICES=str(gpu), BER_THREADS="2"))))
for field, p in procs:
    rc = p.wait()
    summary.append(f"{time.strftime('%H:%M:%S')} rc={rc} embed_{field} (parallel, wall {time.time() - t0:.0f}s so far)")
    (W / "summary.txt").write_text("\n".join(summary) + "\n")
sh("du -sh /tmp/ber_artifacts/emb/*/*; free -g", "embed_done")

# 2) plumbing on the same commit (all E5 texts are cached now -> no GPU needed)
smoke_ok = sh("python src/run.py run -c configs/exp/p4/T12_e5.yaml --smoke --task T12", "smoke", cwd=pkg, env=off) == 0
if smoke_ok:
    sh("python src/run.py run -c configs/exp/p4/T12_e5.yaml --task T12 --split val_v1@0.1", "val_T12_e5", cwd=pkg, env=off)
    sh("python src/run.py run -c configs/exp/p1/T08_sub.yaml --task T12 --split val_v1@0.1", "val_T08_sub", cwd=pkg, env=off)
sh("free -g; ls -R /kaggle/working | head -50", "done")
