"""Kaggle batch kernel for p4 T16b (cross-fitting check; 2x T4, 4 cores, ~31 GB; internet for pip + one-time HF download).
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
   "load_part(os.environ['BER_DATA_DIR'], 'train', cache_dir='/tmp/ber_artifacts/cache')\"",
   "rawcache", cwd=pkg, env=env)
sh("python -c \"from transformers import AutoModel, AutoTokenizer; m='intfloat/multilingual-e5-small'; "
   "AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)\"", "hf", env=env)
off = dict(env, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")



def inp(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}/config.json", recursive=True)
    return str(Path(hits[0]).parent) if hits else ""
M = W / "models"
ft = dict(off, HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0")

# T16b: does cross-fitting change the result vs naive fine-tuning? val_v1@0.1, same base config (best.yaml +
# string_sims + E5 name/full): pretrained p1_e5 | naive (v1 model fine-tuned on the whole val_v1 train side, from kernel
# ber-p4-t12ft) | cross-fitted p4_e5cf (two v1-recipe models on the train side's fold-0 / fold-1 S1 pairs).
AM = Path("/tmp/ber_artifacts/models")
ok = all(sh(f"python exploration/p4/e5_finetune.py --split val_v1+fold2:{k} --out {AM}/ft_h{k} --n 400000",
            f"ft_h{k}", cwd=REPO, env=ft) == 0 for k in (0, 1))
sh("python src/run.py run -c configs/exp/p4/T16_val_p1e5.yaml --task T16 --split val_v1@0.1 --note 'pretrained E5'",
   "harness_pre", cwd=pkg, env=off)
naive = inp("ft_val_v1")
if naive:
    sh(f"python src/run.py run -c configs/exp/p4/T16_val_p1e5.yaml --task T16 --split val_v1@0.1 "
       f"--set features.1.params.model={naive} --note 'naive fine-tuned E5 (v1, whole train side)'", "harness_naive", cwd=pkg, env=off)
if ok:
    sh("python src/run.py run -c configs/exp/p4/T16_val_cf.yaml --task T16 --split val_v1@0.1 --note 'cross-fitted E5'",
       "harness_cf", cwd=pkg, env=off)
    sh(f"mkdir -p {M} && cp {AM}/ft_h0/p4_finetune_stats.json {M}/ft_h0_stats.json && cp {AM}/ft_h1/p4_finetune_stats.json {M}/ft_h1_stats.json", "done")
sh("free -g", "done")
