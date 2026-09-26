"""Kaggle batch kernel for p4 T16a (recipe: unseen-country test; 2x T4, 4 cores, ~31 GB; internet for pip + one-time HF download).
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

# T16a: US-only fine-tuning, judged on India (xc_India@0.1 target candidates). v2b = v2 with equal size to v1
# (neg-frac 0.35 -> >= 400k anchors); v3 = half blocker negatives, half v1 negatives. Compared with pretrained, v1, v2.
sh("python src/run.py run -c configs/exp/p4/T06_base.yaml --task T16 --split xc_India@0.1", "cands", cwd=pkg, env=off)
pairs = sorted(glob.glob("/tmp/ber_artifacts/runs/*T06_base_xc_India@0.1_*/scored_pairs.parquet"))[-1]
variants = {"v2b": "--neg blocker --neg-frac 0.35", "v3mix": "--neg blocker --neg-frac 0.35 --mix 0.5"}
ok = {}
for v, args in variants.items():
    ok[v] = sh(f"python exploration/p4/e5_finetune.py --split xc_India --out {M}/{v}_xc_India --n 400000 {args}",
               f"ft_{v}", cwd=REPO, env=ft) == 0
models = ["intfloat/multilingual-e5-small", inp("ft_xc_India"), inp("ft2_xc_India")] + [f"{M}/{v}_xc_India" for v in variants if ok[v]]
sh(f"python exploration/p4/e5_ft_eval.py --pairs {pairs} --label xc_India@0.1 --models {' '.join(m for m in models if m)} "
   f"--out {W}/e5_ft_eval_out.md --json {W}/e5_ft_eval_t16a.json", "eval", cwd=REPO, env=off)
for v in variants:
    if ok[v]:
        sh(f"python src/run.py run -c configs/exp/p4/T06_e5g.yaml --task T16 --split xc_India@0.1 "
           f"--set features.1.params.model={M}/{v}_xc_India --note 'e5 fine-tuned {v} on US pairs'", f"harness_{v}", cwd=pkg, env=off)
sh("free -g; du -sh /kaggle/working/models/*", "done")
