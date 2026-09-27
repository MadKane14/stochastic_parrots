"""Kaggle batch kernel for p4 T32: reproducibility of the packaged src/e5_finetune.py (1 GPU).
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



# T20 (round 5, p2's task; run on p4's Kaggle at the account owner's request so p2 does not run it locally).
# 1) ft_full = v1 recipe on ALL train pairs; 2) dense_cv_group --group India: T13_sub4 (pretrained) vs T20_ft (ft_full).
# P100 fallback (T4 queue was stuck): PyPI's torch 2.8.0 is the CUDA 12.8 build, which dropped Pascal (sm_60) ->
# same torch version from the CUDA 12.6 index on a P100.
sh("nvidia-smi -L | grep -q P100 && pip install -q --force-reinstall torch==2.8.0 "
   "--index-url https://download.pytorch.org/whl/cu126 && pip uninstall -y -q torchvision torchaudio; "
   "python -c \"import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.ones(2, device='cuda').sum().item())\"",
   "gpu_check", env=env)

ft = dict(off, HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0")
R = Path("/tmp/t32")
def run(tag, cmd):
    # poll GPU memory once a second while the run is going; record the peak
    pol = subprocess.Popen(f"while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0; sleep 1; done > {W}/gpumem_{tag}.txt",
                           shell=True)
    rc = sh(cmd, f"ft_{tag}", cwd=pkg, env=ft)
    pol.terminate()
    vals = [int(x) for x in open(W / f"gpumem_{tag}.txt").read().split() if x.strip().isdigit()]
    summary.append(f"peak GPU memory {tag}: {max(vals) if vals else 'n/a'} MiB (nvidia-smi, whole device)")
    (W / "summary.txt").write_text("\n".join(summary) + "\n")
    return rc
args = "--split full --n 20000 --seed 0"
for tag in ("A", "B"):                                   # 1) packaged script, as is, twice
    run(tag, f"python src/e5_finetune.py {args} --out {R}/{tag}")
sh(f"python exploration/p4/ft_repro.py compare {R}/A {R}/B --out {W}/t32_AB.json", "compare_AB", cwd=REPO, env=off)
for tag in ("C", "D"):                                   # 2) proposed fix (deterministic torch), twice
    run(tag, f"python {REPO}/exploration/p4/ft_repro.py det -- {args} --out {R}/{tag}")
sh(f"python exploration/p4/ft_repro.py compare {R}/C {R}/D --out {W}/t32_CD.json", "compare_CD", cwd=REPO, env=off)
sh(f"python exploration/p4/ft_repro.py compare {R}/A {R}/C --out {W}/t32_AC.json", "compare_AC", cwd=REPO, env=off)
sh(f"cp {R}/A/p4_finetune_stats.json {W}/stats_A.json; cp {R}/C/p4_finetune_stats.json {W}/stats_C.json; nvidia-smi -L", "done", env=off)
