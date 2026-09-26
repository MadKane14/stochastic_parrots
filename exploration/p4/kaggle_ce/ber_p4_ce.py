"""Kaggle batch kernel for p4 top-50 push: cross-encoder reranker proxy on val_v1@0.1 (2x T4 / P100).
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



# T20 (round 5, p2's task; run on p4's Kaggle at the account owner's request so p2 does not run it locally).
# 1) ft_full = v1 recipe on ALL train pairs; 2) dense_cv_group --group India: T13_sub4 (pretrained) vs T20_ft (ft_full).
# P100 fallback (T4 queue was stuck): PyPI's torch 2.8.0 is the CUDA 12.8 build, which dropped Pascal (sm_60) ->
# same torch version from the CUDA 12.6 index on a P100.
sh("nvidia-smi -L | grep -q P100 && pip install -q --force-reinstall torch==2.8.0 "
   "--index-url https://download.pytorch.org/whl/cu126 && pip uninstall -y -q torchvision torchaudio; "
   "python -c \"import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.ones(2, device='cuda').sum().item())\"",
   "gpu_check", env=env)

M = W / "models"
ft = dict(off, HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0")
C = "code/business_entity_resolution/configs/exp/p1/T13_sub4.yaml"
# 1) the current best stack on val_v1@0.1 -> held-out target candidates with the stack probability (scored_pairs)
sh(f"python src/run.py run -c configs/exp/p1/T13_sub4.yaml --task T22 --split val_v1@0.1 --note 'CE proxy baseline'",
   "stack", cwd=pkg, env=off)
runs = sorted(glob.glob("/tmp/ber_artifacts/runs/*T13_sub4_val_v1@0.1_*/scored_pairs.parquet"))
# 2) cross-encoder on the SAME split's train side (T13_sub4's normalizer + blockers make the candidates)
ok = sh(f"python exploration/p4/ce_train.py --split val_v1@0.1 --config {C} --out {M}/ce_val", "ce_train", cwd=REPO, env=ft) == 0
# 3) proxy: stack vs mean(stack, CE) at the config threshold (primary, fixed in advance)
if ok and runs:
    sh(f"python exploration/p4/ce_eval.py --pairs {runs[-1]} --split val_v1@0.1 --model {M}/ce_val "
       f"--out {W}/ce_eval_out.md --json {W}/ce_eval.json", "ce_eval", cwd=REPO, env=off)
sh("free -g; du -sh /kaggle/working/models/* 2>/dev/null", "done")
