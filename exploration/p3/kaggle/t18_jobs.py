"""T18 (p3) Kaggle job runner, called by the launchers exploration/p3/kaggle_t18{emb,a,b}/ber_p3_t18.py from the cloned
repo root (stdlib only: it installs the environment itself).

    emb (GPU 2x T4): pinned GPU env, p1_e5 matrices (fields name, full) of the train India group, one field per GPU,
                     copied to <out>/emb_mm/ (the output is mounted by the a/b kernels via kernel_sources)
    a / b (CPU):     link those matrices into $BER_ARTIFACTS/emb_mm, verify they are the files p1_e5 will look up,
                     then exploration/p3/dense_cv_group_prune.py (= p2's dense_cv_group.py unchanged + out-of-fold
                     pruning) on T13_sub4 (a, baseline = submission #4) or T15_sub5 (b, + p3 T15_final blocking)
Usage: python exploration/p3/kaggle/t18_jobs.py <job> <out_dir>
"""
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / "code/business_entity_resolution"
CFG_BASE = "code/business_entity_resolution/configs/exp/p1/T13_sub4.yaml"
CFG = {"a": CFG_BASE, "b": "code/business_entity_resolution/configs/exp/p3/T15_sub5.yaml",
       # fallback without p1_e5 (no GPU needed): same pair minus the E5 feature set
       "c": "code/business_entity_resolution/configs/exp/p3/T18_base_noe5.yaml",
       "d": "code/business_entity_resolution/configs/exp/p3/T18_p3_noe5.yaml"}
GROUP = "India"
MODEL_DIR = "intfloat__multilingual-e5-small_ml64"      # p1_e5._root() for the configs' p1_e5 spec


def main():
    job, out = sys.argv[1], Path(sys.argv[2])
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    summary = []

    def sh(cmd, log, env=None, cwd=REPO):
        t0 = time.time()
        with open(logs / f"{log}.log", "a", encoding="utf-8") as f:
            f.write(f"$ {cmd}\n")
            f.flush()
            rc = subprocess.run(cmd, shell=True, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        summary.append(f"{time.strftime('%H:%M:%S')} rc={rc} {time.time() - t0:7.0f}s  {log}: {cmd}")
        print(summary[-1], flush=True)
        (out / "summary.txt").write_text("\n".join(summary) + "\n")
        return rc

    art = os.environ["BER_ARTIFACTS"]
    sh("nproc; free -g; nvidia-smi -L 2>/dev/null; df -h /tmp /kaggle/working; python --version; git log -1 --oneline", "setup")
    if job == "emb":
        sh(f"pip install -q -r {PKG}/requirements-gpu.txt", "pip")
        sh("pip uninstall -y -q torchvision torchaudio", "pip")      # image's torchvision breaks transformers (p4 T12)
        env = dict(os.environ, HF_HOME="/tmp/hf", TOKENIZERS_PARALLELISM="true")
        sh("python -c \"import sys; sys.path.insert(0, 'code/business_entity_resolution/src'); import os; "
           "from ber.data import load_part; load_part(os.environ['BER_DATA_DIR'], 'train', "
           "cache_dir=os.environ['BER_ARTIFACTS'] + '/cache')\"", "rawcache", env=env)
        sh("python -c \"from transformers import AutoModel, AutoTokenizer; m='intfloat/multilingual-e5-small'; "
           "AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)\"", "hf", env=env)
        off = dict(env, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        procs = []
        for gpu, field in ((0, "name"), (1, "full")):
            f = open(logs / f"embed_{field}.log", "w", encoding="utf-8")
            procs.append((field, f, time.time(), subprocess.Popen(
                f"python exploration/p3/e5_group.py -c {CFG_BASE} --group {GROUP} --field {field}", shell=True,
                cwd=REPO, env=dict(off, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=f, stderr=subprocess.STDOUT)))
        for field, f, t0, p in procs:
            rc = p.wait()
            f.close()
            summary.append(f"{time.strftime('%H:%M:%S')} rc={rc} {time.time() - t0:7.0f}s  embed_{field} (GPU, parallel)")
            print(summary[-1], flush=True)
        (out / "summary.txt").write_text("\n".join(summary) + "\n")
        sh(f"mkdir -p {out}/emb_mm && cp -r {art}/emb_mm/{MODEL_DIR} {out}/emb_mm/ && "
           f"rm -f {out}/emb_mm/{MODEL_DIR}/*.tmp.npy {out}/emb_mm/{MODEL_DIR}/*.rows.npy; ls -la {out}/emb_mm/{MODEL_DIR}", "copy")
        return
    # ---- a / b: dense CV on CPU with the GPU job's matrices ----
    sh(f"pip install -q -r {PKG}/requirements.txt", "pip")
    if job in ("c", "d"):                          # no embeddings needed
        return run_dense(job, sh, out)
    hits = glob.glob(f"/kaggle/input/**/emb_mm/{MODEL_DIR}", recursive=True)
    if not hits:
        sh("ls -R /kaggle/input | head -50", "link")
        raise SystemExit("embedding matrices not found under /kaggle/input")
    os.makedirs(f"{art}/emb_mm", exist_ok=True)
    os.symlink(hits[0], f"{art}/emb_mm/{MODEL_DIR}")
    rc = sh("python -c \"import sys, importlib.util, os; sys.path.insert(0, 'code/business_entity_resolution/src'); "
            "s = importlib.util.spec_from_file_location('g', 'exploration/p2/dense_cv_group.py'); g = importlib.util.module_from_spec(s); "
            "s.loader.exec_module(g); ds = g.group_loader('" + GROUP + "')(os.environ['BER_DATA_DIR'], 'train', "
            "cache_dir=os.environ['BER_ARTIFACTS'] + '/cache'); fp = ds.fingerprint(); "
            "fs = [os.environ['BER_ARTIFACTS'] + '/emb_mm/" + MODEL_DIR + "/' + fp + '_' + f + '.npy' for f in ('name', 'full')]; "
            "print(fp, [(f, os.path.exists(f)) for f in fs]); sys.exit(0 if all(map(os.path.exists, fs)) else 3)\"", "link")
    if rc != 0:
        raise SystemExit("embedding files do not match the India group's fingerprint: not starting (would embed on CPU)")
    run_dense(job, sh, out)


def run_dense(job, sh, out):
    art = os.environ["BER_ARTIFACTS"]
    sh(f"python exploration/p3/dense_cv_group_prune.py -c {CFG[job]} --group {GROUP}", "dense")
    for pat in ("exploration/p2/dense_cv/*.json", "exploration/p2/dense_cv_out.md", "exploration/p3/dense_cv/*.json",
                "exploration/p3/dense_cv_out.md"):
        for p in glob.glob(str(REPO / pat)):
            dst = out / Path(p).relative_to(REPO)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dst)
    sh(f"free -g; du -sh {art}/* 2>/dev/null", "done")


if __name__ == "__main__":
    main()
