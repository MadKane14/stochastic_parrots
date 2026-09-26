"""Kaggle CPU launcher for p3 T15 job 'e' (4 cores, 30 GB, no swap). Inputs (private datasets):
ber-student-resource (organizers' data) and ber-p3-code (p3.bundle = git bundle of w/p3, so results carry a clean
commit hash). Clones the bundle, installs the pinned env, runs exploration/p3/kaggle/t11_jobs.py e."""
import glob
import os
import subprocess
from pathlib import Path

W = Path("/kaggle/working")


def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    if not hits:
        raise SystemExit(f"input not found: {pattern}")
    return hits[0]


repo = "/tmp/repo"
data_dir = str(Path(find("train_source1.tsv")).parents[1])
subprocess.run(f"git clone -b w/p3 {find('p3.bundle')} {repo} && git -C {repo} log -1 --oneline > {W}/commit.txt", shell=True, check=True)
subprocess.run(f"pip install -q -r {repo}/code/business_entity_resolution/requirements.txt > {W}/pip.log 2>&1", shell=True)
env = dict(os.environ, BER_MEMBER="p3", BER_DATA_DIR=data_dir, BER_ARTIFACTS="/tmp/ber_artifacts",
           BER_RESULTS_DIR=str(W / "results"), BER_THREADS="4", PYTHONUNBUFFERED="1")
subprocess.run(f"python exploration/p3/kaggle/t11_jobs.py e {W}", shell=True, cwd=repo, env=env)
subprocess.run(f"free -g; du -sh /tmp/ber_artifacts/* > {W}/du.txt 2>&1", shell=True)
