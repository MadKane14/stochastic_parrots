"""Kaggle launcher for p3 T18 job 'b' (see exploration/p3/kaggle/t18_jobs.py). Inputs (private): ber-student-resource
(organizers' data), ber-p3-code (p3.bundle = git bundle of w/p3, so outputs carry a clean commit hash), and the
output of kernel ber-p3-t18emb (p1_e5 matrices of the train India group)."""
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
env = dict(os.environ, BER_MEMBER="p3", BER_DATA_DIR=data_dir, BER_ARTIFACTS="/tmp/ber_artifacts",
           BER_RESULTS_DIR=str(W / "results"), BER_THREADS="4", PYTHONUNBUFFERED="1")
subprocess.run(f"python exploration/p3/kaggle/t18_jobs.py b {W}", shell=True, cwd=repo, env=env)
