"""T11/T15 (p3) job runner, called by the Kaggle launchers kaggle_t11{a,b}/ber_p3_t11.py inside the cloned repo.
Usage: python exploration/p3/kaggle/t11_jobs.py <job> <out_dir>. Every step runs even if an earlier one failed;
steps + return codes + wall times go to <out_dir>/summary.txt, logs to <out_dir>/logs/.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CFG = "code/business_entity_resolution/configs/exp/p3"
DENSE = "python exploration/p3/blk_dense.py --dense --qfrac 0.1 --caps 50,70,100"
JOBS = {
    # cheap channel first: baseline (T09_tr blocker) + empty-address channel, misses of T11_e, then a harness sanity
    "a": [("dense", f"{DENSE} --config {CFG}/T11_e.yaml --config {CFG}/T11_base.yaml --misses 60 --out {{out}}/blk_dense_out.md"),
          ("smoke", "cd code/business_entity_resolution && python src/run.py run -c configs/exp/p3/T11_e.yaml --smoke --task T11"),
          ("val", "cd code/business_entity_resolution && python src/run.py run -c configs/exp/p3/T11_e.yaml --task T11")],
    # the expensive name-gram + address channel, alone and with the empty-address channel
    # variants of the empty-address channel (deeper reverse list; short addresses too)
    "c": [("dense", f"{DENSE} --config {CFG}/T11_e10.yaml --config {CFG}/T11_e4.yaml --config {CFG}/T11_e.yaml --out {{out}}/blk_dense_out.md")],
    # T15 organizer-rule study: per-S1 caps x reverse-flood caps for the empty-address channel variants (provenance
    # from p3_compound_rc / p3_chargram), then a pipeline smoke of the new blocker
    "d": [("dense", f"python exploration/p3/blk_dense.py --dense --qfrac 0.1 --caps 20,30,40,50,70 --rev-caps 5,10,20,50 "
                    f"--config {CFG}/T15_e.yaml --config {CFG}/T15_e10.yaml --config {CFG}/T15_e4.yaml --config {CFG}/T15_er.yaml "
                    f"--config {CFG}/T15_base.yaml --out {{out}}/blk_dense_out.md"),
          ("smoke", "cd code/business_entity_resolution && python src/run.py run -c configs/exp/p3/T15_e.yaml --smoke --task T15")],
    "b": [("dense", f"{DENSE} --config {CFG}/T11_eca.yaml --config {CFG}/T11_ca.yaml --misses 60 --out {{out}}/blk_dense_out.md")],
}


def main():
    job, out = sys.argv[1], Path(sys.argv[2])
    (out / "logs").mkdir(parents=True, exist_ok=True)
    summary = []
    for name, cmd in JOBS[job]:
        cmd = cmd.format(out=out)
        t0 = time.time()
        with open(out / "logs" / f"{name}.log", "a", encoding="utf-8") as f:
            f.write(f"$ {cmd}\n")
            f.flush()
            rc = subprocess.run(cmd, shell=True, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode
        summary.append(f"{time.strftime('%H:%M:%S')} rc={rc} {time.time() - t0:7.0f}s  {name}: {cmd}")
        print(summary[-1], flush=True)
        (out / "summary.txt").write_text("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
