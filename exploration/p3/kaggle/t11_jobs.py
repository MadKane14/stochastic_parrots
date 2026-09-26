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
    # T15 speed: the empty-address channel with lower df caps (T15_e at max_df 5000 took 1629 s on full train)
    "e": [("dense", f"python exploration/p3/blk_dense.py --dense --qfrac 0.1 --caps 20,30,40,50 --rev-caps 10,20 "
                    f"--config {CFG}/T15_e_df2000.yaml --config {CFG}/T15_e_df1000.yaml --config {CFG}/T15_base.yaml "
                    f"--out {{out}}/blk_dense_out.md")],
    # T15 name-token x address-word channel on top of the empty-address channel (k 10 and 5)
    "f": [("dense", f"python exploration/p3/blk_dense.py --dense --qfrac 0.1 --caps 20,30,40,50 --rev-caps 10,20 "
                    f"--config {CFG}/T15_ena.yaml --config {CFG}/T15_ena5.yaml --misses 60 --out {{out}}/blk_dense_out.md"),
          ("smoke", "cd code/business_entity_resolution && python src/run.py run -c configs/exp/p3/T15_ena.yaml --smoke --task T15")],
    # T15 final: the shipped config end to end (real rev caps + cap 40 -> real blocking times), two next-step
    # variants, then a pipeline smoke of the prune stage fed with every blocker's columns (no E5 on Kaggle)
    "g": [("dense", f"python exploration/p3/blk_dense.py --dense --qfrac 0.1 --caps 20,30 "
                    f"--config {CFG}/T15_final_t09.yaml --config {CFG}/T15_final_e10.yaml "
                    f"--config {CFG}/T15_final_na1000.yaml --misses 60 --out {{out}}/blk_dense_out.md"),
          ("smoke", "cd code/business_entity_resolution && python src/run.py run -c configs/exp/p3/T15_final_t09_prune.yaml --smoke --task T15")],
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
