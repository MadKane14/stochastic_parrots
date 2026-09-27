#!/usr/bin/env bash
# Clean-checkout reproduction of the final submission (p1, Sun 27 Sep).
# Waits for the FINAL test run (PID $1) to exit, clones the repo at HEAD into scratch, and re-runs
#   python src/run.py test -c configs/best.yaml --no-cache
# in the pinned ber-gpu env (requirements.txt). The fine-tuned encoder and its embedding memmaps are reused from
# $BER_ARTIFACTS (re-embedding would take ~1.5 h GPU and 35 GB); every other stage is recomputed from the clean code.
# Then compares the two submissions pair by pair (exploration/p1/compare_subs.py).
set -u
WAIT_PID=${1:?pid of the running FINAL test}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CLONE=/scratch2/madhur.kankane/repro_clone
LOG=/scratch2/madhur.kankane/ber_artifacts/logs/repro_final.log
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
FINAL_DIR=$(ls -dt /scratch2/madhur.kankane/ber_artifacts/submissions/*_FINAL_* | head -1)
echo "$(date +%T) FINAL run dir: $FINAL_DIR" > "$LOG"
rm -rf "$CLONE" && git clone -q "$REPO" "$CLONE" && git -C "$CLONE" log --oneline -1 >> "$LOG"
cd "$CLONE/code/business_entity_resolution" || exit 1
export BER_MEMBER=repro BER_THREADS=12 BER_DATA_DIR=/scratch2/madhur.kankane/ber_data/dataset
export BER_ARTIFACTS=/scratch2/madhur.kankane/ber_artifacts TMPDIR=/scratch2/madhur.kankane/tmp
export HF_HOME=/scratch2/madhur.kankane/.cache/huggingface CUDA_VISIBLE_DEVICES=""
export PATH=/scratch2/madhur.kankane/.conda/envs/ber-gpu/bin:$PATH
python -c "import sys, torch, lightgbm; print(sys.executable, torch.__version__, lightgbm.__version__)" >> "$LOG" 2>&1
systemd-run --user --scope -p MemoryMax=59G nice -n 10 \
    python -W ignore src/run.py test -c configs/best.yaml --task REPRO --no-cache >> "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
REPRO_DIR=$(ls -dt /scratch2/madhur.kankane/ber_artifacts/submissions/*_REPRO_* | head -1)
PATH=/scratch2/madhur.kankane/.conda/envs/ber/bin:$PATH nice -n 10 \
    python "$REPO/exploration/p1/compare_subs.py" "$FINAL_DIR" "$REPRO_DIR" >> "$LOG" 2>&1
