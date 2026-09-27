#!/usr/bin/env bash
# Overnight queue (p1, Sat → Sun), phoenix rules: one heavy CPU job at a time, MemoryMax=59G, nice 10, BER_THREADS=12.
# 1) dense CV of T16 (final stack)  2) test run of T16 ONLY if its dense CV beats T12_noise2 (0.98405)  3) dense LOCO T16.
set -u
. /tmp/claude-1907800205/-home2-madhur-kankane-stochastic-parrots-stochastic-parrots/85eccf0b-153f-41b0-9160-bc018970db06/scratchpad/env.sh
M=/home2/madhur.kankane/stochastic_parrots/stochastic_parrots
L=$BER_ARTIFACTS/logs/overnight.log
C=code/business_entity_resolution/configs/exp/p1
RUN="systemd-run --user --scope -p MemoryMax=59G nice -n 10"
cd $M
echo "START $(date)" >> $L
$RUN python exploration/p1/dense_cv.py -c $C/T16.yaml >> $L 2>&1
F=$(ls -t exploration/p1/dense_cv/*_densecv_T16_*.json 2>/dev/null | grep -v loco | head -1)
SCORE=$(python -c "import json;print(json.load(open('$F'))['metrics']['f05'])" 2>/dev/null || echo 0)
echo "DONE1 dense CV T16 = $SCORE ($F) $(date)" >> $L
if python -c "import sys; sys.exit(0 if float('$SCORE') > 0.98405 else 1)"; then
  echo "T16 beats T12_noise2 -> test run" >> $L
  (cd code/business_entity_resolution && $RUN python src/run.py test -c configs/exp/p1/T16.yaml --task T16 >> $L 2>&1)
  echo "DONE2 test T16 $(date)" >> $L
else
  echo "T16 does NOT beat 0.98405 -> no test run" >> $L
fi
$RUN python exploration/p1/dense_cv.py -c $C/T16.yaml --loco --tag loco >> $L 2>&1
echo "DONE3 LOCO T16 $(date)" >> $L
