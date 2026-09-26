#!/usr/bin/env bash
# Overnight queue v2 (p1, Sun 00:20). Phoenix rules: one heavy CPU job at a time, MemoryMax=59G, nice 10.
# wait for the running T16 dense CV (pid $1) -> T17 dense CV (only p1_ambig new) -> test run of the better of
# T16/T17 only if > 0.98405 -> dense LOCO of that config.
set -u
. /tmp/claude-1907800205/-home2-madhur-kankane-stochastic-parrots-stochastic-parrots/85eccf0b-153f-41b0-9160-bc018970db06/scratchpad/env.sh
M=/home2/madhur.kankane/stochastic_parrots/stochastic_parrots
L=$BER_ARTIFACTS/logs/overnight.log
C=code/business_entity_resolution/configs/exp/p1
RUN="systemd-run --user --scope -p MemoryMax=59G nice -n 10"
cd $M
while kill -0 "$1" 2>/dev/null; do sleep 30; done
score() { F=$(ls -t exploration/p1/dense_cv/*_densecv_$1_*.json 2>/dev/null | grep -v loco | head -1); \
          python -c "import json;print(json.load(open('$F'))['metrics']['f05'])" 2>/dev/null || echo 0; }
S16=$(score T16); echo "T16 dense CV = $S16 $(date)" >> $L
$RUN python exploration/p1/dense_cv.py -c $C/T17.yaml >> $L 2>&1
S17=$(score T17); echo "T17 dense CV = $S17 $(date)" >> $L
BEST=$(python -c "print('T17' if float('$S17') > float('$S16') else 'T16')")
BS=$(python -c "print(max(float('$S16'), float('$S17')))")
if python -c "import sys; sys.exit(0 if $BS > 0.98405 else 1)"; then
  echo "best $BEST ($BS) beats 0.98405 -> test run $(date)" >> $L
  (cd code/business_entity_resolution && $RUN python src/run.py test -c configs/exp/p1/$BEST.yaml --task T16 >> $L 2>&1)
  echo "DONE test $BEST $(date)" >> $L
else
  echo "neither T16 ($S16) nor T17 ($S17) beats 0.98405 -> no test run" >> $L
fi
$RUN python exploration/p1/dense_cv.py -c $C/$BEST.yaml --loco --tag loco >> $L 2>&1
echo "DONE LOCO $BEST $(date)" >> $L
