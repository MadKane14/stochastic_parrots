#!/usr/bin/env bash
# p1 (Sun 27 Sep): test run of T27 (T18 + p1_mnoise − core-name features) → submission #9 candidate, then label-free checks.
set -u
. "$1"; cd "$(dirname "$0")/../.."
L=/scratch2/madhur.kankane/ber_artifacts/logs/T27_test.log
S=/scratch2/madhur.kankane/ber_artifacts/submissions
cd code/business_entity_resolution
systemd-run --user --scope -p MemoryMax=59G nice -n 10 python -W ignore src/run.py test -c configs/exp/p1/T27_mnoise_nocore.yaml --task T27 > $L 2>&1
echo "EXIT=$?" >> $L
cd ../..
D=$(ls -dt $S/*_p1_T27_*_test_* | head -1)
ln -sfn "$D" $S/sub9_T27
BER_THREADS=4 nice -n 10 python exploration/p1/diff_subs.py $S/sub8_T18 $S/sub9_T27 France 12 > $L.diff 2>&1
BER_THREADS=4 nice -n 10 python -W ignore exploration/p1/sibling_probe.py $S/sub9_T27 > $L.sibling 2>&1
BER_THREADS=4 nice -n 10 python exploration/p1/num_conflict.py $S/sub9_T27 20260927-074356_densecv_T18_2dd8b3 0.8 > $L.numconf 2>&1
echo DONE >> $L
