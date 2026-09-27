#!/usr/bin/env bash
# p1 (Sun 27 Sep): T29 (= #9 blend as one pipeline) test run, then pairwise comparison with submissions/sub9_blend.
set -u
. "$1"; cd "$(dirname "$0")/../.."
L=/scratch2/madhur.kankane/ber_artifacts/logs/T29_test.log
S=/scratch2/madhur.kankane/ber_artifacts/submissions
cd code/business_entity_resolution
systemd-run --user --scope -p MemoryMax=59G nice -n 10 python -W ignore src/run.py test -c configs/exp/p1/T29_blend.yaml --task T29 > $L 2>&1
echo "EXIT=$?" >> $L
cd ../..
D=$(ls -dt $S/*_p1_T29_*_test_* | head -1)
ln -sfn "$D" $S/final_T29
BER_THREADS=4 nice -n 10 python exploration/p1/compare_subs.py $S/sub9_blend $S/final_T29 >> $L 2>&1
echo DONE >> $L
