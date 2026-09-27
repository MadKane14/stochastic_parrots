#!/usr/bin/env bash
# p1 (Sun 27 Sep): gate for submission #9. After T28's dense CV: F0.5@0.8 >= 0.9871 (T18 0.98746 minus fold noise)
# → test run T28, else → test run T27 (T27 dense CV 0.98739 already passed).
set -u
ENV=$1
cd "$(dirname "$0")/../.."
L=/scratch2/madhur.kankane/ber_artifacts/logs/T28_dense.log
until grep -q "EXIT=" $L; do sleep 20; done
F=$(ls -t exploration/p1/dense_cv/*T28_mined_only*.json 2>/dev/null | head -1)
V=$(python3 -c "import json,sys; print(json.load(open('$F'))['metrics']['threshold_curve']['0.8'])" 2>/dev/null || echo 0)
PASS=$(python3 -c "print(1 if float('$V') >= 0.9871 else 0)")
echo "$(date +%T) T28 dense CV @0.8 = $V (file $F) → pass=$PASS" > /scratch2/madhur.kankane/ber_artifacts/logs/gate_t28.txt
if [ "$PASS" = 1 ]; then bash exploration/p1/run_test_t28.sh "$ENV"; else bash exploration/p1/run_test_t27.sh "$ENV"; fi
