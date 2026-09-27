#!/usr/bin/env bash
# p1 queue (Sun 27 Sep): after the running dense_cv (PID $1) exits → T25 in-country dense CV → T25 dense LOCO.
set -u
ENV=$2
while kill -0 "$1" 2>/dev/null; do sleep 20; done
. "$ENV"; cd "$(dirname "$0")/../.."
L=/scratch2/madhur.kankane/ber_artifacts/logs
C=code/business_entity_resolution/configs/exp/p1/T25_nocore.yaml
systemd-run --user --scope -p MemoryMax=59G nice -n 10 python -W ignore exploration/p1/dense_cv.py -c $C > $L/T25_dense.log 2>&1
echo "EXIT=$?" >> $L/T25_dense.log
systemd-run --user --scope -p MemoryMax=59G nice -n 10 python -W ignore exploration/p1/dense_cv.py -c $C --loco --tag loco > $L/T25_loco.log 2>&1
echo "EXIT=$?" >> $L/T25_loco.log
