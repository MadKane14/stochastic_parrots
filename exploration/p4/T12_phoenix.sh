#!/usr/bin/env bash
# T12 (p4) on phoenix — for the lead: dense CV needs 41-61 GB (p1's runs), p4's compute (Kaggle) has 31 GB max.
# Run from the repo root on a commit that contains w/p4's p4_e5.py + configs/exp/p4/T12_*.yaml, env `ber` with
# requirements-gpu.txt (torch 2.8.0, transformers 4.57.1). One GPU, nice, BER_THREADS <= 12 (docs/servers/phoenix.md).
set -euo pipefail
export BER_THREADS=${BER_THREADS:-8} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=${HF_HOME:-$BER_ARTIFACTS/hf} TOKENIZERS_PARALLELISM=true
C=code/business_entity_resolution/configs/exp

# 1) Embed every unique train + test text once (name, name+address) into $BER_ARTIFACTS/emb (resumable; cached per
#    text hash). Kaggle T4: see exploration/p4/e5_embed_all_{name,full}.json for counts/time; an A4000 is faster.
nice -n 10 python exploration/p4/e5_embed_all.py --parts train,test --fields name,full \
    --out-json exploration/p4/e5_embed_all_phoenix.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 2) Dense CV + LOCO with p4_e5 (new feature cache: T08_sub's features + 10 E5 columns; all E5 texts cached above)
nice -n 10 python exploration/p1/dense_cv.py -c $C/p4/T12_e5.yaml
nice -n 10 python exploration/p1/dense_cv.py -c $C/p4/T12_e5.yaml --loco
# 3) Baseline LOCO of T08_sub (hash-fold dense CV already exists: 20260926-063458_densecv_T08_sub_724a04 = 0.97885)
nice -n 10 python exploration/p1/dense_cv.py -c $C/p1/T08_sub.yaml --loco
# 4) Ablations (same feature cache as T12_e5 -> scorer refit only)
nice -n 10 python exploration/p1/dense_cv.py -c $C/p4/T12_e5_nogrp.yaml
nice -n 10 python exploration/p1/dense_cv.py -c $C/p4/T12_e5_s1grp.yaml
# Outputs: exploration/p1/dense_cv/<run>.json — p4 copies them into exploration/p4/ for the T12 report.
