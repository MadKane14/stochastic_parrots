#!/usr/bin/env bash
# T12 (p4) — fine-tuned e5 judged by dense LOCO on phoenix (lead). Needs w/p4 >= this commit (only new files:
# exploration/p4/e5_finetune.py) and requirements-gpu.txt. One GPU, nice, BER_THREADS <= 12.
#
# Leakage-safe protocol (why the India fold only): a model fine-tuned on US pairs has never seen an India pair, so in
# dense LOCO the "predict India" fold (GBDT trained on US, features from the US-only model) is an honest unseen-country
# test — read metrics.by_country.India of the --loco JSON and compare with the same number of T10_e5 (pretrained).
# The US fold of that run is NOT clean (the embedder saw US pairs) — ignore it.
set -euo pipefail
export BER_THREADS=${BER_THREADS:-8} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} TOKENIZERS_PARALLELISM=true
export HF_HOME=${HF_HOME:-$BER_ARTIFACTS/hf}
C=code/business_entity_resolution/configs/exp/p1
MD=$BER_ARTIFACTS/models

# 1) Fine-tune on US pairs only (xc_India train side), ~12 min on a T4 (Kaggle: 682 s train + ~3 min data prep).
nice -n 10 python exploration/p4/e5_finetune.py --split xc_India --out $MD/ft_xc_India --n 400000
# 2) Dense LOCO of T10_e5 with the fine-tuned model (new p1_e5 memmaps for this model: embeds train once, ~1 h on an
#    A4000 at p1's 15k names/s) vs T10_e5 pretrained (run it too if its LOCO does not exist yet).
nice -n 10 python exploration/p1/dense_cv.py -c $C/T10_e5.yaml --loco --tag ftUS --set features.2.params.model=$MD/ft_xc_India
# nice -n 10 python exploration/p1/dense_cv.py -c $C/T10_e5.yaml --loco        # pretrained baseline, if missing
#
# 3) If the India fold improves: final model = fine-tune on ALL train pairs (for test), same recipe:
#    python exploration/p4/e5_finetune.py --split full --out $MD/ft_full --n 400000
#    and use `features: - {name: p1_e5, params: {fields: [name, full], model: <abs path>/ft_full}}` in the test config.
#    (Hash-fold dense CV with ft_full would be leaky: the embedder saw ~20% of the S1 entities' pairs.)
