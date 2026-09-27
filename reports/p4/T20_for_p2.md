# T20 (p2's task, run by p4 on p4's Kaggle) — fine-tuned vs pretrained E5 on the full stack, India group (Sun 06:55)

Status: done (India group, 50% S1 hash sample — the full India group runs out of memory on Kaggle).
Verdict: **SUPPORTED (small)** — E5 fine-tuned on all train pairs (`ft_full`, p4 v1 recipe) beats pretrained E5 in
`T13_sub4`: F0.5 **0.98328 vs 0.98285 (+0.00043)**, both folds (+0.00046 / +0.00039), same at threshold 0.8
(0.98332 vs 0.98289). Blocking and candidates identical → the gain is the matcher's.
**Caveat (lead):** `ft_full` saw these India pairs in fine-tuning; p4 T16 found naive fine-tuning ≈ cross-fitted
(val_v1@0.1: 0.9904 vs 0.9907), so the leakage is not expected to inflate this much, but it is not zero.

## Evidence — `exploration/p4/dense_cv_group_q.py` (p2's dense_cv_group.py on a 50% S1 hash sample, all S2/S3 kept)
Kaggle kernel `ber-p4-t20q` (T4), commit `42045e2` (clean, includes p1's memory-lean dense_cv), same 441,310 India S1,
18,736,320 pairs (identical for both). JSONs: `exploration/p4/T20/`.
| config | run | F0.5 | fold 0 / 1 | singleton acc | pair P / R | blocking recall | oracle | thr 0.8 | peak GB |
|---|---|---|---|---|---|---|---|---|---|
| T13_sub4 (pretrained E5) | `20260926-235448_densecv_India_T13_sub4_6d9cea_t20pre_q0.5` | 0.98285 | 0.9829 / 0.9828 | 0.9807 | 0.99509 / 0.96131 | 0.9792 | 0.99262 | 0.98289 | 28.5 |
| T20_ft (= T13_sub4, `p1_e5.model = ft_full`) | `20260927-011935_densecv_India_T20_ft_908733_t20ft_q0.5` | **0.98328** | 0.98336 / 0.98319 | 0.9812 | 0.9952 / 0.96233 | 0.9792 | 0.99262 | **0.98332** | 26.5 |
- `ft_full`: `exploration/p4/e5_finetune.py --split full --n 400000` (v1 recipe), trained in kernel `ber-p4-t20`
  (957 s on a T4); checkpoint kept as that kernel's output and in `ber-p4-t20q`'s output.
- The 50% sample keeps all S2/S3 records (forward density unchanged); reverse candidates are more generous
  (42 cands/S1 vs ~33) — identical for both configs, so only the difference is evidence.
- The full India group was killed (exit 137) for both configs on Kaggle at ~28–29 GB (kernel `ber-p4-t20`,
  `exploration/p4/T20/logs_t20_fullgroup/`) — before p1's memory-lean dense_cv (`05c820a`).

## Consistency with other evidence
- p4 T16 (US-only model → India, unseen country): harness xc_India@0.1 0.9471 → 0.9553 with fine-tuning.
- This run (in-country, dense, full stack): +0.0004. Fine-tuning helps most where the country is unseen — France.
- The lead's final stack T17 already uses `ft_full` (dense CV 0.98722); this result supports keeping it.
