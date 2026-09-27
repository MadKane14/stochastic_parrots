# T20 is running on p4's Kaggle — p2: please do NOT run T20 on your laptop (Sat 22:40)

At the account owner's request, p4 runs **p2's round-5 task T20** (fine-tuned vs pretrained E5 on the full stack,
India group) on p4's Kaggle, so it is not duplicated:
- kernel `anishtoshniwal006/ber-p4-t20` (GPU), code = `w/p4` (includes main `bdfcf1f`), runner
  `exploration/p4/kaggle_t20/ber_p4_t20.py`
- step 1: `ft_full` = `exploration/p4/e5_finetune.py --split full --n 400000` (v1 recipe, ~15–20 min on a T4);
  the checkpoint is saved as kernel output for the lead
- step 2: `exploration/p2/dense_cv_group.py --group India` on `configs/exp/p1/T13_sub4.yaml` (pretrained) and
  `configs/exp/p4/T20_ft.yaml` (= T13_sub4 with `p1_e5.model: ${BER_ARTIFACTS}/models/ft_full`, nothing else changed)
- results → `exploration/p4/T20/` + `reports/p4/T20_for_p2.md` on `w/p4`, with the caveat the lead asked for
  (ft_full saw these India pairs; naive ≈ cross-fitted per `reports/p4/T16.md`)

Also on p4's Kaggle tonight: p2's earlier queued job (T14_full India dense CV, p2 commit `a12b762`, kernel
`ber-p4-p2full`) and p4's T19 (`ber-p4-t19`). Kaggle runs 2 GPU sessions at a time, so T20 may queue.
