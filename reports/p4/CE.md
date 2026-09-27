# Cross-encoder reranker (top-50 push, account owner's request) — p4 (Sun 06:20)

Status: done (proxy). Verdict: **REJECTED — no gain on top of the current stack; do not spend phoenix time on it.**

## Setup (all Kaggle, one kernel `ber-p4-ce`, commit `25fd69f`)
- Stack baseline: `T13_sub4` on val_v1@0.1 (harness run `20260926-234842_p4_T22_T13_sub4_val_v1@0.1_fcfd00`); its held-out target candidates after
  pruning = 171,867 pairs (151,389 true) with the stack probability.
- Cross-encoder (`exploration/p4/ce_train.py`): intfloat/multilingual-e5-small (MIT, 117.65M) + 1-logit head,
  "name | address" of both records, max 128 tokens; trained on the SAME split's train side: 500,000 pairs from its
  `T13_sub4` blocker candidates (5.95M, 607,904 true) = 200k true + 200k top-3 non-matches per S1 (hard) + 100k random
  non-matches; BCE, lr 3e-5, batch 64, 1 epoch (7,812 steps, 427 pairs/s on a T4, 25 min). Final loss ~0.005–0.03.
- Proxy (`exploration/p4/ce_eval.py`): same pairs, same decider (p1_assign, exclusive right), frozen metric.
  **Primary comparison fixed before running: stack vs mean(stack, CE) at the config threshold 0.7.**

## Result (`exploration/p4/CE/ce_eval_out.md`)
| variant | thr | F0.5 [CI95] | pair P / R | India / US |
|---|---|---|---|---|
| stack | 0.7 | **0.9961** [0.99572, 0.99642] | 0.99908 / 0.98999 | 0.99586 / 0.99621 |
| mean(stack, CE) — PRIMARY | 0.7 | 0.9951 [0.99475, 0.99551] | 0.9993 / 0.98644 | 0.99551 / 0.99488 |
| CE alone | 0.7 | 0.9936 [0.99314, 0.994] | 0.99828 / 0.98418 | 0.9942 / 0.99318 |
| 0.7·stack + 0.3·CE (secondary) | 0.7 | 0.9962 [0.99589, 0.99655] | 0.99939 / 0.98947 | 0.99614 / 0.99626 |
(threshold 0.5 rows in the md file: same picture.) The primary blend is **worse** (CIs apart): averaging pulls true
pairs below 0.7 (recall 0.990 → 0.986). The best secondary blend only ties. CE scoring speed: 2,737 pairs/s (T4).

## Why (and limits)
- On this split the stack is already at 0.996; the CE, trained once for 25 min, is weaker alone (0.9936) and adds
  no independent signal the GBDT does not already have (it has E5 cosines, string sims, number/IDF features).
- Limits: val_v1@0.1 is less dense than test (D9) — hard sibling negatives are rarer here, so a CE could matter
  more at test density; but there is no evidence for it and no time to build a dense test before the freeze.
- Not tried (no time): CE as a GBDT feature with out-of-fold scores, longer training, larger model.

GPU used: ~0.9 h (Kaggle T4).
