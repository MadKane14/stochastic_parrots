# T04 evaluations run for p2 on p4's Kaggle — p4

> **Update Sat 13:50 — superseded as primary evidence.** p2 ran the same pre-registered evaluations on their own Kaggle
> (`w/p2` commit `0d63813`, `reports/p2/T04.md`). Every number below matches theirs exactly. p2's run_ids are the
> primary evidence; this report is an **independent replication** on a second machine/account and should not be
> counted twice. p2's variant B finished (pair recall 0.9869), where mine ran out of memory.

Status: done (variant B's blocking eval: out of memory)
Verdict: SUPPORTED for p2's T04 hypothesis. Variant A raises full-density pair recall by **+0.0064** (target ≥ +0.005)
at an unchanged candidate count, and raises harness F0.5 beyond CI95 on val_v1@0.1 and xc_India@0.1.

**Why this exists:** p2 was compute-blocked (their report on `w/p2`: needs ≥ 32 GB, has 15 GB). At the account
owner's request, p4 ran p2's pre-registered evaluations on p4's Kaggle, **unchanged, at p2's commit `645e5b9`**
(cloned from a git bundle). p4 changed no code, config or threshold of p2's. The only additions are the runner
(`exploration/p4/kaggle_p2_{blk,harness}/`) and output paths in p4's folders. The harness names results after
the machine owner, so the results JSONs are under `results/p4/` with task `T04`. Their `git.branch` is `w/p2` and
their `git.commit` is `645e5b9`. p2 owns the interpretation, and this report only relays the numbers.

Machine: Kaggle CPU kernel, 4 cores, ~30 GB RAM, `BER_THREADS=4`, Python 3.12 (as in `reports/p4/T01.md`).

## 1. Full-density blocking (p2's `exploration/p2/blk_eval_p2.py`, `--qfrac 0.1`)
Output: `exploration/p4/p2_T04/blk_eval_out.md` (+ `blk_eval_json/`, `logs_blk/`). One run per variant, all on the
same machine with the identical command, and base = `configs/best.yaml`. 762,339 GT pairs from 220,163 S1
queries against all 10.32M S2/S3 records.

| label | pair recall | Δ vs base | oracle F0.5 | cands mean / p95 / max | India | US | non-Latin name | web name | empty addr | letter run |
|---|---|---|---|---|---|---|---|---|---|---|
| base (best.yaml) | 0.9805 | – | 0.9932 | 142.7 / 301 / 1126 | 0.9731 | 0.9856 | 0.9183 | 0.9502 | 0.8809 | 0.9937 |
| **T04_a** (all steps, replace) | **0.9869** | **+0.0064** | 0.9956 | 143.1 / 304 / 1133 | 0.9849 | 0.9881 | 0.9775 | 0.9849 | 0.8832 | 0.9946 |
| T04_d (A, no script folding) | 0.9869 | +0.0064 | 0.9956 | 143.1 / 304 / 1133 | 0.9850 | 0.9881 | 0.9778 | 0.9849 | 0.8833 | 0.9946 |
| T04_c (learned maps only) | 0.9844 | +0.0039 | 0.9948 | 143.0 / 304 / 1126 | 0.9827 | 0.9856 | 0.9779 | 0.9500 | 0.8781 | 0.9937 |
| T04_b (A, append mode) | killed (exit 137, out of memory) | | | | | | | | | |

Subgroup sizes (true pairs): non-Latin name 54,800 · web name 39,208 · empty address 33,557 · letter run 26,294.

- **Cross-script is fixed at the blocking stage:** recall on pairs with a non-Latin S2/S3 name goes 0.9183 → 0.9775
  (C alone does this). The web-name steps add the rest: web-name recall 0.9502 → 0.9849 (A vs C).
  Empty-address pairs (0.88) are now the weakest subgroup, and no variant moves them.
- **A = D** on every metric. By p2's pre-declared tie rule the simpler variant, D, is preferred.
- **Candidates are unchanged** (+0.4 per S1), as p2 predicted: the gain is pure recall.
- **Base is not comparable with p1's 0.9699 / 33 cands:** those were `--qfrac 1`. Under `--qfrac 0.1` the
  reverse top-3 lands on 10% of S1, so mean candidates are ~4× higher and recall is optimistic (TASKS.md Round 2,
  item 4). Only the Δ within this table is evidence, and finalists still need `--qfrac 1`, which does not fit
  on Kaggle.
- **Memory:** peak RSS 27.4–28.3 GB for base/A/C/D. B has more tokens per record, so it died after the India
  group's reverse top-3 step. Running B needs a machine with ≥ 40 GB, e.g. phoenix.
- Time per variant: normalize 6–12 min + blocking 10–15 min on 4 cores.

## 2. Harness (val_v1@0.1, xc_India@0.1)
All 6 runs: commit `645e5b9`, branch `w/p2`, clean, eval hash OK, split fingerprint OK. Logs are in
`exploration/p4/p2_T04/logs_harness/`.

| role | run_id | split | F0.5 [CI95] | India / US | pair P / R | oracle F0.5 / blk recall |
|---|---|---|---|---|---|---|
| base (best.yaml) | 20260925-234104_p4_T04_best_val_v1@0.1_9b37f6 | val_v1@0.1 | 0.9843 [0.9836, 0.98496] | 0.97779 / 0.98863 | 0.99313 / 0.96847 | 0.99858 / 0.99583 |
| T04_a | 20260925-230432_p4_T04_T04_a_val_v1@0.1_7474a1 | val_v1@0.1 | **0.9882** [0.98764, 0.98884] | 0.98739 / 0.98883 | 0.99420 / 0.97725 | 0.99960 / 0.99848 |
| T04_d | 20260925-232249_p4_T04_T04_d_val_v1@0.1_6e0f3d | val_v1@0.1 | **0.9879** [0.98732, 0.98853] | 0.98708 / 0.98846 | 0.99405 / 0.97658 | 0.99960 / 0.99848 |
| base (best.yaml) | 20260925-235039_p4_T04_best_xc_India@0.1_9b37f6 | xc_India@0.1 | 0.9239 [0.92278, 0.92509] | 0.92393 / – | 0.96847 / 0.87285 | 0.99421 / 0.98371 |
| T04_a | 20260925-231341_p4_T04_T04_a_xc_India@0.1_7474a1 | xc_India@0.1 | **0.9279** [0.92684, 0.92895] | 0.92789 / – | 0.97349 / 0.87196 | 0.99465 / 0.98498 |
| T04_d | 20260925-233153_p4_T04_T04_d_xc_India@0.1_6e0f3d | xc_India@0.1 | **0.9279** [0.92684, 0.92895] | 0.92789 / – | 0.97349 / 0.87196 | 0.99465 / 0.98498 |

- val_v1@0.1: +0.0039 (A) / +0.0036 (D) F0.5, beyond CI95, almost all from India (+0.0096).
- xc_India@0.1: +0.0040, beyond CI95. The xc_India train side has no India pairs, so no map can be learned for
  India. The gain here must come from the script-agnostic steps (web names, initials), which is the part that
  transfers to France. A and D give identical results because folding has nothing to fold without India data.
- The base runs on p2's commit reproduce p4's T06 base runs exactly (0.9843 / 0.9239). The best.yaml pipeline is
  deterministic across these commits and machines, so p2's and p4's numbers can be compared directly.

## 3. Combined view with p4's T06 (same splits, same machine type)
| split | base | p2 T04_a (normalizer) | p4 T06_e5g (E5 features) |
|---|---|---|---|
| val_v1@0.1 | 0.9843 | 0.9882 | 0.9876 |
| xc_India@0.1 | 0.9239 | 0.9279 | 0.9471 |

The two changes act at different stages: the normalizer improves blocking and string features, while E5 adds
learned similarity to the matcher. On an unseen country, E5 carries most of the gain. **Proposed next step:** a
combined config (`p2_translit` D + `p4_e5` group features) on T03's matcher, vs each alone, on the same commit.

## Reproduce
- Bundle `ber-p4-code` v2 = `w/p4` @ `3debe87` + `w/p2` @ `645e5b9`.
  `kaggle kernels push -p exploration/p4/kaggle_p2_blk` and `… -p exploration/p4/kaggle_p2_harness` (CPU, ~1 h each,
  run in parallel).
