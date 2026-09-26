# STATE — single source of truth for "where are we" (planner drafts, lead commits)

_Last updated: Sat 26 Sep 17:30 IST (round 4 — all branches merged; T10_e5 dense CV 0.9831; submission #3 running)_

## Team
| handle | person | machine / GPU | branch | current focus |
|---|---|---|---|---|
| p1 | MadKane (lead) | phoenix (16c/32t, 128 GB, 3× A4000; shared) | main | T03 matcher v1 + decision rule + submission #1 |
| p2 | _fill_ | _fill_ | w/p2 | T04 cross-script / noisy-name normalization |
| p3 | _fill_ | _fill_ | w/p3 | T05 blocking recall ≥ 0.98 (char-gram channel) |
| p4 | _fill_ | _fill_ | w/p4 | T06 multilingual embedding features (GPU) |

## Timeline (IST)
- Syncs: 09:30 · 18:30 · 23:30 (overnight launch). Deadline **Sun 27 Sep 23:59**.
- Sun 12:00 feature freeze · Sun 18:00 code freeze · Sun 21:00 zip + methodology submitted.
- Portal: max **5 submissions/day** (Fri/Sat/Sun). Keep ≥ 1 spare each day.

## Eval
- Lock: `planning/eval_lock.json` committed (`095191c`), eval hash `57af2f140244d7b2`. Runs before the lock are not evidence.
  Check your data copy (`sha256sum $BER_DATA_DIR/train/*.tsv`) against:
  `70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037` train_ground_truth.tsv ·
  `591af0e1dfeb65cab71ea6ee8cb69df00f92d6ba6fa79e05746c938775d14973` train_source1.tsv ·
  `6336c1a055eec79cf8a6d99fdc8d32a2e4d9dc2662e00963cb35d66b89ed09ed` train_source2.tsv ·
  `67da22f5151898ff3006febd836c1a159e97ae95efa7257a5aff4fda685e58e9` train_source3.tsv
- Main split `val_v1` (20% of S1 entities); dev iteration on `val_v1@0.1`. Unseen-country proxies: `xc_India(@0.1)`, `xc_US(@0.1)`.
- **Density caveat (D5, I-08):** val splits contain 5–50× fewer distractors per S1 than test → val F0.5 and val blocking
  recall are optimistic. Blocking is judged at full density with `exploration/p1/blk_eval.py`.
- Data (line counts): train ≈ 2.21M S1 / 5.03M S2 / 5.29M S3; test ≈ 1.73M S1 / 4.89M S2 / 5.08M S3 (France = 15% of test S1).
- Public LB context (Fri ~15:00): top 0.984, 5th ≈ 0.980.

## Current best pipeline
- **Decision metrics (D9):** dense CV (out-of-fold over the full train set, test-like density) and dense LOCO
  (leave-one-country-out, unseen-country proxy) — `exploration/p1/dense_cv.py [--loco]`. val_v1@0.1 = plumbing only.
- Candidate for submission #2: `configs/exp/p1/T08_sub.yaml` = basic normalize → p1_compound (k=30, k_rev=3,
  max_df=1000) → p1_match + p1_extra (`idf_group: country`) → p1_lgbm (uncapped) → p1_assign (one owner, thr 0.7).

| config | dense CV F0.5 | India / US | dense LOCO | evidence |
|---|---|---|---|---|
| T03_sub (= submission #1) | 0.96961 | 0.9575 / 0.9777 | 0.90504 | `exploration/p1/dense_cv/20260926-022106_densecv_T03_sub_d4f44c.json`, `…031622_…_loco.json` |
| T07_lgb (uncapped LightGBM) | 0.97152 | 0.9605 / 0.9789 | 0.90493 | `…030301_densecv_T07_lgb_e2e7f4.json`, `…040322_…_loco.json` |
| T08_g (group IDF + romanized skeletons + number distance) | 0.97679 | 0.9723 / 0.9798 | 0.92322 | `…T08_g_fafd84.json`, `…T08_g_fafd84_loco.json` |
| T08_sub (T08_g + uncapped + thr 0.7) | 0.97885 | 0.9747 / 0.9816 | — | `…063458_densecv_T08_sub_724a04.json` |
| T09_tr (T08_sub + p2_translit D; blocking recall 0.970 → 0.979) | 0.98279 | 0.9826 / 0.9829 | — | `…143459_densecv_T09_tr_490034.json` |
| **T10_e5** (T09_tr + pretrained E5 `p1_e5` name/full) | **0.98313** | 0.9829 / 0.9833 | — | `…161049_densecv_T10_e5_08f49f.json` |

Full-density blocking (best.yaml blocker): pair recall 0.970, oracle F0.5 0.989 — blocking misses are the largest
remaining dense-CV loss (0.0129 of 0.030 for T03_sub, `exploration/p1/dense_analysis.md`).

## Leaderboard submissions
| # | day | time | test run_id | config/commit | dense CV | public LB F0.5 | note |
|---|---|---|---|---|---|---|---|
| 1 | Fri | ~00:40 | 20260925-233233_p1_T03_T03_sub_test_d4f44c | T03_sub / 94bb5d8 | 0.96961 | **0.9487** | val_v1@0.1 said 0.9917; top of LB 0.9884 |
| 2 | Sat | ~09:00 | 20260926-063526_p1_T09_T08_sub_test_724a04 | T08_sub / c1b9f99 | 0.97885 | **0.969** | validator PASS (--check-ids); expected-F on test 0.981 vs 0.961 for #1 (`exploration/p1/expected_f_out.md`) |
| 3 | – | not uploaded | 20260926-171638_p1_T13_T10_e5_test_08f49f | T10_e5 / 4e6d871 | 0.98313 | _pending_ | validator PASS (--check-ids); T09 (+p2_translit) + pretrained E5 |
| 4 | Sat | ~22:00 | 20260926-201615_p1_T13_T13_sub4_test_fb141a | T13_sub4 / 1d9e21b | 0.98405 (T12_noise2) | **0.978** | + p1_noise + learned pruning: ~6.5 cands/S1 (was ~35); validator PASS |
| 5 | Sat | _to upload_ | 20260926-221844_p1_T13_T13_sub5_ft_test_19770e | T13_sub5_ft / 04cf1fc | – | _pending_ | #4 + E5 fine-tuned on all train pairs (p4 v1); PASS |
| 6 | Sat | _to upload_ | blend_sub4_sub5 (exploration/p1/blend_runs.py) | mean of #4 and #5 probabilities | – | _pending_ | same candidates; corr 0.998; PASS |

## Known issues / decisions
- D1: Handles p1–p4; worker branches `w/p1..w/p4`; only the lead pushes to `main`.
- D2: Numbers in planning come only from results JSONs (see PLANNER.md §1); blocking numbers from committed
  `exploration/*/blk_eval_out.md` sections.
- D3: data lives at `<repo>/data/` (organizers' student_resource contents, git-ignored) or `BER_DATA_DIR`; organizer docs in `docs/`.
- D4: `best.yaml` blocker = `p1_compound` (brute-force `tfidf_knn` cannot run on full data).
- D5: val splits are less dense than test (I-08) → full-density blocking eval is the reference for blocking; expect LB < val.
- D6: `p1_compound` is **transductive** (IDF/df caps from the records being blocked, i.e. test records at submission; no
  labels) — same as the baseline `tfidf_knn`. Provisionally accepted; open question Q7 in QUESTIONS.md, lead to confirm.
- D7: requirements pin `scipy==1.15.2` (1.15.3 is not on conda-forge); conda-forge install documented in requirements.txt.
- D8: the full `val_v1` split does not fit in 64 GB with the current pipeline (all train pairs are featurized) → p1 T03
  adds fit-sampling (sample of train S1, all S2/S3 kept). Until then: `val_v1@0.1`, `xc_*@0.1`.
- D9: dense CV + dense LOCO are the decision metrics for matcher/decider/features (exploration/p1/dense_cv.py; lead-approved
  exception to "numbers only from results JSONs": the frozen harness cannot evaluate at test density). Each run writes a
  JSON with commit/config; cite the run name. Needs ~55–60 GB RAM.
- D10: p1_match/p1_extra are transductive (IDF from the processed records, now within country-string groups), like the blocker.
- D11: stage-2 stacking and expected-F0.5 set selection were tested and dropped (≤ +0.0001, I-21).
- D12: GitHub auth on phoenix failed at 06:10 (`git fetch`: "Repository not found / Authentication failed") — lead to re-login before pushing.
- D13: **Organizer update (Sat): smaller candidate sets per S1 rank higher in the final evaluation** (`docs/organizer_updates.md`).
  Current: ~35 cands/S1 on test (max ~380 from the reverse direction). Every blocking/pruning result must report cands/S1
  (mean, p95, max) next to recall; a learned pruning stage before the final matcher is the planned way to shrink it.
- Open organizer questions: see QUESTIONS.md.
- video-slides.pdf in the claude.ai project is image-only; the planner could not read it.
