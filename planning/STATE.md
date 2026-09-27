# STATE — single source of truth for "where are we" (planner drafts, lead commits)

_Last updated: Sun 27 Sep 16:30 IST (round 8 — LB #8 0.981, #9 **0.983**; #10 = min-blend ready; T29 reproduces #9)_

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
| T10_e5 (T09_tr + pretrained E5 `p1_e5` name/full) | 0.98313 | 0.9829 / 0.9833 | — | `…161049_densecv_T10_e5_08f49f.json` |
| T12_noise2 (T10_e5 + p1_noise, unsup noise-ness) = LB #4 0.978 | 0.98405 | 0.9844 / 0.9838 | — | `…201525_densecv_T12_noise2_c3f279.json` |
| T17 (p3 T15 blocking 0.979→0.989 recall + noise + pruning + E5 fine-tuned + p1_ambig) | 0.98722 | 0.9861 / 0.9880 | 0.93443 | `…013724_densecv_T17_530df3.json`, `…060231_…_loco.json` |
| **T18** (T17 + normalizer p4_tcanon) = **FINAL** (`configs/best.yaml`, thr 0.8) | **0.98752** (thr 0.8: 0.98746) | 0.9862 / 0.9884 | **0.93538** (thr 0.8: 0.93813) | `…074356_densecv_T18_2dd8b3.json`, `…081325_…_loco.json` (LOCO run flagged dirty: best.yaml edit in the tree, T18.yaml unchanged) |

Full-density blocking (best.yaml blocker): pair recall 0.970, oracle F0.5 0.989 — blocking misses are the largest
remaining dense-CV loss (0.0129 of 0.030 for T03_sub, `exploration/p1/dense_analysis.md`).

## Leaderboard submissions
| # | day | time | test run_id | config/commit | dense CV | public LB F0.5 | note |
|---|---|---|---|---|---|---|---|
| 1 | Fri | ~00:40 | 20260925-233233_p1_T03_T03_sub_test_d4f44c | T03_sub / 94bb5d8 | 0.96961 | **0.9487** | val_v1@0.1 said 0.9917; top of LB 0.9884 |
| 2 | Sat | ~09:00 | 20260926-063526_p1_T09_T08_sub_test_724a04 | T08_sub / c1b9f99 | 0.97885 | **0.969** | validator PASS (--check-ids); expected-F on test 0.981 vs 0.961 for #1 (`exploration/p1/expected_f_out.md`) |
| 3 | – | not uploaded | 20260926-171638_p1_T13_T10_e5_test_08f49f | T10_e5 / 4e6d871 | 0.98313 | _pending_ | validator PASS (--check-ids); T09 (+p2_translit) + pretrained E5 |
| 4 | Sat | ~22:00 | 20260926-201615_p1_T13_T13_sub4_test_fb141a | T13_sub4 / 1d9e21b | 0.98405 (T12_noise2) | **0.978** | + p1_noise + learned pruning: ~6.5 cands/S1 (was ~35); validator PASS |
| 5 | – | not uploaded | 20260926-221844_p1_T13_T13_sub5_ft_test_19770e | T13_sub5_ft / 04cf1fc | – | _pending_ | #4 + E5 fine-tuned on all train pairs (p4 v1); PASS |
| 6 | Sat | ~23:50 | blend_sub4_sub5 (exploration/p1/blend_runs.py) | mean of #4 and #5 probabilities | – | **0.979** | same candidates; corr 0.998; PASS |
| 7 | Sun | _to upload_ | 20260927-033731_p1_T16_T17_test_530df3 | T17 / cb81d3a | 0.98722 | _pending_ | p3 blocking (recall 0.979→0.989) + noise + pruning + ft E5 + p1_ambig; PASS (--check-ids); expected-F 0.9876 |
| 7b | Sun | ~08:30 | T17_thr080 (blend_runs.py, same scores, threshold 0.8) | T17 / cb81d3a | 0.98717 (LOCO 0.937 vs 0.934) | **0.979** | no LB gain over #6 despite +0.003 dense CV → D16 |
| 8 | Sun | ~15:45 | 20260927-081609_p1_FINAL_best_test_a7adec (= submissions/sub8_T18) | best.yaml (T18 + thr 0.8) / af1a1be | 0.98746 @0.8 | **0.981** | official validator PASS; vs #7b: France +10.1k/−6.0k pairs, US/India ±5-7k |
| – | – | not uploaded | 20260927-130314_p1_T28_T28_mined_only_test_ea4e7e (sub9_T28) | T28 / 9489112 | 0.98732 @0.8 | – | PASS, but its 38k added France pairs have 39% house-number conflicts (kept 0.9%) → suffix-sibling false merges (D19) |
| 9 | Sun | ~15:45 | blend_T18_T28 (= submissions/sub9_blend; blend_runs.py mean of #8 and T28 scores, thr 0.8) | T18 + T28 | **0.98748 @0.8** (OOF mean) | **0.983** | official validator PASS (--check-ids); vs #8: France −15.5k/+5.7k pairs (added 4.5% conflicts; France partial-name same-address assigned 65.6% → 58.1%), US/India ±2-3k |
| 10 | Sun | ~16:40 | sub10_min (blend_runs.py --combine min of #8 and T28 scores, thr 0.8: both members must pass) | T18 + T28 | 0.98721 @0.8 (OOF min) | **0.983186** | official validator PASS; sha256 82cab7bb…; vs #9 removes 1.34% of France pairs vs 0.18-0.19% US/India (`exploration/p1/test_variant_churn_out.txt`) |

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
- D14: decision threshold 0.7 → **0.8** for the final (validation-based, not LB): T17 in-country dense CV is flat 0.7–0.8
  (0.98722 vs 0.98717) while the clean unseen-country fold (India-trained model → US) rises 0.926 → 0.931 (0.9: 0.939);
  the model is overconfident on an unseen country (France). `exploration/p1/dense_decide_out.md` (T17 runs).
  T17 re-decided at 0.8 from saved scores: `submissions/T17_thr080` (validator PASS).
- D15 (Sun 08:15): FINAL = T18 at threshold 0.8 = `configs/best.yaml` (dense CV 0.98752 > T17 0.98722, LOCO 0.93538 >
  T17 0.93443; both folds up). Feature freeze: no further stack changes; Round-2 best frozen as `best_r2.yaml`.
- D16 (Sun 09:45): the LB stall is the unseen country, not US/India. Label-free evidence (exploration/p1/, d7795bc): US/India
  test predictions reproduce dense-CV in-country statistics (house-number conflicts 6.18% vs 6.07% US, 2.16% vs 2.15% India;
  empty-address assignment ≈ 54-60% both); test has 2x the distractors/S1 of train but they are easy (pruned cands/S1 similar).
  France (15% of test S1) churns 3-5x more than US/India under train-fitted changes (ft E5 6.8% vs 1.4% of S1; thr 0.8 3.8%
  vs 1.1%); with US/India at dense CV the LB implies France ≈ 0.94 (#4) → ≈ 0.93 (#7b). LOCO's failure mode (shifted-number
  near-copies) is absent in France predictions (0.9% conflicts) → LOCO misled D14 for France. Next: T24 self-training
  (`p1_selftrain`), judged by dense LOCO adapted-vs-base on the same folds + in-country not worse; the LB is the only France
  signal, used for ≤ 3 discrete choices (public LB SE ≈ 0.0002 → negligible selection optimism).
- D17 (Sun 12:15): T24 self-training rejected (LOCO 0.93333 < base 0.93813 on the same folds: recall +, confident false
  merges reinforced). T25 (drop core-name features) = T18 in-country (0.98746 @0.8). T27 (T25 + p1_mnoise, label-free
  per-country noise-word rates) 0.98739 @0.8 (fold spread 0.0003) → test run = #9.
- D17b (Sun 13:05): T27's matcher kept using p1_noise's label-learned word rates (8.7% of gain vs 1.4% for p1_mnoise,
  `exploration/p1/feat_importance.py`), which are the prior for every France word → T28 drops them: pair-level slice check
  logloss 0.00724 vs 0.00728 (T27) / 0.00736 (T18); dense CV 0.98732 @0.8 (T18 0.98746, T27 0.98739) → gate passed, T28 = #9.
  In-country (labelled) the same-address sibling classes are already ~99.9% precise for T18 and T27 alike
  (`exploration/p1/sibling_oof.py`): the gain, if any, is France-only by construction.
- D19 (Sun 14:55): T28 alone is not uploaded — label-free: the France pairs it adds vs #8 have 39% house-number conflicts
  (kept pairs 0.9%, #8's additions over #7b 10%): mined rates make French sibling suffixes ("participations" .97,
  "holding" .89) look like noise. The probability mean of T18 and T28 keeps T28's removals of same-address word-swap
  siblings and rejects its number-shifted additions (added France pairs 4.5% conflicts); in-country OOF mean 0.98748 @0.8
  (T18 0.98746, T28 0.98732; precision 0.99833 best). → #9 = blend. `exploration/p1/{diff_numconf,oof_blend}.py`.
- D20 (Sun 16:30): #8 0.981 / #9 0.983. With US/India at dense CV (0.9872 at test composition; D16), France ≈ 0.946 (#8) → 0.959 (#9),
  ±0.003 from LB rounding; France perfect would cap the LB at ≈ 0.989, so 0.99 needs US/India > dense CV as well. Both LB steps
  came from label-free / target-adaptive changes (tcanon, mined noise) and #9's step was mostly France removals where the
  label-learned (T18) and label-free (T28) members disagree. In-country OOF cannot separate the variants (mean 0.98748, 0.25/0.75
  0.98743, guard 0.98738, mean3 with T27 0.98748, min 0.98721 @0.8; `exploration/p1/oof_variants_out.txt`). min@0.8 concentrates the
  same move on France (−1.34% of France pairs vs −0.18/0.19% US/India, removed pairs have 3.6% number conflicts; mean@0.9 would cost
  in-country −0.001 for −2.0% France) → #10 = min@0.8. #11 only if #10 ≥ #9 + 0.001 (one more step, min@0.85); else final = #9.
  Decision rule D18 extends to #10/#11 (ties within 0.0003 → higher dense CV). If a min variant wins, `weighted_mean` gets a
  `combine: min` option after T29 finishes and one test run confirms the packaged config reproduces the file.
- D21 (Sun 16:45): T29 (one config) reproduces #9: identical candidate_pairs.tsv, 2,356 of 5.8M matched pairs differ (members
  retrained in one run). T30 = T29 + `combine: min` (reproduces #10) test run launched 16:39. Remaining in-country loss of the #9
  OOF (0.01252): empty-address entities 14% of S1 / 55% of loss, blocking misses 19%, FN in candidates 16%, FP 7%, singletons 3%;
  ceilings: perfect blocking +0.0038, zero FP +0.0015 (`exploration/p1/loss_breakdown_out.txt`) → 0.99 is out of reach today.
  Private LB = remaining portion of the same test file (US/India/France only); a non-random split with more France would hurt
  us (France ≈ 0.96 vs 0.987) → France stays the lever for public and private alike.
- D22 (Sun 17:00): #10 (min) 0.983186 ≈ #9 (0.983, exact value pending) → France precision beyond #9 is neutral. Final
  under D18 = #9 unless #9 < 0.982886. `configs/best.yaml` = flattened T29 (#9 ensemble; the T18 final frozen as best_t18.yaml,
  all other configs hash-identical); provisional zip built from the T29 run dir. #11 = the packaged run's own output of the
  winner (T29, or T30 if #10 wins) so the zip's matching_results.tsv is the last uploaded file.
- D23 (Sun 19:10): LB #9 0.983146, #10 0.983186 → tie (D18) → final = #9 ensemble = `configs/best.yaml` (T29). Two last checks,
  both rejected on the #9-equivalent in-country OOF (0.98748 @0.8): (1) T31 bagging over training data (member A with seed 1 =
  a different 15M-pair sample): 0.25A+0.5B+0.25A1 0.98750, mean3 0.98752 — below the +0.0002 gate fixed before the run
  (`exploration/p1/oof_bag_out.txt`); (2) p3's cross-copy rule (unassigned record joins the owner of its exact-name
  siblings): every variant loses (−0.00008 … −0.00311), added pairs only 18–63% correct — same-name copies are mostly other
  businesses (`exploration/p1/sibling_rule_out.txt`). Last upload = T29's own output so the zip's matching_results.tsv is an
  uploaded file regenerated by the packaged code.
- D18 (written before #8/#9 LB are known): FINAL = highest public LB among #7b / #8 / #9; if the top two are within
  0.0003, take the one with the higher dense CV (blend 0.98748 > T18 0.98746 > T17 0.98717). best.yaml / README / methodology are then set
  to that config and the zip is built from its run dir.
- Open organizer questions: see QUESTIONS.md.
- video-slides.pdf in the claude.ai project is image-only; the planner could not read it.
