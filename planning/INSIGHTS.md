# INSIGHTS — append-only log (planner drafts, lead commits)

Format: `I-NN [confidence] statement — evidence: run_ids / exploration script — (round, date)`
Superseded insights are struck through with a pointer, never deleted.

## Prior knowledge (from the problem statement — not yet verified on data)
- I-01 [given] Metric is macro F0.5 per S1 entity incl. singletons; a single false match on a singleton costs a full 1.0 on that entity → the decision rule must be conservative, and the singleton rate sets how conservative. — problem statement
- I-02 [given] S1 is deduplicated → an S2/S3 record should belong to at most one S1 entity (enables "exclusive right" assignment). To verify in EDA (T03 Q3). — problem statement
- I-03 [given] Test contains France, unseen in train → components must generalize across countries; `xc_*` splits are our proxy. — problem statement
- I-04 [given] candidate_pairs.tsv is audited for blocking quality (recall ceiling, reduction ratio) → keep candidate sets reasonably tight, not "everything". — problem statement

## Round 2 — Fri 25 Sep (real data; p1 T02)
- I-05 [high] True pairs share the exact `country` string 100% of the time; no S2/S3 id is in more than one GT list (confirms I-02) → grouping by country string and one-owner assignment are safe. — `exploration/p1/eda_out.md` Q3/Q4 (R2)
- I-06 [high] Singletons are 5.6% of S1; non-singletons have 3.67 matches on average (max 11) → recall matters as much as precision; the no-match gate still matters. — `eda_out.md` Q2 (R2)
- I-07 [high] Names repeat across different entities: identical normalized names are true matches only 34% of the time; a name-only IDF top-40 blocker reaches just 0.44 pair recall at full density. Name alone never decides; address (numbers) must. — `eda_out.md` Q8b; `exploration/p1/blk_eval_out.md` run 19:16 (R2)
- I-08 [high] Validation splits are 5–50× less dense than test (val side keeps only its own 20% / 2% of S2/S3 records) → val blocking recall and F0.5 are optimistic. Same blocker at k=10: 0.9945 recall on val_v1@0.1 vs 0.963 at full density. — run `20260925-205410_p1_T02_T02_cmp_val_v1@0.1_bc4a15`; `blk_eval_out.md` run 20:51 (R2)
- I-09 [high] Compound keys (name-word pairs, address bigrams, number×word, number×name) + reverse top-3 per S2/S3 record: 0.970 pair recall / 0.989 oracle F0.5 at 33 cands/S1 at full density, vs 0.926 at 45 cands for single-token IDF channels. The reverse direction adds most at small k (k=5: 0.860 → 0.959). — `blk_eval_out.md` runs 19:16, 20:51 (R2)
- I-10 [medium] "Number variant" keys (drop first/last digit) lower recall (0.9613 vs 0.9645 at k=30, qfrac 0.1): extra keys dilute the cosine more than they recover perturbed numbers. — `blk_eval_out.md` runs 20:32/20:36 (R2)
- I-11 [high] The sparse-product blocker is memory-bandwidth-bound: 24 chunks take 11.5 s / 5.1 s / 5.9 s with 1 / 4 / 12 workers → use 4 workers; speed comes from lower df caps, not threads. max_df 5000 → 1000 costs 0.005 recall at k=30 and is ~5× faster. — `exploration/p1/blk_profile_out.md`; `blk_eval_out.md` (R2)
- I-12 [medium] Of the 3% true pairs the best.yaml blocker misses at full density: 29% have a non-Latin S2/S3 name, 22% an empty S2/S3 address, only 30% share an address number; many are web-style names (`supremepackaging`, `heartcommittee com`) or typo'd names with no address. — `exploration/p1/blk_misses_out.md` (R2)
- I-13 [medium] Data noise patterns seen in misses: leading zeros (`001954`), digits glued to letters (`26th`, `5ahaya`, `23805c`), dropped/changed digits (`3829`→`829`), numbers spelled out (`twentieth street`), state names vs codes (`virginia`/`va`). — `blk_misses_out.md`, `exploration/p1/blk_eval.py --misses` (R2)
- I-14 [low] Baseline string-sim HGB on the compound blocker scores 0.9840–0.9850 on val_v1@0.1 for k=10/30/50 (ties); more candidates don't help at val density but should at test density (I-08). — runs `…bc4a15`, `…35fb75`, `…7e3121` (R2)

## Round 3 — Sat 26 Sep (LB feedback + dense CV; p1 overnight)
- I-15 [high] Submission #1 scored **0.9487** on the public LB vs 0.9917 on val_v1@0.1. The model's own expected F0.5 on test (simulated from its probabilities) was 0.961, and on val_v1@0.1 the same estimate matched the truth (0.990 vs 0.992) → the gap is test density, not a bug. — run `20260925-233233_p1_T03_T03_sub_test_d4f44c`; LB (R3)
- I-16 [high] **Dense CV** (out-of-fold over the full train set = test-like density; `exploration/p1/dense_cv.py`) scores submission #1's config at 0.9696 (India 0.957, US 0.978), frozen-metric cross-check identical; it is our decision metric for matcher/decider work from now on. — `exploration/p1/dense_cv/20260926-022106_densecv_T03_sub_d4f44c.json` (R3)
- I-17 [high] Test has ~24% more S2/S3 records per S1 than train (5.8 vs 4.7, every country) while predicted matches/S1 are unchanged → more unmatched distractors in test (≈40% vs 26% of right records); unmatched distractors are already 70% of dense-CV false positives. — test/train line counts (eda_out.md Q1), `exploration/p1/dense_analysis.md` (R3)
- I-18 [high] Dense-CV loss decomposition (0.0304 total): blocking misses 0.0129, FN inside candidates 0.0077, FP on non-singletons 0.0070, singletons 0.0027. Non-Latin S2/S3 names drive many FPs (same address, different company written in Tamil/Hindi) and FNs. — `exploration/p1/dense_analysis.md` (R3)
- I-19 [high] **Unseen-country penalty at density is large:** leave-one-country-out dense CV (train on one country, predict the other) 0.905 vs 0.970 in-country. More boosting rounds do not change it (0.9049); dropping absolute IDF masses (0.900), raw string sims (0.893) or group sizes (0.904) does not help. — `exploration/p1/dense_cv/*loco*.json` (R3)
- I-20 [high] **IDF within equal-country-string groups + boilerplate-free "core" tokens + script-agnostic romanized skeletons (`p1_romanize`, unicodedata only) + house-number edit distance** raise dense CV 0.9696 → 0.9768 (India 0.957 → 0.972) and LOCO 0.905 → 0.923. — `…T08_g_fafd84.json`, `…T08_g_fafd84_loco.json` (R3)
- I-21 [medium] At density the best threshold is ~0.65–0.7 (+0.001 over 0.5); expected-F0.5 set selection ≈ same; stage-2 stacking on OOF-probability context adds +0.0001 → not worth it. The 300-round cap (base.yaml's HGB `max_iter` leaking into LightGBM) cost 0.002. — `exploration/p1/dense_decide_out.md`, `dense_stack_out.md`, `…T07_lgb_e2e7f4.json` (R3)

## Round 5 — Sun 27 Sep (p1 overnight)
- I-22 [high] No data leakage: entity ids are random and file row order is shuffled (corr of S1 vs match row position / id = 0.000, train GT) → top teams do real ER; public ≈ private (LB standard error ~0.0002). — read-only check (R5)
- I-23 [high] Remaining loss is concentrated: entities with an empty-address true record = 14% of S1 but 46% of dense-CV loss (mean F 0.948); entities with a blocking-missed true record = 6% of S1, 49% of loss; all other entities F 0.995. 24% of empty-address true records share their exact core name with ≥ 1 other S1 of the country (1.06% of true pairs, irreducible). — T12_noise2 OOF analysis (R5)
- I-24 [high] p3's T15 blocking (3 channels + reverse caps) raises dense-CV blocking recall 0.979 → 0.989 (oracle 0.9928 → 0.9963); with noise + pruning + fine-tuned E5 + p1_ambig: dense CV 0.98722 (T17) vs 0.98405. — `exploration/p1/dense_cv/20260927-013724_densecv_T17_530df3.json` (R5)

## Round 7 — Sun 27 Sep (p1, test-vs-train diagnostics, all label-free on test)
- I-25 [high] Test has 5.5–5.8 S2/S3 records per S1 vs 4.68 in train (≈ 2.3 vs 1.2 unmatched distractors per S1), but the
  extra distractors are easy: US/India test predictions reproduce dense-CV statistics (house-number conflicts among predicted
  pairs 6.18% vs 6.07% US, 2.16% vs 2.15% India; empty-address records assigned ≈ 54–60% both; same-address class counts
  match). → US/India LB ≈ dense CV. — `exploration/p1/{orphan_probe,num_conflict,empty_oof,empty_addr_rates}.py` (R7)
- I-26 [high] The LB stall is France (15% of test S1): #4→#7b changes France 3–5× more than US/India (fine-tuned E5 6.8% vs
  1.4% of S1; threshold 0.8 3.8% vs 1.1%); LB then implies France ≈ 0.94 → 0.93. LOCO's failure (shifted-number near-copies,
  321k FP) is absent in France predictions (0.9% number conflicts) → LOCO is a poor France proxy for thresholds. —
  `exploration/p1/{diff_subs,loco_fp}.py` (R7)
- I-27 [high] France names come from a small vocabulary: 56 words exceed 1% S1 document frequency (club 8.7%, amicale 5.5%,
  ecole, maison, centre, comite …) → 77.8% of France S1 names keep ≤ 1 "core" word (US 8.4%, India 44%) → core-name features
  are degenerate for France (siblings "DD Comite"/"DD Musique" share the core). — `exploration/p1/name_vocab.py` (R7)
- I-28 [high] France has 5–8× more same-address, differently named candidates per S1 after pruning (0.60/S1 vs 0.07 US, 0.13
  India; partial-name 0.38 vs 0.24/0.28), i.e. co-located businesses / siblings; we assign 66% of France partial-name and 23%
  of renamed ones (≈ 12% of France predictions) — the likely France precision loss. — `exploration/p1/sibling_probe.py` (R7)
- I-29 [high] With both label-learned (p1_noise) and label-free mined (p1_mnoise) noise-word features, LightGBM uses the
  label-learned ones (8.7% vs 1.4% of gain): redundancy hides the country-agnostic signal from the model. Dropping the
  label-learned per-word rates costs nothing measurable in-country (slice logloss 0.00724 vs 0.00728; dense CV 0.98732 vs
  0.98739) and forces the model onto statistics that exist for an unseen country. General lesson: when a feature has a
  label-learned and a label-free version, keep only the label-free one if the target domain is unseen. —
  `exploration/p1/{feat_importance,slice_ablation}.py`, dense CV T27/T28 (R7)
