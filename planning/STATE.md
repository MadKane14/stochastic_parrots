# STATE — single source of truth for "where are we" (planner drafts, lead commits)

_Last updated: Fri 25 Sep 21:30 IST (round 2 — eval locked, scalable blocker on main; submission #1 in progress)_

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
- Config: `configs/best.yaml` = basic normalize → **`p1_compound` blocker (k=30, k_rev=3, max_df=1000)** → string_sims →
  HGB → threshold 0.5 + exclusive right.

| split | F0.5 | CI95 | oracle F0.5 | cands/S1 | run_id |
|---|---|---|---|---|---|
| val_v1@0.1 | 0.9843 | [0.98361, 0.98496] | 0.9986 | 33.53 | 20260925-205725_p1_T02_T02_cmp_val_v1@0.1_35fb75 |
| val_v1 | _not run (memory, see T03)_ | | | | |
| xc_India@0.1 | _pending (T03)_ | | | | |

Full-density blocking (full train, `exploration/p1/blk_eval_out.md`, run 20:51): pair recall 0.9699, oracle F0.5 0.9892,
33.4 cands/S1 (p95 43, max 377); ≈ 12 min per full dataset with 4 workers, peak 37 GB.

## Leaderboard submissions
| # | day | time | test run_id | config/commit | val_v1 F0.5 | public LB F0.5 | note |
|---|---|---|---|---|---|---|---|

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
- Open organizer questions: see QUESTIONS.md.
- video-slides.pdf in the claude.ai project is image-only; the planner could not read it.
