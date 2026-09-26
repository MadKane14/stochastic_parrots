# TASKS — workers: do only the tasks with your handle in the CURRENT round

## Round 4 — Sat 17:30, FINAL NIGHT (lead p1). Push results + a short report by **21:00**; last sync Sun 09:30.
Sun 12:00 feature freeze · 18:00 code freeze · 21:00 zip + methodology. Top-100 LB ≥ 0.984; we are at 0.969 (#2).

### Where we are (dense CV = decision metric; `exploration/p1/dense_cv/`)
| config | dense CV | India / US | note |
|---|---|---|---|
| T08_sub (= LB #2 0.969) | 0.97885 | 0.9747 / 0.9816 | |
| T09_tr (+ p2_translit D) | 0.98279 | 0.9826 / 0.9829 | blocking recall 0.970 → 0.979 |
| T10_e5 (+ pretrained E5 via p1_e5) | 0.98313 | 0.9829 / 0.9833 | submission #3 test run on phoenix now |
Remaining loss ≈ 0.017: blocking ≈ 0.007 (oracle 0.9928), matcher ≈ 0.010; LB sits ~0.010 below dense CV (France unseen +
more distractors in test). All branches (p2 face0ff/27acb15/6ca940b, p3 10fec1b, p4 d81aecd) are merged into main.
**Phoenix is the only ≥ 60 GB machine: send p1 exact commands for anything that needs dense CV / LOCO.**

### T14 — p2: normalization for density + France (Kaggle CPU)
a) Screen `T10_e` (abbreviation canon) and `T10_f` (+ junk) against `T10_d` at TRUE density with p3's
   `exploration/p3/blk_dense.py --dense` (fits 30 GB): recall + cands/S1 (mean/p95/max) + subset recalls. By 20:30.
   p1 runs dense CV only for the winner.
b) France audit on TEST inputs only (no labels; record it as unsupervised/transductive): which S1↔S2/S3 surface
   variations in France (`R.`/`RUE`/`R`, `AV`, `BD`, `N°`, hyphenated regions, accents, `SARL`/`S.A.R.L.`) do our
   learned maps NOT cover (they were learned on US/India pairs)? Propose a label-free fix (e.g. abbreviation pairs
   mined from co-occurrence within blocker candidate pairs of the target data) with a small unit test. By 21:00.

### T15 — p3: blocking recall at small candidate sets (Kaggle CPU)
a) Finish T11 at true density (`blk_dense.py --dense`): empty-address channel variants (T11_e / e10 / e4 / eca) vs
   T11_base; pick the best by recall gain per extra cand/S1. By 20:00 → p1 runs dense CV of the winner.
b) Organizer rule (smaller candidate sets rank higher): for the chosen blocker union, recall vs per-S1 cap (20/30/40/50)
   and a cap on the reverse flood (current max ~380). Note: p1 adds a learned pruning stage after features
   (33 → ~6 cands/S1 at −0.0002 dense F0.5), so blocker caps mainly save compute — report both.

### T16 — p4: fine-tuned E5 (recipe on Kaggle, decision runs on phoenix by p1)
- Running on phoenix now: your US-only recipe (`ft_xc_India`) → train embeddings → dense LOCO, India fold vs pretrained.
a) Improve the recipe (v2 blocker hard negatives, epochs, fields); intrinsic eval on held-out candidates on Kaggle;
   report by 21:00 with the exact `e5_finetune.py` command of the best recipe. If the phoenix LOCO confirms fine-tuning,
   p1 trains `ft_full` with that command overnight and embeds train + test (~2 h GPU).
b) Stretch: cross-fitted embeddings (2 models on S1 halves `int(unit_hash(str(row), 777) * 2)`, row = position in
   `load_part(train).records`) + a `p1_e5` option picking the model by the S1's fold → honest hash-fold dense CV.

### T17 — p1: integration, decisions, submissions
- Submission #3 (T10_e5) ~18:45 → LOCO T09 / T10 / T10-ftUS (India) → E5 + fine-tuning decision.
- Learned pruning stage (pending lead approval) → small `candidate_pairs.tsv`; cluster-consistency features;
  merge p2/p3 winners → submission #4 ~22:00; overnight: final config (ft_full embeddings), #5 Sun morning.

---
## Archive

## Round 3 — Sat 26 Sep 09:30 (lead p1). First finish + push your Round 2 report (T04/T05/T06), then start here.
Checkpoint Sat 18:30; overnight launch 23:30. Feature freeze Sun 12:00.

### UPDATE Sat 14:00 (lead) — read this first
- **LB #2 = 0.969** (T08_sub; #1 was 0.9487). Top of LB ≈ 0.99. Dense CV of T08_sub 0.97885.
- **Organizer update (`docs/organizer_updates.md`): smaller candidate sets per S1 rank higher in the final evaluation.**
  Report cands/S1 (mean/p95/max) next to every recall/F0.5. p1 study (`exploration/p1/prune_eval_out.md`): a cheap
  stage-1 LightGBM (60 trees, 12 cheap features) keeping candidates with score ≥ 0.005 cuts 33.4 → **6.2 cands/S1**
  at −0.0002 dense F0.5; a `prune:` pipeline stage is pending lead approval.
- **p2's `p2_translit` is merged into main** (thanks p4 for running the evals). p1 is running dense CV of
  `configs/exp/p1/T09_tr.yaml` = T08_sub + p2_translit (D).
- **p4:** `p4_e5` merged. At full scale its shard cache holds all vectors in RAM (~18 GB/field) — too much for test
  mode, so p1 added `p1_e5` (same model/texts/cosines, per-dataset memory-mapped matrices, + within-record rank/gap).
  Full train+test embeddings (name, full) are being computed on phoenix now (15k names/s on one A4000).
  Please build on `p1_e5` for T12 (e.g. field ablation incl. `addr`, contrastive fine-tuning on train pairs judged by
  dense LOCO).
- **p3:** nothing pushed yet — please push T05 (even partial) + report before the 18:30 checkpoint. Blocking recall
  is still the largest dense-CV loss, and now the candidate count matters too.

### READ FIRST (everyone) — what changed overnight
1. **LB #1 = 0.9487** while val_v1@0.1 said 0.9917 (I-15). Cause: test density (I-15..I-18), not a bug.
   **val_v1@0.1 is no longer evidence for matcher / decider / feature work** — only for plumbing (smoke-like).
2. New decision metrics (both needed; tools in `exploration/p1/`, need ~55–60 GB RAM, 15–40 min per run):
   - **Dense CV** — out-of-fold over the full train set (test-like density):
     `nice -n 10 python exploration/p1/dense_cv.py -c <your.yaml> [--tag x]` → `exploration/p1/dense_cv/<run>.json`
   - **Dense LOCO** — leave-one-country-out at density, our proxy for unseen France: add `--loco`.
   Copy the JSON into `exploration/<h>/` with your report; cite the run name. Improvements must show on dense CV
   (both folds) and not regress LOCO. Features are cached per feature-config; scorer/decider changes are cheap.
3. Current best (candidate for submission #2): `configs/exp/p1/T08_sub.yaml` = p1_compound blocker + p1_match &
   p1_extra with `idf_group: country` + uncapped p1_lgbm + threshold 0.7. Dense CV of its features: 0.9768
   (T08_g), LOCO 0.9232. Build on it: `inherit: ../p1/T08_sub.yaml`.
4. `ber.features.p1_romanize.name_skeleton(text)` — script-agnostic romanization + consonant skeleton (unicodedata only):
   `"रेड कंसल्टेंट्स प्राइवेट लिमिटेड"` and `"red consultants private limited"` → `"rd knsltnts prvt lmtd"`. Reuse it.

### T09 — Submission #2/#3, merges, final pipeline (p1)
- Test run of `T08_sub` (dense-CV-verified) → validator → lead uploads; log dense CV vs LB in STATE.
- Merge verified worker components into `best.yaml`; re-run dense CV + LOCO on main after each merge.

### T10 — Cross-script / noisy-name normalization at density (p2)
- Hypothesis: adding romanized skeleton tokens (p1_romanize) and/or your learned token map to `name_norm`/`addr_norm`
  of non-Latin records lets the blocker propose cross-script pairs (29% of blocking misses, I-12) → full-density
  blocking recall (blk_eval `--qfrac 0.1`) up ≥ +0.004 and dense CV up, India most.
- Change: your `p2_translit` normalizer; config inherits `p1/T08_sub.yaml` with `normalize: {name: p2_translit}`.
- Metric: blk_eval (vs best.yaml blocker, same machine) → then dense CV + LOCO. Don't: country branches, hand dictionaries.

### T11 — Blocking recall at density (p3)
- Blocking misses are the largest dense-CV loss (0.0129 of 0.030, I-18). Hypothesis: your char-gram channel / reverse
  cap / adaptive k raise full-density pair recall 0.970 → ≥ 0.98 at ≤ 50 cands/S1, and dense CV rises accordingly.
- Metric: blk_eval `--qfrac 1` for finalists, then dense CV with the new blocker (note: new blocker = new cached
  features, ~15 min extra). Keep full-test blocking ≤ 30 min, ≤ 60 GB.

### T12 — Multilingual embeddings as matcher features, judged at density + LOCO (p4)
- Hypothesis: e5 cosine features (name, name+address) raise dense CV over T08_sub AND raise LOCO (multilingual
  encoders should transfer to an unseen country/script better than string features).
- Change: your `p4_e5` FeatureSet added to `features:` of a config inheriting `p1/T08_sub.yaml`; embeddings cached
  per text hash; embed all train + test texts once (GPU overnight if needed).
- Metric: dense CV + dense LOCO vs the same config without p4_e5. Report throughput and total embedding time.


## Round 2 — Fri 25 Sep 21:30 (lead p1 + planner). Reports due **Sat 09:30 sync**; checkpoint at Sat 18:30.

### READ FIRST (everyone)
0. **Guardrails (workers p2–p4, once per machine, before starting your agent):**
   `cp docs/worker_settings.local.json .claude/settings.local.json` — blocks agent edits to lead-only files
   (planning/, CLAUDE.md, eval/, best.yaml, requirements). The file is git-ignored; restart your agent session after copying.
1. **Setup (if T01 is not done yet, do it now — 45 min max):** `git fetch origin && git checkout -B w/<h> origin/main`
   (or `git checkout w/<h> && git merge origin/main`), then install the env. **Pins changed** (`scipy==1.15.2`); on shared
   servers prefer conda-forge (see the header of `code/business_entity_resolution/requirements.txt`); run pip only with the
   env's own Python (`which pip`). Then `python src/run.py check`, `python tests/test_metrics.py`,
   `python src/run.py run --smoke`. Write `reports/<h>/T01.md` (machine, cores, **RAM**, GPU, disk, smoke timings) and push.
2. **Read** `planning/STATE.md` (D4–D8 are new), `planning/INSIGHTS.md` (I-05..I-14 are new), `exploration/p1/eda_out.md`,
   `exploration/p1/blk_eval_out.md`.
3. **`best.yaml` changed:** it now uses the scalable blocker `p1_compound` (k=30, k_rev=3, max_df=1000). Every experiment
   inherits it. Current baseline on `val_v1@0.1`: `20260925-205725_p1_T02_T02_cmp_val_v1@0.1_35fb75` = **0.9843**
   (≈ 4 min per run on a 16-core box; blocking ≈ 75 s, features ≈ 55 s, HGB ≈ 30 s).
4. **Density warning (I-08):** val splits have 5–50× fewer distractors than test, so val scores and val blocking recall are
   optimistic. **Blocking is judged only at full (test-like) density** with the shared tool:
   ```bash
   # from repo root; needs ~30 GB RAM, ~5-10 min; BER_THREADS=4 is enough (memory-bound, I-11)
   nice -n 10 python exploration/p1/blk_eval.py --config code/business_entity_resolution/configs/exp/<h>/<cfg>.yaml \
       --qfrac 0.1 --out exploration/<h>/blk_eval_out.md            # add --misses all=1000000 to print missed pairs
   ```
   `--qfrac 0.1` = all 10.3M S2/S3 records, 10% of S1 as queries (forward recall unbiased; label-fitted components are fitted
   only on the other 90% of S1). Reverse-direction retrieval is optimistic under qfrac → confirm finalists with `--qfrac 1`.
5. **Do not run the full `val_v1` split yet** (≈ 58M training pairs → out of memory). p1 lands a fit-sampling fix (T03);
   until then iterate on `val_v1@0.1` and `xc_India@0.1`.
6. Resource etiquette on shared machines: `BER_THREADS` ≤ 12, `nice -n 10`, one heavy job at a time, tmux + log files.

### T03 — Matcher v1, decision rule, submission #1 (p1)
- Owner: p1 · Branch: main (lead) · Time-box: 3 h (tonight) · report: `reports/p1/T02.md` + T03 section, Sat 09:30
- Hypothesis: address-number features, IDF-weighted name overlap / mismatch mass and group-relative features (rank and gap
  to the best within each S1 and within each S2/S3 record) + LightGBM (no `is_unbalance`) + exclusive-right assignment beat
  the baseline string-sim HGB on `val_v1@0.1` (0.9843) and `xc_India@0.1`.
- Change: `src/ber/features/p1_*.py`, `src/ber/deciders/p1_*.py`; core patch (shown to the lead first): fit on a sample of
  train S1 while keeping all S2/S3 records (keeps test-like density, fixes the full-val memory blow-up).
- Done when: threshold curve + FP/FN breakdown reported; `test --task T03` passes the official validator; lead uploads #1.

### T04 — Cross-script + noisy-name normalization (p2)
- Owner: p2 · Branch: w/p2 · Time-box: 4 h (+ overnight run) · report `reports/p2/T04.md` Sat 09:30
- Hypothesis: a normalizer that adds Latin forms of non-Latin tokens (a token map **learned from train true pairs**: for
  each non-Latin token, the Latin token it co-occurs with most in matched pairs, with support/precision thresholds) and
  splits web-style names (`delhiconsulting.com` → `delhi consulting`, strip `com`/`www`/tld, split concatenations with a
  word list learned from all names; map 1→l / 0→o only in a variant) raises full-density pair recall of `best.yaml`'s
  blocker by **≥ +0.005** over the identical command run on `configs/best.yaml` (same machine, `--qfrac 0.1`), with the
  largest gain on India pairs whose S2/S3 name is non-Latin (29% of current misses, I-12).
- Change: `src/ber/normalize/p2_translit.py` registered `p2_translit`. It must still output `name_norm`/`addr_norm`
  (put the Latin forms INTO those columns, or `name_norm` = original + appended Latin tokens — your choice, test both).
  Learn the map in `fit(train)` (gt allowed there only). Script-agnostic: detect "non-Latin" by Unicode category/script of
  the characters, never by country. Optional: `anyascii` (ISC licence) as a fallback for unmapped tokens — ask the lead to
  add it to requirements before merge (do not add it yourself).
- Metric: full-density recall via `blk_eval.py --config configs/exp/p2/T04_*.yaml --qfrac 0.1` (run the same command on
  `configs/best.yaml` first as your baseline on your machine). Secondary: `val_v1@0.1` and `xc_India@0.1` F0.5 vs 0.9843.
- Done when: best variant has a blk_eval section + a val_v1@0.1 run_id + xc_India@0.1 run_id; report lists the 30 most
  frequent learned mappings (sanity) and how many S2/S3 tokens got a Latin form.
- Overnight: val_v1@0.1 + xc_India@0.1 for your best 2 variants.
- Don't: hand-write country-specific dictionaries; touch blockers/features; branch on the `country` value.

### T05 — Blocking recall to ≥ 0.98 at ≤ 50 cands/S1 at full density (p3)
- Owner: p3 · Branch: w/p3 · Time-box: 4 h (+ overnight) · report `reports/p3/T05.md` Sat 09:30
- Hypothesis: adding a second channel to `p1_compound` — IDF cosine over **character 3/4-grams of the de-spaced name**
  (catches typos `hemandez`/`hernandez`, `consturcrions`, and web names `supremepackaging`) within the same country-string
  groups — plus a per-S1 candidate cap on the reverse flood (current max 377 cands for one S1) raises full-density pair
  recall from **0.9699 → ≥ 0.98** at ≤ 50 mean cands/S1 (p1 baseline: k=30 + k_rev=3, `blk_eval_out.md` run 20:51,
  `--qfrac 1`).
- Change: new blocker `src/ber/blocking/p3_chargram.py` registered `p3_chargram` (reuse `_group_topk`/`topk_rows` from
  `ber.blocking.p1_idf_overlap` by import — don't edit p1 files; copy into your own file if you need changes). Configs
  `configs/exp/p3/T05_*.yaml` with blockers `[p1_compound, p3_chargram]` + `blocking.max_candidates`. Budget: the whole
  blocking of test (1.73M S1 × 10M) ≤ 30 min and ≤ 60 GB RAM.
- Metric: `blk_eval.py --config … --qfrac 0.1` for iteration; finalists with `--qfrac 1` (needs ~40 GB, ~15-25 min).
  Report pair recall, oracle F0.5, cands mean/p95/max, India/US recall, runtime.
- Done when: best config meets the target (or the report says what is left, with `--misses` breakdown), plus one
  `val_v1@0.1` harness run with it (sanity: F0.5 not below 0.9843 beyond CI).
- Overnight: `--qfrac 1` for the best 2 configs.
- Don't: change features/scorer/decider; exceed 12 threads (4 workers is as fast, I-11).

### T06 — Multilingual embedding features for the matcher (p4, GPU)
- Owner: p4 · Branch: w/p4 · Time-box: 4 h (+ overnight GPU) · report `reports/p4/T06.md` Sat 09:30
- Hypothesis: cosine similarities from a small multilingual encoder — `intfloat/multilingual-e5-small` (MIT, ~118M params;
  **verify licence + params on the model card and record both in the config**) — on name, address and name+address, fed as
  a FeatureSet into the shared GBDT, raise `val_v1@0.1` F0.5 above the same-commit baseline (0.9843 now; re-baseline on
  p1's T03 matcher once it is on main) and help most on India / cross-script pairs and on `xc_India@0.1`.
- Change: `src/ber/features/p4_e5.py` registered `p4_e5` (FeatureSet, columns `f_p4_e5_*`): embed each unique record text
  once (`"query: " + text`), cache vectors under `$BER_ARTIFACTS/emb/<model>/<text-hash>.npy`-style shards keyed by text
  hash (never in git), cosine per candidate pair. One GPU via `CUDA_VISIBLE_DEVICES`, batch within 16 GB, fp16, run offline
  after the first download (`HF_HOME` on the big disk). Needs `requirements-gpu.txt`; list any extra package in the report.
- Metric: `val_v1@0.1` F0.5 (secondary: by-country, xc_India@0.1, scorer AUC/AP) vs a baseline run on the same commit.
  Also report embedding throughput (records/s) → projected time to embed all train + test records (~24M texts).
- Done when: 2 runs (with/without p4_e5) on val_v1@0.1 + xc_India@0.1, throughput numbers, report pushed.
- Overnight: embed the full train + test record texts if throughput allows (for the final pipeline).
- Don't: fine-tune yet (next round); use any model that is not MIT/Apache-2.0 or > 8B; send data to any API.

## Round 1 — Fri 25 Sep (real data is in; goal: submission #1 today). Reports due at the 18:30 sync.

Order: p1 runs T02 (EDA-lite → scalable blocker → lock → baseline submission) and pushes to `main`.
Meanwhile p2–p4 do T01. Directions (Round 2) are assigned right after p1's push.
**Full/dev runs made before `planning/eval_lock.json` is on main don't count as evidence.**

### T01 — Environment + real data on every lab machine (p2, p3, p4)
- Owner: p2, p3, p4 · Branch: w/<handle> · Time-box: 45 min
- Hypothesis: the scaffold runs on every machine against the real data.
- Steps (repo root):
  1. `git pull && git checkout -b w/<handle> && git push -u origin w/<handle>`
  2. `conda create -n ber python=3.11 -y && conda activate ber && pip install -r code/business_entity_resolution/requirements.txt`
     (GPU machines also `requirements-gpu.txt`)
  3. Extract the organizers' zip; put the **contents** of `student_resource/` into `<repo>/data/`
     (→ `data/dataset/train/…`, `data/utils/validate_submission.py`). Set `BER_MEMBER`, `BER_ARTIFACTS=<big disk>`.
  4. `cd code/business_entity_resolution && python src/run.py check && python tests/test_metrics.py`
  5. `python src/run.py run --smoke` (1% sample; expected to work with the brute-force baseline blocker).
  6. Report `reports/<handle>/T01.md`: machine, CPU cores, RAM, GPU, disk path, `check` output summary, smoke per-stage
     timings (from the smoke JSON `timings_s`), and how long the first raw-data load took. Commit + push.
- Done when: report pushed. Then read `docs/organizers_README.md` and wait for Round 2.
- Don't: start modelling work; commit anything under `data/`.

### T02 — Lead session: EDA-lite, scalable blocking, eval lock, baseline submission #1 (p1)
- Owner: p1 · Branch: main (lead) · Time-box: ~3 h
- Hypothesis: a simple scalable blocker + the baseline features/HGB/threshold gives a valid submission with val F0.5 ≳ 0.95.
- Steps:
  1. `python src/run.py check`; `python tests/test_metrics.py`.
  2. EDA-lite `exploration/p1/eda.py` → `exploration/p1/eda_out.md` (vectorized, minutes not hours): rows per source ×
     country (train/test); match-count distribution + singleton rate by country; do matched pairs share the country
     label; any S2/S3 id in >1 GT list; fraction of Devanagari names per source; empty name/address rates; how often a true
     pair shares ≥1 name token / a digit token of length ≥ 5 (PIN/ZIP); 20 true pairs + 20 hard negatives.
  3. `python src/run.py run --smoke` → per-stage timings.
  4. Scalable blocker (new file `src/ber/blocking/p1_*.py`), guided by EDA: e.g. inverted index on rare name tokens +
     postal-code tokens (optionally grouped by the `country` string — generic grouping, allowed if EDA shows pairs
     share the label), top-k per S1 by IDF-weighted overlap; or TF-IDF → `sparse_dot_topn`. Target on `val_v1@0.1`:
     oracle F0.5 ≥ 0.97, ≤ 50 cands/S1, full test-set blocking in < 1 h.
  5. `python src/run.py lock` → commit `planning/eval_lock.json` (+ EDA, blocker, config) to main, push.
  6. `run --split val_v1@0.1,xc_India@0.1 --task T02` → `test --task T02` → official validator PASS → submit #1 →
     log run_ids + LB score in STATE.md (planner does this at sync).
- Don't: modify `src/ber/eval/` (it's final; tell the planner if something is wrong *before* `lock`).

- Round 0 (pre-data, fake data): T01–T06 drafts superseded by this round.
