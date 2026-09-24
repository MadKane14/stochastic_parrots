# TASKS — workers: do only the tasks with your handle in the CURRENT round

## Round 1 — Fri 25 Sep (setup → data → first baseline submission). Reports due 18:30 sync.

Order of the day: T01 now (everyone, on fake data) → data lands → T02 (p1) freezes eval and submits the
baseline → everyone merges `origin/main` (gets the lock) → T03/T04/T05 full runs.
**Full runs made before the lock is on main don't count as evidence** (smoke runs are fine anytime).

### T01 — Setup + dry-run of the whole loop on fake data (ALL)
- Owner: p1, p2, p3, p4 · Branch: w/<handle> · Time-box: 45 min (now / first thing in the morning)
- Hypothesis: the scaffold runs end-to-end on every machine and results reach the planner through git.
- Change: none to code. Steps (from repo root):
  1. `git clone <repo> && cd <repo> && git checkout -b w/<handle> && git push -u origin w/<handle>`
  2. `conda create -n ber python=3.11 -y && conda activate ber && pip install -r code/business_entity_resolution/requirements.txt`
     (GPU machines: `requirements-gpu.txt`)
  3. `python scripts/make_fake_data.py --out data/fake_dataset` and set `BER_MEMBER`, `BER_DATA_DIR=<repo>/data/fake_dataset`, `BER_ARTIFACTS=<big disk>`
  4. `cd code/business_entity_resolution && python src/run.py check && python tests/test_metrics.py`
  5. `python src/run.py run --smoke` then `python src/run.py run --task T01 --split val_v1,xc_India`
  6. Fill your row of the Team table in a report `reports/<handle>/T01.md` (machine, GPU, disk path); commit results + report; push.
- Metric: none (plumbing). Baseline to beat: n/a.
- Done when: the lead's `python scripts/collect.py` lists a T01 run from your branch.
- Don't: change code; start real work before this passes.

### T02 — Data landing, eval freeze, baseline submission #1 (p1)
- Owner: p1 · Branch: main (lead) + w/p1 · Time-box: 1.5 h after the dataset is released
- Hypothesis: the unmodified baseline (`configs/best.yaml` = base) produces a valid submission and a val_v1 F0.5
  that tracks the public LB within ~0.03.
- Change: none to pipeline code. Steps: put the dataset at `BER_DATA_DIR` (tell teammates the path/structure);
  copy `Documentation_template.md` to repo root and `utils/validate_submission.py` next to the dataset;
  `python src/run.py check`; `python src/run.py lock` → commit `planning/eval_lock.json` to main + push;
  `python src/run.py run -c configs/best.yaml --task T02 --split val_v1,xc_India,xc_US`;
  `python src/run.py test -c configs/best.yaml --task T02` → official validator PASS → submit to portal → log in STATE.
- Metric: validator PASS; val_v1 F0.5 (report), public LB F0.5.
- Baseline to beat: n/a (this creates the baseline everyone else compares to).
- Done when: lock on main, T02 run_ids in results, LB score logged, teammates notified to `git merge origin/main`.
- Don't: tune anything; if any stage crashes on real data, fix minimally in core and note it.

### T03 — EDA: facts that decide blocking keys, country use and the decision rule (p2)
- Owner: p2 · Branch: w/p2 · Time-box: 3 h after data lands
- Hypothesis: fact-finding (no model). Each answer must be a number/table produced by `exploration/p2/eda.py`
  (committed) and written to `exploration/p2/eda_out.md`.
- Questions:
  1. Rows per source × country, train and test (how many France S1 in test?). Parse sanity: line counts = rows?
  2. Match-count distribution per S1 (0/1/2/3+), singleton rate by country; share of matches in S2 vs S3.
  3. Is any S2/S3 id in >1 S1 list? What fraction of S2/S3 records match nothing?
  4. Do matched pairs always share the same `country` string? Empty/odd country values?
  5. For true pairs vs random same-country pairs: exact normalized-name match rate, token-Jaccard distribution,
     shared digit tokens (ZIP/PIN), empty name/address rates.
  6. Duplicates inside S2 (or S3) that map to the same S1: how similar are they to each other?
  7. Train vs test shift: name/address lengths, non-ASCII / non-Latin script rate, vocabulary overlap; 20 France test rows verbatim.
  8. 30 random true pairs and 30 hard negatives (high name similarity, not matched) printed for eyeballing.
- Metric: n/a. Baseline to beat: n/a.
- Done when: `reports/p2/T03.md` lists the answers + 5 concrete recommendations (each tagged blocking / features / decider / normalization).
- Don't: build models; use any external data.

### T04 — Blocking recall ceiling v1 (p3)
- Owner: p3 · Branch: w/p3 · Time-box: 4 h (+ overnight sweep optional)
- Hypothesis: the base blockers (name k=20 + address k=10 char-TF-IDF) leave recall on the table; adding a
  digit-token blocker (shared ZIP/PIN/house numbers) and tuning k reaches **oracle F0.5 ≥ 0.98 on val_v1 at ≤ 40 cands/S1**.
- Change: new blocker(s) `src/ber/blocking/p3_*.py`; configs in `configs/exp/p3/`. Keep scorer/decider as in best.yaml.
- Metric: `metrics.blocking.oracle_f05` and `pair_recall` vs `cands_per_s1_mean` on val_v1 (secondary: same on xc_India; end-to-end f05).
- Baseline to beat: T02 val_v1 run (p1) — oracle F0.5 = _fill from T02_.
- Done when: report has a recall-vs-candidates table (run_ids) and one recommended blocker config.
- Overnight: `--grid "blocking.blockers.0.params.k=10|20|40|80"` × your blockers, `--autocommit --push`.
- Don't: filter by country; change features/scorer.

### T05 — Matcher features v2 (p4)
- Owner: p4 · Branch: w/p4 · Time-box: 4 h
- Hypothesis: token-importance-aware features (IDF-weighted token overlap fitted on the train side, rarest shared
  token, numeric-token agreement) + LightGBM raise **val_v1 F0.5 by ≥ 0.01** over base at fixed blocking.
- Change: `src/ber/features/p4_*.py` (+ `scorer: lgbm` in your config). Keep blocking identical to best.yaml.
- Metric: val_v1 F0.5 (secondary: xc_India F0.5, scorer AUC/AP, singleton_acc).
- Baseline to beat: T02 val_v1 run (p1) — F0.5 = _fill from T02_.
- Done when: report with ablation (base features / + new / lgbm vs hgb), run_ids.
- Don't: change blocking or the decider; tune thresholds (that's T06).

### T06 — Decision rule for F0.5 (p1, after T02, optional today)
- Owner: p1 · Branch: w/p1 · Time-box: 2 h
- Hypothesis: a more precision-leaning rule (threshold > 0.5 and/or keep a candidate only if its score ≥ r × the best
  score of that S1) raises val_v1 F0.5 by ≥ 0.005 without hurting xc_India.
- Change: `src/ber/deciders/p1_*.py`; sweep threshold/r on val_v1, confirm on xc_India (guards against overfitting val).
- Metric: val_v1 F0.5 and xc_India F0.5. Baseline to beat: T02 val_v1 run.
- Done when: report + recommended decider params.

### Planned for tonight's 23:30 sync (tentative, not to start yet)
- Merge T04/T05/T06 winners into best.yaml → submission #2.
- Overnight neural directions (GPU): multilingual embedding cosine as feature + blocker (e.g. multilingual-e5-base /
  bge-m3, MIT); cross-encoder fine-tune (xlm-roberta-base, MIT) as a scorer; data-driven abbreviation mining from
  train pairs for normalization (p2).

---
## Archive
_(previous rounds move here)_
