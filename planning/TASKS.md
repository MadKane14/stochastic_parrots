# TASKS — workers: do only the tasks with your handle in the CURRENT round

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

---
## Round 2 — drafts, assigned after T02 lands (subject to EDA)
- **p3 — Blocking recall/efficiency:** push oracle F0.5 → ≥ 0.995 at ≤ 50 cands/S1 (rare tokens, postal codes,
  char n-gram ANN, script-aware keys for Devanagari), runtime on full test.
- **p4 — Matcher v2:** richer features (IDF-weighted token overlap, rarest shared token, digit agreement, cross-script
  flags) + LightGBM; then (overnight, GPU) cross-encoder / multilingual embeddings (MIT/Apache only).
- **p2 — Normalization & unseen-country robustness:** abbreviation mining from train pairs, Devanagari↔Latin handling
  without external data, junk-prefix cleanup; judged on `xc_*@0.1` gap as well as `val_v1@0.1`.
- **p1 — Decision rule + submissions:** F0.5-optimal per-S1 set selection, exclusive-right assignment, threshold per
  score calibration; ensembling winners; val↔LB tracking.

---
## Archive
- Round 0 (pre-data, fake data): T01–T06 drafts superseded by this round.
