# README reproduction on GPU from a clean copy of main — p4 (Sun 22:40; ran after the 21:00 submission)

Status: done. Verdict: **the README path reproduces end to end on a GPU from a clean copy of the final `main`
(`9db9ed4`, `configs/best.yaml` = T29 ensemble): every step exits 0.** No README or code change is required.
Kaggle kernel `ber-p4-repro` (2× T4, 4 cores, 31 GB, **Python 3.12.13**), queued 17:22–22:10, ran 22:10–22:32.
Logs: `exploration/p4/repro/logs/`; smoke results JSONs: `exploration/p4/repro/`.

| README step | command (from `code/business_entity_resolution/`) | exit | time |
|---|---|---|---|
| environment | `pip install -r requirements.txt` (only that file) | 0 | 153 s |
| check | `python src/run.py check` (all pins OK, data found, eval hash = lock) | 0 | 17 s |
| 1) fine-tune | `python src/e5_finetune.py --split full --out $BER_ARTIFACTS/models/ft_full --n 20000` (small n) | 0 | 173 s |
| 2) run (smoke) | `python src/run.py run -c configs/best.yaml --smoke` | 0 | 309 s |
| 2) test (smoke) | `python src/run.py test -c configs/best.yaml --smoke` | 0 | 509 s |

- The smoke run exercises every final component: `p4_tcanon`, the three p3 blockers, all six feature sets incl. `p1_e5`
  loading the step-1 model from `${BER_ARTIFACTS}/models/ft_full`, the learned pruning stage, the two-member
  `weighted_mean` ensemble, `p1_assign` @0.8. Smoke val_v1@0.1: F0.5 0.9982 (1% sample, plumbing only, not evidence).
- Test smoke writes both TSVs; **our validator: no issues**. The official validator reports FAIL **only because the
  smoke predicts a 1% sample** (17,276 of 1,732,544 S1 rows: "required S1 entity(ies) missing"). Expected for `--smoke`;
  the full run writes all S1 (the lead's #9 file passed the official validator with `--check-ids`).
- Environment notes (not README issues): Python 3.12 instead of the README's 3.11 works with every pinned wheel; on Kaggle
  the image's preinstalled `torchvision` breaks `import transformers` under torch 2.8.0 and was uninstalled first
  (Kaggle-specific, a fresh env from requirements.txt has no torchvision).
- Cosmetic: `pipeline.py:131` emits pandas `PerformanceWarning: DataFrame is highly fragmented` (columns attached one
  set at a time for memory, `cb81d3a`); harmless. If wanted: `warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)` there.
