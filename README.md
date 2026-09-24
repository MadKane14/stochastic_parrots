# stochastic_parrots — Amazon ML Challenge 2026 (Business Entity Resolution)

Team workspace. **Agents: read `CLAUDE.md`. Planner/lead: read `planning/PLANNER.md`.**

```
repo root  ==  root of the final <team>_submission.zip
├── output/                          final TSVs (committed only at the end, `git add -f`)
├── code/business_entity_resolution/ the runnable package: src/, configs/, README.md, requirements.txt
├── Documentation_template.md        methodology doc (copy the official template here when released)
├── CLAUDE.md                        worker-agent rules
├── planning/                        STATE, TASKS, INSIGHTS, QUESTIONS, eval_lock.json, PLANNER.md  (lead only)
├── reports/<handle>/                worker reports
├── results/<handle>/                machine-generated results JSONs (one per run)
├── exploration/<handle>/            EDA / notebooks
└── scripts/                         make_fake_data.py, collect.py, package_submission.py
```

## The loop
1. **Plan** (sync 09:30 / 18:30 / 23:30 IST): lead + planner write `planning/TASKS.md` on `main`.
2. **Execute**: each worker merges `origin/main` into `w/<handle>`, runs its tasks with `src/run.py`
   (every run writes `results/<handle>/<run_id>.json` with commit, config, eval hash, metrics).
3. **Report**: worker writes `reports/<handle>/<task>.md`, pushes `w/<handle>`.
4. **Consolidate**: lead runs `git fetch --all && python scripts/collect.py` → `sync/digest.md`;
   planner verifies, updates STATE / INSIGHTS / TASKS; lead merges winning components + updates `configs/best.yaml`.

## Quickstart (every machine)
```bash
git checkout -b w/<handle>
conda create -n ber python=3.11 -y && conda activate ber
pip install -r code/business_entity_resolution/requirements.txt
python scripts/make_fake_data.py --out data/fake_dataset      # until the real data lands
export BER_MEMBER=<handle> BER_DATA_DIR=$PWD/data/fake_dataset BER_ARTIFACTS=/big/disk/ber_artifacts
cd code/business_entity_resolution
python src/run.py check && python src/run.py run --smoke
```

## Submitting (lead)
```bash
python src/run.py test -c configs/best.yaml --task SUBn
python ../../scripts/package_submission.py --run-dir $BER_ARTIFACTS/submissions/<run_id>   # Sunday only
```
