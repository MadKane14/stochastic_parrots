# PLANNER.md — procedure for each planning round (Cowork planner + lead)

Syncs (IST): **09:30 morning**, **18:30 evening**, **23:30 night (launch overnight runs)**.
Target: ≤ 30 min per sync. The lead does all git operations; the planner only reads the repo and
drafts the files in `planning/` for the lead to review and commit on `main`.

## 0. Collect (lead, 2 min)
```bash
git checkout main && git pull
git fetch --all --prune
python scripts/collect.py --since <last sync ISO time>
```
Planner reads `sync/digest.md`, `sync/results_all.jsonl`, `sync/reports/**`, and the current
`planning/STATE.md`, `TASKS.md`, `INSIGHTS.md`.

## 1. Verify (planner) — before believing anything
For every claim in every report:
- **Numbers only from results JSONs.** Every score must match a `run_id` in `results_all.jsonl`.
  A number with no run_id, or not matching the JSON → mark **UNVERIFIED** in the synthesis; it cannot
  change `best.yaml` or STATE.
- **Eval integrity.** Exclude runs flagged `EVAL_MODIFIED`, `SPLIT_MISMATCH`, `FAILED`, `SAMPLE`.
  If any worker branch touched `src/ber/eval/`, `planning/`, or `CLAUDE.md` → flag to the lead
  (`git diff origin/main...origin/w/<h> --stat -- code/business_entity_resolution/src/ber/eval planning CLAUDE.md`).
- **Comparability.** Same split, same eval hash; baseline run on the same commit lineage.
  Overlapping CI95 → "tie", not "improvement". `DIRTY` runs: usable as signal, not as final evidence.
- **Rules.** Any new model: licence MIT/Apache-2.0, ≤ 8B params? Any country-specific branching,
  external data, or `TRANSDUCTIVE` component? Flag in the synthesis.
- **Sanity.** oracle_f05 must be ≥ f05; singleton_acc and by-country numbers plausible; xc_* gap noted.

## 2. Synthesize (planner → lead)
Short message to the lead:
1. **Scoreboard delta**: best verified val_v1 / xc_* F0.5 now vs last sync (run_ids).
2. **Per task**: verdict (supported / rejected / inconclusive / blocked), 1 line of evidence.
3. **Flags**: unverified claims, eval-integrity issues, rule risks, failed overnight runs.
4. **Proposed merges**: which worker components to merge into `main` and the proposed new `best.yaml`
   — only if verified better than current best on val_v1 AND not worse on xc_* beyond CI.
5. **Next round tasks** (draft, format below) + submission plan (≤ 5/day).
Lead approves / edits.

## 3. Write (planner drafts, lead commits on main)
- `STATE.md`: current best (config + run_ids + scores), LB log, known issues, decisions. Overwrite.
- `INSIGHTS.md`: append new numbered insights (`I-NN`), each with evidence run_ids and confidence
  (high / medium / low). Mark superseded ones `~~struck~~ (see I-MM)`; never delete.
- `TASKS.md`: new round on top (`## Round N — <sync>`), previous round moved below "Archive".
- `configs/best.yaml`: only after the lead merges the winning components.
Lead: `git add planning code/business_entity_resolution/configs/best.yaml && git commit -m "plan: round N" && git push`.
Merging a worker's components: `git merge --no-ff origin/w/<h>` (their files are additive), run
`python src/run.py run -c configs/best.yaml --task MERGE --split val_v1,xc_India` on main to confirm.

## 4. Task design rules
- **One hypothesis per task**, falsifiable, with a metric and a baseline run_id to beat.
- Time-box ≤ 4 h of active work (overnight sweeps extra). Anything bigger → split.
- Keep the 4 directions independent (different files/components) so they combine at merge time.
- Always keep one person on "the thing that makes the next submission better".
- Prefer changes that help the **unseen country** (check xc_*), since France is in test.
- Submissions: each needs a `test`-mode run_id; log val score + LB score in STATE to track val↔LB agreement.

## 5. Task format (copy from templates/TASK.md)
```
### T<NN> — <short title>
- Owner: p<k> · Branch: w/p<k> · Time-box: <h> (report due: <sync>)
- Hypothesis: <one falsifiable sentence>
- Change: <what to implement; component/file names>
- Metric: <primary metric + split> (secondary: …)
- Baseline to beat: <run_id> = <value>   (verified)
- Done when: <concrete criteria incl. report + results committed>
- Overnight: <sweep grid or "no">
- Don't: <scope limits>
```

## 6. Endgame (Sunday 27 Sep)
- 12:00 feature freeze: no new directions; only tuning of best.yaml.
- 18:00 code freeze on main. Final `test` run from a clean checkout of main; package with
  `scripts/package_submission.py`; re-run in a fresh env from the zip's README.
- Final portal submission choice: best verified val_v1 with acceptable xc_* — not the best public LB alone
  (private LB decides; public LB is a subset).
- 21:00 zip + methodology doc submitted (buffer until 23:59).
