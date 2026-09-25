# STATE — single source of truth for "where are we" (planner drafts, lead commits)

_Last updated: Fri 25 Sep 15:30 IST (round 1 — data released; lead running T02)_

## Team
| handle | person | machine / GPU | branch | current focus |
|---|---|---|---|---|
| p1 | MadKane (lead) | _fill_ | w/p1 | data landing, eval freeze, submissions, decision rule |
| p2 | _fill_ | _fill_ | w/p2 | EDA → normalization |
| p3 | _fill_ | _fill_ | w/p3 | blocking / candidate generation |
| p4 | _fill_ | _fill_ | w/p4 | features / matcher → neural |

## Timeline (IST)
- Syncs: 09:30 · 18:30 · 23:30 (overnight launch). Deadline **Sun 27 Sep 23:59**.
- Sun 12:00 feature freeze · Sun 18:00 code freeze · Sun 21:00 zip + methodology submitted.
- Portal: max **5 submissions/day** (Fri/Sat/Sun). Keep ≥ 1 spare each day.

## Eval
- Lock: `planning/eval_lock.json` — **not yet written** (lead runs `python src/run.py lock` in T02).
- Main split `val_v1` (20% of S1 entities, grouped + stratified); dev iteration on the locked 10% sample `val_v1@0.1`.
  Unseen-country proxies: `xc_India(@0.1)`, `xc_US(@0.1)`.
- Data (line counts): train ≈ 2.21M S1 / 5.03M S2 / 5.29M S3; test ≈ 1.73M S1 / 4.89M S2 / 5.08M S3 (~2.4 GB).
- Public LB context (Fri ~15:00): top 0.984, 5th ≈ 0.980 → the race is in the last 1–2 points.

## Current best pipeline
- Config: `configs/best.yaml` (= base.yaml: basic normalize → name+address char-TF-IDF kNN → string sims → HGB → threshold 0.5 + exclusive right)
- Evidence: _none yet (no real data)_

| split | F0.5 | CI95 | oracle F0.5 | cands/S1 | run_id |
|---|---|---|---|---|---|
| val_v1 | | | | | |
| xc_India | | | | | |

## Leaderboard submissions
| # | day | time | test run_id | config/commit | val_v1 F0.5 | public LB F0.5 | note |
|---|---|---|---|---|---|---|---|

## Known issues / decisions
- D1: Handles p1–p4; worker branches `w/p1..w/p4`; only the lead pushes to `main`.
- D2: Numbers in planning come only from results JSONs (see PLANNER.md §1).
- Open organizer questions: see QUESTIONS.md (answers change what is allowed — update here when answered).
- video-slides.pdf in the claude.ai project is image-only; the planner could not read it.
- D3: data lives at `<repo>/data/` (organizers' student_resource contents, git-ignored); organizer docs in `docs/`;
  official validator copied to `scripts/official/`; `Documentation_template.md` at repo root.
