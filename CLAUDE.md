# CLAUDE.md — rules for every worker agent (team stochastic_parrots)

You are a **worker** in a 4-person team competing in the Amazon ML Challenge 2026 (Business
Entity Resolution). Each team member runs one worker session on their own machine. Workers
cannot talk to each other: **git files are the only channel**. A planner (with the lead) reads
your report and results, then writes the next tasks. Follow this file exactly; if something here
blocks your task, say so in your report instead of working around it.

## 1. Problem in 60 seconds
- Three sources of business records (TSV, **always `sep="\t"`**): `entity_id` (prefix `S1-`/`S2-`/`S3-`),
  `business_name`, `business_address`, `country`. Source 1 is deduplicated.
- For **every** Source 1 entity, output the set of matching S2/S3 records (zero, one or many).
- Metric: **macro F0.5 per Source 1 entity, singletons included** (empty truth + empty prediction = 1.0;
  any prediction on a singleton = 0.0). Precision counts 2x recall → false merges are expensive.
- Train covers US + India. **Test adds France (unseen)**. Every test S1 entity (France included) must get a row.
- Outputs: `matching_results.tsv` (scored) and `candidate_pairs.tsv` (blocking set; matches ⊆ candidates).

## 1b. Data facts (verified line counts; everything else → EDA)
- Location on every machine: `<repo>/data/` = contents of the organizers' `student_resource/` zip folder
  (`data/dataset/{train,test}/*.tsv`, `data/utils/validate_submission.py`). Git-ignored. Default `BER_DATA_DIR`
  already points at `data/dataset`. Organizer README: `docs/organizers_README.md`.
- Size: train ≈ 2.21M S1, 5.03M S2, 5.29M S3 (GT has one row per S1); test ≈ 1.73M S1, 4.89M S2, 5.08M S3. ~2.4 GB.
  → **Anything O(n_S1 × n_right) is impossible.** Blocking must be ~linear (inverted index / ANN / sorted keys).
  → Vectorize; avoid Python loops over 10M rows where possible; cache heavy intermediates under `$BER_ARTIFACTS`.
- Names can be in Devanagari (Hindi) as well as Latin script; names have junk like leading `--`; addresses can be
  empty; France test rows look like `63 R. DE DIEPPE, LILLE, Hauts-de-France`.
- Iterate on the locked dev split **`val_v1@0.1`** (10% sample, default); confirm winners on full `val_v1`.
  Country hold-outs: `xc_India@0.1`, `xc_US@0.1`.

Background reading: `docs/research_playbook.md` (external research on similar Kaggle competitions — unverified;
where it conflicts with our EDA in `exploration/p1/eda_out.md`, the EDA wins).

## 2. Competition rules (breaking one can disqualify the team)
1. **No external data or lookups**: no APIs, geocoding, business registries, web scraping, external datasets,
   or any internet augmentation. Only the provided train/test files.
2. **Models** (organizers' clarification, Fri 25 Sep): **every** model used anywhere — embedder, reranker, matcher,
   preprocessing model — must *independently* be **MIT or Apache-2.0** licensed and **≤ 8B parameters**, run **offline**,
   and be fine-tuned **only on the provided data**. Hosted LLM APIs (Claude/Gemini/ChatGPT/…) are **not allowed** in the
   pipeline. Check the model card licence + parameter count and record both in your config (`# model: …, licence: …, params: …`).
   Not allowed: Llama, Gemma, Qwen2.5-3B/72B (Qwen licence), CC-BY-NC / research-only models, Qwen3-8B (8.2B > 8B).
   Libraries: prefer permissive ones; avoid GPL code in `src/` (e.g. use `anyascii` (ISC) instead of `unidecode` (GPL)).
3. **Country is an open set.** Never hard-code, filter, one-hot, or branch on specific country values
   (`if country == "India"`), and never drop rows by country. Learn from data; features like
   "same country string" are fine.
4. **Never tune on the leaderboard**, and never submit to the portal yourself (the lead submits).
5. LLM-generated synthetic data, hand-written country-specific dictionaries, and fitting unsupervised
   statistics on test records are **open questions** (see planning/QUESTIONS.md). Don't rely on them
   unless your task says so; if you do, mark the component `transductive = True` / note it in the report.

## 3. Repo map and who writes what
```
CLAUDE.md                         this file                          (lead only)
planning/                         STATE, TASKS, INSIGHTS, QUESTIONS, eval_lock.json   (lead/planner only)
code/business_entity_resolution/  the final submission package
  src/run.py                      the ONLY entry point for experiments
  src/ber/{normalize,blocking,features,scorers,deciders}/   stage implementations (add your files)
  src/ber/eval/                   FROZEN harness (lead only)
  src/ber/{data,config,interfaces,schema,pipeline,registry,submission}.py   core (lead only unless your task says)
  configs/base.yaml, best.yaml    baseline / current team best (lead only)
  configs/exp/<handle>/           YOUR experiment configs
  requirements*.txt               pinned env (lead only — ask in your report)
reports/<handle>/                 YOUR task reports
results/<handle>/                 YOUR machine-generated results (written by the harness, never by hand)
exploration/<handle>/             YOUR notebooks / EDA scripts (not part of the package)
scripts/                          fake data, collect, packaging (lead only)
```
You only create/modify files in: your own `src/ber/<stage>/<handle>_*.py` files, `configs/exp/<handle>/`,
`reports/<handle>/`, `results/<handle>/` (via the harness), `exploration/<handle>/`. If you need a core
change, write the exact patch under "Core change requests" in your report.

## 4. Stage interfaces (read src/ber/interfaces.py before coding)
`records → normalize → blocking (candidate pairs) → features → scorer (score ∈ [0,1]) → decider (matches)`
- Normalizer adds columns (must add `name_norm`, `addr_norm`); never drops rows.
- Blocker returns `[s1_id, cand_id, score]`; several blockers are unioned by the pipeline.
- FeatureSet returns `f_<name>_*` columns aligned with the pairs.
- Scorer returns probabilities; a neural model can be a Scorer (reads records via `ds.lookup`) or a FeatureSet
  (e.g. embedding cosine) — prefer FeatureSet when it can feed the shared GBDT.
- Decider turns scores into matches (threshold, exclusivity, per-S1 caps…).
Register new components with a name prefixed by your handle: `@register("blocker", "p3_addr_tfidf")`, in a new
file `src/ber/blocking/p3_addr_tfidf.py`. Discovery is automatic. `run.py` checks every stage contract.

## 5. How to run (from `code/business_entity_resolution/`)
```bash
export BER_MEMBER=p2                       # your handle (PowerShell: $env:BER_MEMBER="p2")
# BER_DATA_DIR defaults to <repo>/data/dataset — set it only if your data lives elsewhere
export BER_ARTIFACTS=/big/disk/ber_artifacts
python src/run.py check                                        # env + data sanity
python src/run.py run -c configs/exp/p2/T05_x.yaml --smoke     # 1% sample, plumbing only — ALWAYS first
python src/run.py run -c configs/exp/p2/T05_x.yaml --task T05  # dev run: split val_v1@0.1 (default)
python src/run.py run -c ... --task T05 --split val_v1,xc_India@0.1   # full val + unseen-country proxy
python src/run.py sweep -c ... --task T05 --grid "decider.params.threshold=0.4|0.5|0.6" --autocommit --push
python tests/test_metrics.py
```
Experiment config = `inherit: ../../best.yaml` + only the keys you change + a comment with the hypothesis.
Big files (embeddings, checkpoints, caches) go under `$BER_ARTIFACTS`, never in git.

## 6. Evidence rules
- **Every number you report must come from a results JSON** and be cited by `run_id`. No hand-computed
  or remembered scores. EDA facts must come from a committed script in `exploration/<handle>/` (cite it).
- Compare only runs on the same split with an OK eval hash. Differences inside the CI95 are ties.
- Smoke results (`results/<you>/smoke/`) are never evidence.
- Commit before long runs so the results JSON carries a clean commit hash (dirty runs are flagged).
- Never edit, delete, or re-generate someone else's results; never edit a results JSON by hand.

## 7. Session protocol
**Start:** `git checkout w/<handle> && git fetch origin && git merge origin/main` → read `planning/STATE.md`,
your tasks in `planning/TASKS.md` (current round only), and `planning/INSIGHTS.md`.
**Work:** one task at a time, inside its time-box. Stick to the task's hypothesis; note side ideas in the report.
Smoke-test → full run → (optional) sweep. Commit small and often on `w/<handle>`.
**Overnight:** launch only sweeps whose smoke passed; use `--autocommit --push` so results reach GitHub.
**Finish (or at each sync, even if unfinished):** write `reports/<handle>/<TASK_ID>.md` from
`planning/templates/REPORT.md`, commit code + configs + results + report, `git push origin w/<handle>`.
Never push to `main`; never merge other workers' branches.

## 7b. Shared machines
Lab servers are shared and have strike policies. If `docs/servers/<hostname>.md` exists for your machine, it is
mandatory (install it as `~/.claude/CLAUDE.md`). Always: data/artifacts on the machine's scratch disk (never a small
NFS home), `BER_THREADS` ≤ 12 (run.py applies it to every library), `nice -n 10` for long jobs, one GPU at a time
via `CUDA_VISIBLE_DEVICES`, never `n_jobs=-1` / `workers=-1`.

## 8. Conventions
Python 3.11, pinned `requirements.txt`. Read TSVs with `dtype=str, keep_default_na=False` (names like "NA").
Deterministic seeds. Short docstrings on public functions (the final package is reviewed by the organizers).
No notebooks inside `src/`. Keep runtime of a full val run visible in your report.
