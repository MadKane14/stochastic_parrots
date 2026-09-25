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
