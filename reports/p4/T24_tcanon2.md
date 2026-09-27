# p4 — France follow-up to T23 (lead-approved, Sun 14:05): p4_tcanon2

Status: done (code + tests + label-free proxy on France). Not run on dense CV (phoenix only).
Verdict: **MARGINAL — not worth the pre-freeze risk; recommend keeping FINAL = T18 (p4_tcanon).** The department-vs-region
gap is real (31.8% of France S2/S3) but the mismatched tokens are common, low-IDF words, so the IDF-based blocker and
features already pay little for them. Measured gain on France (label-free, IDF-weighted): +0.0014 separation.

## What p4_tcanon2 adds (`src/ber/normalize/p4_tcanon2.py`, new file; `p4_tcanon.py` untouched so T18 caches stay valid)
1. **Side-skewed address segments dropped.** Raw address split at `, ;` + parenthesised parts; each segment is first
   canonicalized with p4_tcanon's mined map; a segment S2/S3 uses >= 100x more often than S1 (df >= 1000, S1 df <= 0.1%)
   is removed (its tokens, last occurrences, from addr_norm). France: `gironde`, `nord`, `pas de calais`,
   `loire atlantique`. Generic: no word lists, no country branch, per country-string group.
2. **Glued number+letter split** (`6b` -> `6 b`), so the miner's `b -> bis` applies and the house number becomes a key.
Unit tests: `exploration/p4/test_tcanon2.py` (3/3). Config: `configs/exp/p4/T18b_tcanon2.yaml` (= T18, normalizer only).

## Evidence — `exploration/p4/tcanon2_compare.py` -> `exploration/p4/tcanon2_compare_out.md` (France TEST, no labels)
162,186 label-free confident pairs (same (house number, address word) keys + name token_set_ratio >= 90, fixed on basic
clean text, independent of both normalizers) vs 50,000 random S1 × S2/S3 pairs; addr_norm token overlap:
| normalizer | confident Jaccard | random Jaccard | confident IDF-weighted | random IDF-weighted | IDF separation |
|---|---|---|---|---|---|
| p4_tcanon (FINAL) | 0.7947 | 0.1008 | 0.8897 | 0.0298 | 0.8599 |
| p4_tcanon2 | 0.7913 | 0.0990 | **0.8907** | **0.0294** | **0.8613** |
- Plain Jaccard drops because dropping `loire atlantique` / `pas de calais` also drops tokens S1 shares by accident
  (`loire` in "Pays de la Loire", `de`); IDF weighting (what the pipeline uses) shows a small gain.
- A first version dropped city segments `st herblain`, `st nazaire` (S1 writes "Saint-…") — fixed by canonicalizing
  segments with the mined map BEFORE the side-skew test (now only the four departments are dropped).
- Not measured: US/India (harm check needs labels → dense CV) — the rule only fires on segments S1 ~never uses.

## If the lead still wants it (phoenix, ~1–1.5 h incl. new normalize/block/feature caches)
```bash
nice -n 10 python exploration/p1/dense_cv.py -c code/business_entity_resolution/configs/exp/p4/T18b_tcanon2.yaml
nice -n 10 python exploration/p1/dense_cv.py -c code/business_entity_resolution/configs/exp/p4/T18b_tcanon2.yaml --loco
```
Adopt only if dense CV >= 0.98752 (both folds) and LOCO >= 0.93538 (T18). Expected change: within noise.
