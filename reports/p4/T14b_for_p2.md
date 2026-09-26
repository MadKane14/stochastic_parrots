# T14b (France audit + label-free fix) — done by p4 for p2 (Sat 22:15)

Status: done (audit + fix + unit tests + validation on train countries), and **ready to run**: normalizer `p4_tcanon`
(`src/ber/normalize/p4_tcanon.py`, a subclass of p2_translit in p4's own file) + config `configs/exp/p4/T14b_tcanon.yaml`.
Dense CV / LOCO is not run (phoenix only). Adoption is the lead's call before the Sun 12:00 freeze.
Verdict: **SUPPORTED (proxy level).** The train-learned maps miss the most common French variants and contradict one
(`st`). A label-free map mined on the target data fixes this. On train countries, fitted without labels, it raises
true-pair address overlap by +0.05 (India) / +0.19 (US) while random pairs barely move.
p2 had not started T14b (their last push: T10 report, `06e1692`). The user asked p4 to help.

## 1. France audit (test inputs only, no labels) — `exploration/p4/france_audit.py` → `exploration/p4/france_audit_out.md`
1,694,445 France test records (S1 259,452; S2/S3 1,434,993). Runtime 55 s on a laptop.
- **What `clean` already handles:** accents (`établissements` → `etablissements`), hyphenated regions
  (`Hauts-de-France` → `hauts de france`), punctuation (`R.` → `r`, `S.A.R.L.` → `s a r l`, which p2's
  `join_initials` joins to `sarl`).
- **What it does not handle.** These are tokens S2/S3 uses and S1 almost never uses (df S2/S3 vs S1):
  `r` 356,715 vs 47 (`R.`/`R` = rue); `no` 94,415 vs 1 and `n` 70,366 vs 58 (from `N°`); `ave` 21,614 vs 0;
  `blvd` 6,046 vs 0; `crs`, `ch`, `pass`, and typos `aveue`/`aenue`/`avnue`/`alee`. Department names
  (`gironde` 150,701 vs 62, `nord`, `atlantique`) appear where S1 uses the region.
- **Train-learned map (p2, US/India true pairs) vs France:** of the top-40 train entries only `ave→avenue` and
  `pl→place` apply. **`st→street` contradicts France**, where `st` = `saint` (24,418 S2/S3 addresses).
- **Name side:** additions S1 never has (`participations`, `holding`, `distribution`, `snc`, `dba`, `formerly`,
  `known`, `aka`, `com`, glued `...sascom`, typo `5arl`). These are side-skewed noise tokens. p1's unsupervised
  `p1_noise` (main, `05809d7`) is built for exactly this kind of token, so I didn't add anything for names.

## 2. Label-free fix — `exploration/p4/unsup_canon.py` (+ `test_unsup_canon.py`, 4 tests pass)
Country-agnostic, fitted only on the records being normalized (**transductive = True**), no labels, no hand lists:
1. `confident_pairs`: S1 × S2/S3 pairs sharing ≥ 2 (house number, address word ≥ 4 letters) keys, with name
   `token_set_ratio ≥ 90`. On train, these pairs are 94.5% (India) / 99.6% (US) true matches.
2. `mine_canon`: p2's own `learn_canonical_map` on those pairs (support 50, precision 0.5), plus single-letter forms
   that p2's learner skips (`_is_word` needs ≥ 2 letters) under stricter thresholds (support 500, precision 0.8).
   → `r → rue`, `b → bis`.
3. `merge_maps`: target evidence overrides the train map (`st`: `saint` beats `street`).
4. `side_skew_junk`: after canonicalization, drop tokens S2/S3 uses ≥ 100× more often than S1
   (df_right ≥ 1000, S1 df ≤ 0.1%) → France: `n`, `no`, `gironde`, `nord`.

## 3. Evidence
**France (test, no labels, proxy):** 162,186 confident pairs (68,231 S1). Address token Jaccard, confident vs random:
| | raw | mined map only | full fix |
|---|---|---|---|
| confident pairs | 0.7071 | 0.7277 | **0.7946** |
| random S1 × S2/S3 pairs | 0.0832 | 0.0841 | 0.1000 |
Mined France map (20): `av/ave/aenue/avnue/aveue→avenue, st→saint, all→allee, bd/blvd→boulevard, rte→route,
imp→impasse, crs→cours, pl→place, ch→chemin, res→residence, apt/appt/app→appartement, r→rue, b→bis`.
57% of France S2/S3 addresses change, mostly `r → rue`.

**Validation on train countries** (`exploration/p4/unsup_canon_validate.py` → `exploration/p4/unsup_canon_validate_out.md`;
fitted without labels, scored on 50k true pairs vs 50k random pairs):
| country | true pairs: raw → fix | random: raw → fix | map size / junk |
|---|---|---|---|
| India | 0.6771 → **0.7255** (+0.0484) | 0.0240 → 0.0258 (+0.0018) | 29 (floor/flat/nagar/road typos, state codes) / 28 (null, dl, wb, native-script state names) |
| US | 0.5543 → **0.7470** (+0.1928) | 0.0093 → 0.0178 (+0.0085) | 116 (street types, state codes, typos) / 40 (cdp, pmb, po, box, city/township typos) |
For reference, p2's label-learned map (both countries, held-out): true +0.128, random +0.006
(`exploration/p2/canon_out.md`). This is a different sample, so it is only indicative.

## 4. Caveats
- Proxy metric only (token overlap). Blocking recall / dense CV with the fix are not measured.
- The `rue` effect on random pairs (+0.017 France) comes from a very common word. The IDF-weighted blocker and
  features discount it, but dense CV has to confirm.
- Some junk tokens are typos that a map could repair instead (`raod`, `strete`). Dropping them is the safe default.
- Uses rapidfuzz on the target only; runtime is seconds to minutes per country group on a laptop.

## 5. Ready-to-run component + commands (p1, phoenix)
- `p4_tcanon` = `p2_translit` (same params as T12_noise2 / T13_sub4) + the label-free step on `addr_norm` per
  country-string group; `transductive = True`. End-to-end on all France test records (laptop): 38 s vs 23 s for
  p2_translit alone. `addr_norm` changes for 57.2% of S2/S3 and 3.6% of S1 records (e.g. `154 bd du president wilson`
  → `154 boulevard …`, `appt a419` → `appartement a419`, `16 b impasse` → `16 bis impasse`, `92 rte de la` →
  `92 route de la`); map 22 entries, junk `n no nord gironde`.
- `configs/exp/p4/T14b_tcanon.yaml` = `T12_noise2` (dense CV 0.98405) with only the normalizer swapped.
```bash
nice -n 10 python exploration/p1/dense_cv.py -c code/business_entity_resolution/configs/exp/p4/T14b_tcanon.yaml
nice -n 10 python exploration/p1/dense_cv.py -c code/business_entity_resolution/configs/exp/p4/T14b_tcanon.yaml --loco
```
Adopt only if dense CV ≥ 0.98405 (both folds) and LOCO is not lower than T12_noise2's LOCO.

## 6. Original integration sketch (if p2 prefers to fold it into `p2_translit`)
In `TranslitNormalizer.transform`, per country-string group of the target records (no country values in code):
`cs, cr = confident_pairs(...)`; `cmap = merge_maps(self.canon['addr'], mine_canon(...))`; apply; then drop
`side_skew_junk(...)`. Behind a flag `target_canon: true`, `transductive = True`. Judge by dense CV + dense LOCO
(the LOCO India fold is the unseen-country test: the fix is then fitted on India's own records without labels).
