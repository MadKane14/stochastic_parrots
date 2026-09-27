# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** stochastic_parrots
**Team Members:** [fill: 4 members]
**Submission Date:** 27 September 2026

> Sun 27 Sep (lead). Final configuration: `configs/best.yaml`. Only the final leaderboard score is still marked [final].

---

## 1. Executive Summary
A scalable blocking → learned pruning → gradient-boosted matching pipeline. Candidates come from IDF-weighted
**compound keys** (token pairs such as house-number × street word, house-number × name word, name-word pairs) searched
within groups of equal country string, in both directions (top-k S2/S3 per S1 and top-k S1 per S2/S3 record). A cheap
learned pruning stage keeps **~6 candidates per Source 1 entity**. A LightGBM matcher uses script-agnostic string,
IDF, house-number, multilingual-embedding and learned "noise-word" features, and a one-owner-per-record decision rule.
We validated with a **test-density cross-validation** we built after finding that ordinary hold-outs hide the main
error class.

---

## 2. Methodology

### 2.1 Problem Analysis
- Scale: 2.2M S1 × 10.3M S2/S3 (train), 1.7M × 10M (test) → nothing quadratic; ~linear sparse retrieval.
- True pairs always share the exact `country` string; no S2/S3 record belongs to two S1 entities (one-owner rule).
- Singletons are 5.6%; non-singletons have 3.67 matches on average (max 11) → recall matters as much as precision.
- **Names repeat across entities**: identical normalized names are a true match only 34% of the time; the data
  contains "sibling" entities (same address, name differing by one word) and unmatched near-copy distractors
  (same name, house number shifted by a few units). Addresses — above all house numbers — decide.
- Noise: typos, digit/letter swaps (`Recyc1ing`), dropped or appended words ("Center", "Services", `[Inc]`, `#8036`),
  web-style names (`supremepackaging.com`), word reordering, abbreviations (`St`/`Street`, `VA`/`Virginia`), leading
  zeros and glued digits in numbers (`001954`, `26th`), empty addresses (3.3% of S2/S3), and names/addresses in
  Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam, Odia, Gurmukhi (~14% of true pairs have a
  non-Latin S2/S3 name; S1 is always Latin).
- **Density matters**: a random 20% hold-out keeps only 20% of the other entities' records, so each S1 faces far fewer
  siblings/distractors than in test. Our first model scored 0.992 on such a hold-out but 0.949 on the leaderboard.

### 2.2 Solution Strategy
**Approach Type:** Blocking + learned pruning + GBDT classifier + assignment decision rule
**Core Innovation:** (1) compound-key bidirectional IDF blocking; (2) test-density cross-validation (out-of-fold over
the full training set) and leave-one-country-out at density as the unseen-country (France) proxy; (3) country-
agnostic features that transfer to an unseen country: IDF within each country-string group, script-agnostic
romanization from Unicode character names, unsupervised "noise-ness" of words (S1 is clean, S2/S3 are noisy).

---

## 3. Candidate Generation (Blocking)
- **Normalization:** lowercase, Latin accent stripping, punctuation → space; learned native-script → Latin token map
  (competitive linking on train true pairs), web-name splitting, joined initials; digits split from letters and
  leading zeros stripped; address abbreviations and junk tokens (`r`→`rue`, `st`→`saint`, `N°`) canonicalized from
  label-free statistics mined on each country-string group's own records, so the unseen country's forms are learned from
  its data rather than from train (train says `st`→`street`).
- **Blocking keys:** within each group of equal `country` string, IDF-weighted cosine over single tokens **and token
  pairs**: name-word pairs, adjacent address bigrams, house-number × address-word, house-number × name-word. Pairs are
  rare even when each word is common, and survive word reordering. Features with in-group document frequency > 1000
  are not indexed (they still count in the norms).
- **Two directions:** top-30 S2/S3 per S1, plus top-3 S1 per S2/S3 record (every record has at most one owner, so its own
  best S1 is a strong candidate). Sparse matrix products in chunks; all three channels take ~18 min per full dataset on 4 processes.
- **Learned pruning:** a LightGBM of at most 60 trees on 16 cheap features (incl. each channel's score and rank) keeps, per S1, candidates with score ≥ 0.005 (≤ 10);
  the final model and `candidate_pairs.tsv` see only survivors.
- **Channels:** (1) compound-key IDF cosine (forward top-30, reverse top-3 with a per-entity cap); (2) a name channel
  for S2/S3 records without an address (their dominant failure mode); (3) a core-name × address-word channel for
  records whose house number is perturbed. Per-entity cap of 40 before pruning.
- **Candidate pairs generated:** ≈ 35 per S1 before pruning (77.4M on the full training set), **≈ 6.9 per S1 after
  pruning** on test (France 7.2, India 6.9, US 6.5) — `candidate_pairs.tsv` is the pruned set.
- **How we ensured true matches were not lost:** recall measured at test density on the full training set (not on a
  sparse hold-out): **98.9% of true pairs retrieved** (97.9% with the compound channel alone; 97.0% before the learned transliteration map); the pruning stage
  keeps 99.9% of them.

---

## 4. Matching Model
**Features used:**
- Name: rapidfuzz ratio / token-set / token-sort / partial / Jaro-Winkler, de-spaced ratio (web names), IDF-weighted
  token overlap, IDF mass only on one side, rarest shared token; boilerplate-free "core" tokens (tokens common within
  their own country group dropped, learned from the data); consonant skeletons of romanized names (script-agnostic:
  "रेड कंसल्टेंट्स" and "red consultants" → "rd knsltnts").
- Address: token-set / partial / ratio, IDF-weighted word overlap, house numbers (shared count, Jaccard, first-number
  equality, dropped-digit match, digit edit distance, log numeric difference, conflict / missing flags).
- Embeddings: cosine of multilingual-e5-small (MIT, 118M params) embeddings of name and name+address,
  contrastively fine-tuned on the train true pairs only (in-batch negatives, `src/e5_finetune.py`).
- Noise words: learned from train true pairs, how often each word is an added / dropped noise word; unexplained added /
  dropped words; unsupervised noise-ness (word frequency in S2/S3 vs S1 names within the country group).
- Name ambiguity: number of S1 entities in the same country group sharing the record's core / full name, the S1's own
  name uniqueness and supply of same-named records, address token counts (for name-only records the name is the only
  evidence; when several S1 share it, F0.5 prefers leaving the record unassigned).
- Context: rank / gap / margin of key signals within the S1 entity's candidates and within the S2/S3 record's
  competing S1 entities.

**Model type:** LightGBM (binary, early stopping on held-out S1 groups; no class re-weighting).
**Threshold selection method:** macro-F0.5 curves on test-density out-of-fold scores, in-country and leave-one-
country-out. After the one-owner assignment (each S2/S3 record goes only to its best-scoring S1), the in-country curve
is flat between 0.7 and 0.8 while the unseen-country curve keeps rising (the model is overconfident on a country it
never saw) → **0.8**, chosen on validation data only.

---

## 5. Results & Error Analysis
- **F_0.5 Score (macro):** test-density out-of-fold CV over the full training set (2.2M S1): 0.9696 → 0.9789 →
  0.9828 → 0.9841 → 0.9872 → **0.9875** (final); leave-one-country-out 0.905 → 0.923 → 0.934 → **0.935** (0.938 at
  the final threshold 0.8).
  Public leaderboard: 0.949 → 0.969 → 0.978 → 0.979 → [final]. A 20% random hold-out reported 0.992 for the first
  model — it hides siblings/distractors, so we stopped using it for decisions.
- **Unseen country (France) — what validation could not show, and what we measured without labels.** Our test-density
  CV contains only US and India. On the test set, label-free statistics of our US/India predictions reproduce the CV
  exactly (share of predicted pairs with conflicting house numbers 6.18% vs 6.07% US, 2.16% vs 2.15% India; share of
  empty-address records assigned 54–60% in both), so the leaderboard gap is France (15% of test entities). France behaves
  differently in two measurable ways: (1) its business names come from a small vocabulary — 56 words (club, amicale,
  école, maison, centre, comité …) occur in more than 1% of France names, so 78% of France names keep at most one
  "rare" word (US 8%) and any frequency-based notion of the distinctive part of a name degenerates; (2) it has 5–8×
  more same-address candidates with a different name per entity (co-located businesses and "siblings" that differ by
  one word). A word-level noise model learned from US/India labels cannot tell a sibling ("DD Musique" at the address
  of "DD Comite") from a noisy copy ("DD Comite SARL"). We therefore learn noise-word statistics **without labels inside
  each country group** from likely copies mined among its own candidates (same house numbers, similar street and name):
  on France test data the mined rates put "holding" (0.89) and "participations" (0.97) as frequently added noise and
  "club" (0.02), "comité" (0.01), "musique" (0.01) as stable words whose disagreement signals a different business.
  We also tried self-training (pseudo-labels from the model's own confident scores): on the leave-one-country-out proxy
  it raised recall but reinforced confident false merges (0.938 → 0.933), so it is not used.
- **Common false positives (wrong merges):** unmatched near-copies with a shifted house number; siblings at the same
  address whose name differs by one word; non-Latin names of a different company at the same address.
- **Common false negatives (missed matches):** S2/S3 records with an empty address and a common name (ambiguous between
  same-named entities); records never proposed by blocking (empty/short addresses, S1 entities with many matches);
  heavily perturbed names with no address.

---

## 6. Conclusion
Scalable sparse compound-key retrieval plus a feature-rich GBDT solves most of the problem; the decisive steps were
measuring at test density (which exposed siblings and distractors that ordinary hold-outs hide) and making every
feature country- and script-agnostic so it transfers to the unseen country.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: `src/run.py` (CLI), `src/ber/pipeline.py` (stages incl. pruning),
`src/ber/normalize/` (basic, p2_translit, p4_tcanon), `src/ber/blocking/` (p1_compound, p3_compound_rc, p3_chargram, p3_nameaddr), `src/ber/features/` (p1_match, p1_extra,
p1_romanize, p1_e5, p1_noise, p1_ambig), `src/ber/scorers/p1_lgbm.py`, `src/ber/deciders/p1_assign.py`, `src/ber/eval/` (metric,
splits). Reproduce: `python src/run.py test -c configs/best.yaml` (see README.md) → both TSVs.

### B. Additional Results
| iteration | main change | dense CV | LOCO | public LB |
|---|---|---|---|---|
| 1 | compound-key blocking + LightGBM (300-round cap) + one-owner rule | 0.9696 | 0.905 | 0.949 |
| 2 | + IDF within country groups, core tokens, romanized skeletons, number edit distance, uncapped LightGBM | 0.9789 | 0.923¹ | 0.969 |
| 3 | + transliteration map, pretrained E5 cosines | 0.9831 | — | not submitted |
| 4 | + noise-word features, learned pruning (~6.5 candidates/S1) | 0.9841 | — | 0.978 |
| 4b | + E5 fine-tuned on train pairs (mean with 4) | — | — | 0.979 |
| 5 | + 3-channel blocking (recall 98.9%), name-ambiguity features | 0.9872 | 0.934 | [final] |
| final | + target canonicalization (`p4_tcanon`), threshold 0.8 | 0.9875 | 0.935 | [final] |

Dense CV = 2-fold out-of-fold macro F0.5 over all 2.2M train S1 at test density; LOCO = train on one country, score the
other (unseen-country proxy). ¹ measured on the same features without the final threshold/uncapped model (T08_g).
