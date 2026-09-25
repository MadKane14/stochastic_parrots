# Research playbook — ER solutions from similar competitions (external research, Fri 25 Sep)

> Status: background reading compiled by the lead from public write-ups (Foursquare Location Matching 2022,
> Shopee Price Match 2021, ER papers). **Numbers from those write-ups are unverified.** Where anything here conflicts
> with our own EDA (`exploration/p1/eda_out.md`) or the organizers' rules, **EDA and rules win**.

## Reconciliation with our EDA and rules (read this first)
| Playbook assumption | Our data / rules | Consequence |
|---|---|---|
| Postcode/PIN keys are strong blocking keys | Numbers with ≥5 digits appear in < 9% of true pairs; PINs essentially absent | Use **any shared address number** (house/plot/unit numbers: shared on 78–83% of true pairs vs < 6% random) instead of postcode keys |
| Singletons are a big cliff | Singletons are only **5.6%**; non-singletons have **3.67** matches on average | Recall matters as much as precision; still keep an explicit "no match" gate |
| Exclusivity "verify on labels" | **Verified:** no S2/S3 id belongs to more than one S1 | Enforce one-owner assignment |
| Country-agnostic everything | True pairs share the exact country string **100%** | Grouping candidates by the country *string* (open set) is safe; never branch on specific values |
| `unidecode` for normalisation | unidecode is **GPL** | Use `anyascii` (ISC) or our own mapping |
| "Final model" licence | Organizers: **every** model (embedder, reranker, matcher, preprocessing) must be MIT/Apache-2.0, ≤ 8B params, offline, fine-tuned only on provided data; no hosted LLM APIs | Qwen3-8B (8.2B) is out; no LLM-generated synthetic data |
| Non-Latin names | S2 ~5% Devanagari + ~4% other Indic scripts (e.g. Odia); S3 ~3% + ~2%; S1 is always Latin | Address channel must carry cross-script pairs; multilingual embeddings help |

## 1. Architecture that won the closest analogue (Foursquare 2022, per-id IoU metric)
retrieve → rerank: **multi-strategy blocking union** (char n-gram TF-IDF on name and name+address, multilingual
embedding kNN, numeric keys) → **LightGBM pairwise matcher** on 50–120 country-agnostic similarity features →
optional multilingual **cross-encoder** (score fed back as a GBDT feature) → **metric-aware post-processing**.
- 1st place kept ~40 candidates/id at a 0.979 recall ceiling; GBDT alone CV 0.875, cross-encoders + blend 0.911.
- Most portable idea: **group-relative features** per query (rank, p / p_max, sim − max sim in group) — and the same
  from the reverse side (per S2/S3 record over its S1 candidates).
- 7th place: asymmetric **containment** similarities (char precision/recall of one string in the other), similarities on
  **digits-only** extractions of name and address, 3 normalisation variants per field, sample weights penalising FPs.
- Graph post-processing (union-find / neighbours-of-neighbours) gave large gains there; our cheap equivalent is
  S2↔S3 transitivity features.
- Old-school TF-IDF blocking is strong (Sparkly, PVLDB 2023: top-k TF-IDF beat 8 learned blockers).
- Fine-tuned PLM matchers transfer poorly to unseen domains (Peeters et al. 2023) → hand-built, language-agnostic
  features are our France insurance; cross-encoders are a marginal add-on.

## 2. Normalisation ideas (own dictionaries = domain knowledge, allowed; disclose in the methodology)
- NFKC → lowercase → accent strip → `&`/`et` → `and` → punctuation → collapse "p v t", "l.t.d", "s.a.r.l.".
- **Legal forms**: strip into a separate field (keep "forms equal / same family" as features, don't just delete).
  US: inc, incorporated, corp, corporation, co, company, llc, ltd, limited, lp, llp, pllc, pc, dba.
  IN: pvt, private, ltd, limited, llp, opc, "& sons", "& co". FR: sarl, sas, sasu, sa, eurl, snc, sci, scop, selarl,
  "et cie", "& fils", société, ste, ets/établissements.
- DBA/trade names: split on dba, d/b/a, t/a, "trading as", aka, parentheses, " - "; similarity = max over parts.
- Acronym flag (initials(a) == compact(b)); abbreviation table (intl, mfg, svc(s), bros, ent, mkt, hosp) **plus
  abbreviations mined from our own true pairs** (token pairs co-occurring in matches).
- Transliteration/typos: phonetic keys (Double Metaphone/Soundex) and an Indic **skeleton key** (drop vowels after
  the 1st letter, collapse doubles, w→v, ph→f, z→j, q→k, ck→k; Laxmi/Lakshmi, Shree/Sri/Shri, Agarwal/Aggarwal).
- Web-style names seen in our data: `delhiconsulting.com`, `cequee1ement.com` (digit-for-letter) → strip TLD, split
  concatenations, map 1→l / 0→o in a variant.
- **IDF weighting** over all names (train + test, names/addresses only, no labels = transductive, disclose):
  IDF-weighted Jaccard/cosine, rare-token-matched flag, **IDF sum of mismatched tokens** (kills "Acme Bakery" vs
  "Acme Robotics").

## 3. Address ideas
- Numbers: set Jaccard, first-number equality, conflict count (both have numbers, disjoint), "missing in one" as its
  own state; unit/suite/plot numbers; string sims on digits-only extraction.
- Street-type canonicalisation: US st/rd/ave/blvd/dr/ln/hwy/ste/apt/fl; IN rd, marg, nagar, colony, sector, opp/nr,
  behind, chowk, gali, mohalla, cross, main; FR r/rue, av, bd, pl, all, imp, ch, rte, qu, fg, za/zi/zac, bp, cedex(+digits),
  stopwords de/du/des/la/le/l'. State names/abbreviations (VA ↔ Virginia, TG ↔ Telangana; also Devanagari state names).
- Landmarks ("near/nr/opp/behind/beside/next to …") overlap as a separate feature from street overlap.
- **Containment** similarity (shorter address inside longer) so missing components aren't penalised like conflicts.
- libpostal: gray area (trained on OSM/OpenAddresses) → **don't use**.

## 4. Models that satisfy the rules (verify licence + params on the model card before use)
multilingual-e5-small/base/large (MIT), paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0), bge-m3 (MIT),
bge-reranker-v2-m3 (Apache-2.0), xlm-roberta-base/large (MIT), mdeberta-v3-base (MIT), LaBSE (Apache-2.0),
Qwen3-0.6B/1.7B/4B (Apache-2.0), Qwen2.5-0.5B/1.5B/7B (Apache-2.0), Mistral-7B (Apache-2.0), Phi-3/3.5-mini (MIT).
Not allowed: Llama, Gemma, Qwen2.5-3B, Qwen3-8B (8.2B), CC-BY-NC rerankers, Zingg (AGPL).
Practical picks: blocking = multilingual-e5-base (`query: ` prefix both sides) or multilingual MiniLM; cross-encoder =
fine-tune bge-reranker-v2-m3 or xlm-roberta-base on serialized pairs (`name: … [SEP] address: …`), 1–2 epochs, len 128,
hard negatives from our candidates; feed its OOF score to the GBDT.

## 5. The metric (macro F0.5 per S1, singletons included)
- 1 true match: exact → 1.0; right + 1 wrong → 0.556; nothing / wrong → 0.0. 2 true: 1 right → 0.833.
- Analytic starting points (re-tune on OOF with the exact scorer): predict top-1 if calibrated p ≳ 0.5; add extra
  candidates only if q ≳ 0.73 (+0.167 if right vs −0.444 if wrong). *With our 3.67 matches/entity the recall side is
  larger than in this 1–2-match example — tune, don't assume.*
- Best: per S1, sort candidates by p and choose the prefix k maximising **expected F0.5** under Bernoulli(p_i)
  (exact enumeration for small k or ~200 Monte-Carlo draws).
- Calibrate: OOF probabilities with GroupKFold by S1, then isotonic/Platt. Avoid `is_unbalance` (inflates p).
- Entity-level "has any match" classifier (top-1 p, top-2 p, gap, #cands > 0.3, max name/addr sim, rarity).
- Exclusivity: each S2/S3 record → its argmax S1 only; reverse-rank features for a second-pass GBDT.
- Transitivity: for (S1, S3-y) use max over S2-x of p(S1,S2-x)·sim(S2-x,S3-y).

## 6. France
No country one-hot / country branches; features that mean the same in any language (numbers, char n-grams, IDF
overlap, embedding cosine); IDF fitted on all records incl. test; leave-one-country-out CV (`xc_India`, `xc_US`);
multilingual encoders only; hand French dictionaries; sanity-check France's predicted-empty rate, matches/S1 and top-1 p
distribution against US/India before submitting. Pseudo-labelling only if leave-one-country-out shows a gain.

## 7. Impact ordering (from the playbook)
1 metric-aware decision rule + no-match gate + exclusivity · 2 blocking recall · 3 IDF / numeric / group-relative
features · 4 embedding kNN + cosine · 5 cross-encoder score as feature · 6 S2↔S3 transitivity · 7 pseudo-labels.
