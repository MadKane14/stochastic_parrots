# Questions for the organizers (send via the Google Form in the guidelines)

Status: _not yet sent_. Record answers here + in STATE.md "decisions" when they arrive.
Until answered, we take the conservative reading (in brackets).

## Models and licences
1. **8B limit scope.** Does "final model up to 8B parameters" apply to each model or to the whole pipeline combined
   (e.g. a 7B LLM re-ranker + a 0.5B embedding model + GBDT)? Are ensembles allowed? [each model ≤ 8B and total ≤ 8B]
2. **Counting parameters.** Is a model like Qwen3-8B (≈8.2B total incl. embeddings) within "up to 8B"? [no]
3. **Licence scope.** Does the MIT/Apache-2.0 requirement cover every pretrained model used anywhere (embeddings,
   tokenizers, rerankers), or only the final matching model? What about models used only offline during
   training (e.g. a teacher that labels pairs)? [all models, including offline]
4. Are **pretrained weights downloaded from Hugging Face** (MIT/Apache) acceptable under "no external data"? [yes]

## Data use
5. Is **LLM-generated synthetic training data** allowed (e.g. a local Apache-licensed LLM generating noisy variants
   of training names/addresses, or French-style variants)? Is rule-based augmentation (typos, abbreviations) OK? [rule-based yes, LLM no]
6. May we use **hand-written domain knowledge** in code — abbreviation lists (St→Street, Pvt→Private), legal-suffix
   lists, common French address terms (rue, bd, cedex) — or does French-specific handling count as "hard-coding
   countries"? [generic lists yes; no country-conditional logic]
7. May we fit **unsupervised statistics on the test records** (TF-IDF vocabulary/IDF, clustering, self-training /
   pseudo-labels on test)? [unsupervised stats yes, pseudo-labels no]

## Evaluation and submission
8. Which submission is used for the **private leaderboard**: the last one, the best public one, or one we select? [ask!]
9. How is the public/private split made — random over S1 entities? Does France appear in both?
10. Methodology doc: guidelines say **1–2 pages**, the problem statement says **no page limit** — which applies?
11. Do the **5 submissions/day** reset at 00:00 IST? Do failed-validation uploads count?
12. Must the zip include **trained weights**, or is code that retrains from the data enough? Any runtime/hardware
    limit for the reproduction check?
13. Can an S2/S3 record be matched to more than one S1 entity (S1 is "deduplicated")? Can a record match an S1
    entity with a different `country` label?
