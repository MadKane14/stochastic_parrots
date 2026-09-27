# From 0.983 to 0.99+: what we got wrong, what it would take, and the odds (p3, Sun 27 Sep, for the lead)

Top of the public LB is ≈ 0.993, so 0.99+ is achievable on this data. Our 0.983 is a limit of our model, not of the data.

## 1. The arithmetic
- **Composition:** LB ≈ 0.850 × (US/India) + 0.150 × France.
  - Test S1 shares: France 0.1498, India 0.4675, US 0.3827.
  - Source: `results/p1/20260927-081609_p1_FINAL_best_test_a7adec.json` test_stats.
- **Now:**
  - US/India ≈ 0.987. Test statistics match test-density CV (STATE D16/D20).
  - France ≈ 0.96, implied (D20).
  - Together: 0.983.

| US/India | France | LB |
|---|---|---|
| 0.987 | 0.960 | 0.983 (now) |
| 0.990 | 0.970 | 0.987 |
| **0.992** | **0.980** | **0.990** |
| 0.993 | 0.985 | 0.992 |

**Reaching 0.99 needs both levers:**
- **US/India +0.005.** That removes ~40% of our in-country loss (0.01252, `exploration/p1/loss_breakdown_out.txt`).
- **France +0.02.** That halves France's error.

**Measured in-country ceilings** (same file):

| fix | gain |
|---|---|
| perfect blocking | +0.0038 |
| every true pair inside the candidates found | +0.0075 |
| zero false merges | +0.0015 |

So US/India +0.005 means recovering missed pairs, both unproposed and proposed-but-rejected, not better precision.

## 2. Mistakes we made (with evidence)
1. **We judged the first day on sparse validation splits.** val_v1@0.1 said 0.992 while the LB said 0.949 (I-08, I-15).
   Test-density CV only existed from Saturday morning, and blocking work before then was measured optimistically.
2. **Blocking was the largest loss for too long, and its main hole was found late.**
   - Blocking misses were 0.0129 of 0.030 at T03.
   - Address-less right records were missed 14.7% of the time (vs 1.5%), but that was found only Saturday evening
     (`exploration/p1/blk_miss_profile_T10.md`).
   - The three-channel blocking (recall 0.979 → 0.989) entered the stack only on Sunday morning (T17).
3. **The matcher is strictly pairwise.** It scores (S1, one record) and never uses the entity's other copies.
   - The largest remaining in-country loss is entities with an address-less copy: 14% of S1, **55% of the loss**.
   - 97.9% of those address-less copies have a sibling copy of the same entity WITH an address (22% with the identical
     name) — `exploration/p3/empty_addr_siblings_out.md`.
   - The information to resolve them exists in the data; we never looked across copies.
   - The Foursquare-matching winners' graph post-processing is the standard form of this.
4. **Name noise is handled by hand-built similarity features, not a learned text model.**
   - 70% of address-less copies have a name that equals NO S1 name exactly (same file).
   - We used E5 only as bi-encoder cosines. The 2nd place in Kaggle's Foursquare Location Matching added a BERT
     cross-encoder over the final candidates.
5. **No France proxy that behaves like France.**
   - LOCO fails on shifted-house-number near-copies; France fails on same-address siblings from a small name vocabulary
     (I-26..I-28).
   - LOCO misled the threshold (D14/D16).
   - Label-learned word rates carried US/India priors into France until T28 (I-29). That was found Sunday afternoon.
6. **Iteration speed was the real constraint.**
   - Only phoenix (128 GB) could run dense CV. Worker machines (30 GB) OOM-killed every stack with E5 or the new blocking
     (p3 T18, p4 T19), and GPU queues cost hours.
   - Most worker time went to proxies. The decisive runs were serialized on one machine.
7. **Decision rule:** one global threshold for every kind of pair. Address-less, name-only pairs and full pairs have very
   different calibration. Expected-F set selection was tried globally (D11) but never per entity class.

## 3. The plan, if we had the time (ranked by expected gain; the numbers are estimates, not measurements)
| # | work | targets | est. gain (LB) | effort |
|---|---|---|---|---|
| A | **Cross-copy (collective) resolution** | 55% of in-country loss, France siblings | +0.002 … +0.004 | 1–1.5 days |
| B | **Cross-encoder reranker** (MIT/Apache ≤ 8B, e.g. XLM-R/MiniLM-class) fine-tuned on train pairs over the ≤ 10 pruned candidates | name noise, siblings | +0.001 … +0.003 | 1 day + GPU |
| C | **Recall of the last 1.07% unproposed pairs** | blocking misses (19% of loss) | +0.001 … +0.002 | 0.5 day |
| D | **France without labels** | France error | +0.0015 … +0.003 | 1 day |
| E | **Class-specific decisions** | calibration | +0.0005 … +0.001 | 0.5 day |
| F | **Memory-lean dense CV** | iteration speed | enables everything above | 0.5 day |

**A. Cross-copy (collective) resolution:**
- Block right↔right: S2 vs S3 copies within the country group, with the same keys.
- Features per (S1, record): best similarity of the record to the S1's other confident copies; the address "borrowed"
  from a sibling copy; the agreement count.
- Then a group-level decision: an address-less copy follows its strongest sibling's owner.

**C. Recall of the last 1.07% unproposed pairs:**
- Propagate candidates through the copy graph (a sibling of an assigned copy becomes a candidate).
- Add the char-gram + address channel at pruned scale (T11_ca was +0.0036 recall, but too heavy unpruned).

**D. France without labels:**
- Agreement-based pseudo-labels: the members agree, no house-number conflict, high margin. Naive self-training failed
  (D17), so it must be guarded.
- A France-like proxy built from train records by the label-free canonicalization/sibling mechanics (rules only — no
  external data, per CLAUDE.md §2).
- Judged by label-free test diagnostics (number conflicts, churn), as in D16/D19.

**E. Class-specific decisions:** thresholds or expected-F set selection per entity class (address-less vs full), on
test-density OOF.

**F. Memory-lean dense CV:** float16 assembly and fewer copies, so 30 GB machines can run India/US group dense CV
(p3 T18 died at 31 GB in fold 1). Four machines in parallel instead of one.

**Summing the estimates:**
- In-country ≈ +0.004 … +0.009, i.e. US/India to ≈ 0.991–0.996 at best.
- France ≈ +0.01 … +0.02.
- That brackets **LB ≈ 0.988 … 0.993**.

## 4. Chances (judgement, not measurement)
- **Today:** very low (< 5%).
  - Code freeze 18:00, zip 21:00, one upload left.
  - Any of A–D needs features + retraining + dense CV, i.e. ≥ 4–6 h of phoenix time before the first honest number.
- **With 2–3 more days, same team and machines:** roughly even odds (≈ 30–50%) of 0.99.
  - A + B + C are the core. D decides whether France gets to ≈ 0.98.
- **What would raise the odds most:**
  - Item F first, so four machines iterate in parallel.
  - Then A, the largest measured loss with a measured unused signal.

## 5. The one thing still possible today (only if the lead wants to spend the last upload)
- **Rule:** a label-free post-processing of the saved #9 scores. An address-less copy that is (near-)identical by name to
  another copy assigned to S1 X, and has no assignment or a weaker one, is assigned to X.
- **Test it first** on the in-country OOF scores of #9 (dense CV, phoenix): accept only if in-country F0.5 does not drop.
- **Size:** at most ~22% of the address-less class (identical-name siblings), so ≈ +0.001 at best. It is not 0.99.
