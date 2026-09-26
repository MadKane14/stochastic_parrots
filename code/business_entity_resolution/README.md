# Business Entity Resolution — team stochastic_parrots

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the challenge data
(train/test TSVs only — no external data, no hosted APIs).

## Environment
Python 3.11, pinned versions in `requirements.txt` (one file). Verified with conda-forge:
```bash
conda create -n ber -c conda-forge python=3.11 numpy=2.2.6 pandas=2.3.3 scipy=1.15.2 scikit-learn=1.7.2 \
    rapidfuzz=3.14.1 "lightgbm=4.6.0=cpu*" pyarrow=21.0.0 pyyaml=6.0.3 pytorch=2.8.0 transformers=4.57.1 -y
conda activate ber            # or: pip install -r requirements.txt  (with the env's own Python 3.11)
```
Hardware used: 16-core CPU, 128 GB RAM (the pipeline stays below 64 GB), one 16 GB GPU (RTX A4000) for the encoder.
The encoder is `intfloat/multilingual-e5-small` (MIT licence, 117,653,760 parameters), downloaded once from the
Hugging Face hub (set `HF_HOME`); everything else runs offline.

## Data and working directories
```bash
export BER_DATA_DIR=/path/to/dataset          # folder containing train/ and test/ (official file names)
export BER_ARTIFACTS=/path/to/scratch         # caches, models, embeddings, outputs (~100 GB free recommended)
export BER_MEMBER=repro
export BER_THREADS=12                         # CPU threads for every library
```

## End-to-end (data → blocking → matching → output)
```bash
# 1) fine-tune the encoder on TRAIN true pairs only (~15 min on one GPU)
CUDA_VISIBLE_DEVICES=0 python src/e5_finetune.py --split full --out $BER_ARTIFACTS/models/ft_full --n 400000
# 2) train on all labelled data, block + score + decide the test set, validate, write both TSVs (~2.5 h; the first run
#    also embeds all train + test names/addresses with the encoder, ~1.5 h on one GPU, cached afterwards)
CUDA_VISIBLE_DEVICES=0 python src/run.py test -c configs/best.yaml
# 3) the two files are in $BER_ARTIFACTS/submissions/<run_id>/ — copy them to output/
```
Step 2 runs the organizers' validator (`utils/validate_submission.py`, if present next to the dataset) and our own
checks, and writes a results JSON (config, commit, timings, per-country prediction stats) under `results/`.

## Pipeline (configs/best.yaml)
1. **Normalize** (`p2_translit`): lowercase, Latin accent stripping, punctuation; native-script → Latin token map
   learned from train true pairs; web-style names split; digits split from letters, leading zeros stripped.
2. **Block** (`p3_compound_rc`, `p3_chargram`, `p3_nameaddr`): within each group of equal `country` string, IDF cosine
   over single tokens and token pairs (name-word pairs, address bigrams, house-number × word, house-number × name word),
   forward top-k per Source 1 entity and reverse top-k per S2/S3 record; a name channel for address-less records and a
   name × street-word channel; per-entity caps.
3. **Features** (`p1_match`, `p1_extra`, `p1_e5`, `p1_noise`, `p1_ambig`): string similarities, IDF overlaps computed
   within each country group, house-number agreement / edit distance, script-agnostic romanized skeletons, encoder
   cosines, learned noise-word statistics, name-ambiguity counts, ranks/margins within the entity and the record.
4. **Prune** (learned stage-1 LightGBM): keeps ~6 candidates per Source 1 entity — `candidate_pairs.tsv` is this set.
5. **Score** (`p1_lgbm`, LightGBM) and **decide** (`p1_assign`): each S2/S3 record goes to at most one Source 1 entity
   (its best), then probability ≥ 0.7.

## Validation used during development
- `python src/run.py run -c configs/best.yaml --split val_v1` — grouped split by Source 1 entity (frozen harness).
- Test-density cross-validation (out-of-fold over the full training set) and leave-one-country-out: see
  `exploration/p1/dense_cv.py` in the team repository (not needed for reproduction).

## Code layout
- `src/run.py` — CLI (check / run / test / sweep); `src/e5_finetune.py` — encoder fine-tuning.
- `src/ber/pipeline.py` — stage orchestration incl. pruning; `src/ber/interfaces.py` — stage contracts.
- `src/ber/normalize|blocking|features|scorers|deciders/` — stage implementations (selected in the YAML config).
- `src/ber/eval/` — metric (macro F0.5 per Source 1 entity, singletons included), validation splits, run harness.
- `configs/best.yaml` — final configuration; `configs/exp/` — experiment configurations.
