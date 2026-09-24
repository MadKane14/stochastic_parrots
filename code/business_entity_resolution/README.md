# Business Entity Resolution — team stochastic_parrots

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the challenge data.

## Environment
```bash
conda create -n ber python=3.11 -y && conda activate ber
pip install -r requirements.txt
```

## Data
Point `BER_DATA_DIR` at the dataset folder containing `train/` and `test/` (official file names):
```bash
export BER_DATA_DIR=/path/to/dataset        # Windows PowerShell: $env:BER_DATA_DIR="C:\path\to\dataset"
export BER_ARTIFACTS=/path/to/scratch       # caches and outputs (large)
export BER_MEMBER=repro
```

## End-to-end (data → blocking → matching → output)
```bash
python src/run.py test -c configs/best.yaml
```
This trains on all labelled training data, generates candidates for the test set (blocking), scores them,
applies the decision rule, validates the files, and writes both TSVs to
`$BER_ARTIFACTS/submissions/<run_id>/`. Copy them to `output/`.

## Validation score
```bash
python src/run.py run -c configs/best.yaml --split val_v1      # grouped 80/20 split by Source 1 entity
python src/run.py run -c configs/best.yaml --split xc_India    # train without India, evaluate on India
```

## Code layout
- `src/run.py` — CLI (check / run / test / sweep).
- `src/ber/pipeline.py` — stage orchestration; `src/ber/interfaces.py` — stage contracts.
- `src/ber/normalize|blocking|features|scorers|deciders/` — stage implementations (selected in the YAML config).
- `src/ber/eval/` — metric (macro F0.5 per Source 1 entity, singletons included), validation splits, run harness.
- `configs/best.yaml` — final configuration.
