# How to reproduce the report results

All commands below are run from the **repository root**. We always tune on
validation and evaluate once on test — the test set is never used for model
selection.

The repo already contains the configs that produced the report numbers, so you
can regenerate results without re-running the long searches:

| File | What it outputs |
|------|-----------------|
| `configs/tuned/*.yaml` | Individual model hyperparameters |
| `configs/mixed_hybrid_best.yaml` | Mixed Hybrid quotas / order |
| `configs/hybrid_regression_best.yaml` | Hybrid Regression mix + logistic params |
| `configs/hybrid_feature_aug_best.yaml` | Feature Augmentation target / sources |
| `configs/rerankers/<base>.yaml` | Per-base λ (and Fairness extras) |

Outputs land under `outputs/` (gitignored).

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `ray` fails to install on a recent Python, skip it — none of the scripts
below need it:

```bash
grep -v '^ray' requirements.txt | pip install -r /dev/stdin
```

The dataset is MovieLens 100K (with plot descriptions) under `dataset/ml-100k/`.
Shared split settings live in `configs/base.yaml` (seed 2020, ratings ≥ 3,
80/10/10 per user). Must keep the file once models are trained, as otherwise the
exports will no longer share a split.

---

## Recommended path (our tuned configs)

This regenerates the numbers in the report without re-searching.

### Individual models and baselines

```bash
python run_model.py Pop Random \
  UserKNN ItemKNN EASE SLIMElastic \
  BPR LightGCN SimpleX RecVAE ContentBased
```

Uses `configs/tuned/<model>.yaml` when present. Writes
`outputs/<model>/{scores.npz,recs_valid.tsv,recs_test.tsv,...}`.

To re-tune individuals (Task 1.2) instead:

```bash
python run_tuning.py UserKNN ItemKNN EASE SLIMElastic \
  BPR LightGCN SimpleX RecVAE ContentBased
python run_model.py UserKNN ItemKNN EASE SLIMElastic \
  BPR LightGCN SimpleX RecVAE ContentBased
```

### Hybrids

```bash
python run_mixed_hybrid.py
python run_hybrid_regression.py
python run_hybrid_feature_aug.py
```

Writes `outputs/MixedHybrid/`, `outputs/HybridRegression/`, and
`outputs/HybridFeatureAug/` (including `recs_test.tsv`).

To re-tune hybrids (Task 1.5; this takes a while):

```bash
python tune_mixed_hybrid.py            # ~15 min
python tune_hybrid_regression.py       # tens of minutes
python tune_hybrid_feature_aug.py      # ~hours
```

### Evaluation metrics (Tasks 2.1 / 2.2)

```bash
python -m unittest discover -s tests/evaluation -v
python run_evaluation.py --stage test
```

Writes `outputs/evaluation/test_metrics.csv` (accuracy + beyond-accuracy).

### Hybrid contribution / ablation (Tasks 2.3 / 2.4)

```bash
python analyze_hybrids.py
# or one hybrid only:
# python analyze_hybrids.py --hybrids regression
# python analyze_hybrids.py --hybrids mixed
# python analyze_hybrids.py --hybrids feature_aug
```

Writes analysis tables under `outputs/` (coefficients, Mixed attribution,
FA Shapley / pseudo-items, leave-one-out ablations retuned on validation).

### User / item groups (Task 2.5)

```bash
python run_group_analysis.py --stage test
# optional notebook for the report table:
# jupyter notebook task2_5_group_analysis.ipynb
```

Writes:

- `outputs/evaluation/test_user_group_analysis.csv`
- `outputs/evaluation/test_item_group_analysis.csv`
- (notebook) `results/task2_5_group_table.csv`

### Rerankers (Tasks 3.1–3.4)

Tune λ on validation (within a 5% NDCG budget), apply once on test, then analyse:

```bash
python tune_rerankers.py
# optional subset: python tune_rerankers.py --bases RecVAE MixedHybrid

python run_rerankers.py
python run_rerank_analysis.py
```

Writes `configs/rerankers/<base>.yaml`, `outputs/Rerank_<name>/<base>/`,
and `outputs/evaluation/rerank_*.csv` (plus figures / summary).

---

## Summary of commands

After setup, with the configs already in the repo:

```bash
python run_model.py Pop Random \
  UserKNN ItemKNN EASE SLIMElastic \
  BPR LightGCN SimpleX RecVAE ContentBased

python run_mixed_hybrid.py
python run_hybrid_regression.py
python run_hybrid_feature_aug.py

python run_evaluation.py --stage test
python analyze_hybrids.py
python run_group_analysis.py --stage test

python tune_rerankers.py
python run_rerankers.py
python run_rerank_analysis.py
```

---

## Which script for which report task

| Report task | What to run |
|-------------|-------------|
| 1.1 / 1.2 Individuals | `run_model.py`, `run_tuning.py` |
| 1.3 Hybrid Regression | `recbole/model/.../hybrid_regression.py`; fit via `tune_` / `run_hybrid_regression.py` |
| 1.4 Mixed + Feature Aug | `mixed_hybrid.py`, `hybrid_feature_aug.py` + tune/run scripts |
| 1.5 Hybrid tuning | `tune_mixed_hybrid.py`, `tune_hybrid_regression.py`, `tune_hybrid_feature_aug.py` |
| 2.1 Metrics | `recbole/utils/evaluation.py`, `tests/evaluation/`, `run_evaluation.py` |
| 2.2 Test comparison | `run_evaluation.py` |
| 2.3 / 2.4 Hybrid analysis | `analyze_hybrids.py` |
| 2.5 Groups | `run_group_analysis.py`, `task2_5_group_analysis.ipynb` |
| 2.6 Insights | from 2.2–2.5 (no extra script) |
| 3.1–3.4 Reranking | `tune_rerankers.py`, `run_rerankers.py`, `run_rerank_analysis.py` |

---

## Tests

```bash
python -m unittest discover -s tests/evaluation -v
python -m unittest discover -s tests/hybrid -v
```
