# DSAIT4335 Recommender Systems: Final Project

This repository is a fork of [RecBole](https://github.com/RUCAIBox/RecBole), extended for the course
project on MovieLens 100K. RecBole's original documentation follows [below](#about-recbole).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `pip` fails on `ray` (the pinned `ray<=2.6.3` has no build for recent Python versions), install
everything else instead. `run_model.py` and `run_tuning.py` do not need ray, but `run_recbole.py`
and `run_hybrid.py` do, because RecBole's `quick_start` imports it:

```bash
grep -v '^ray' requirements.txt | pip install -r /dev/stdin
```

All scripts are run from the repository root.

## Individual models (Task 1.1 / 1.2)

`run_model.py` trains the baselines, neighbourhood models and matrix factorisation, and writes every
model's results to `outputs/` in the same format. Hybrid, evaluation and reranking code can then work
from these files without re-running RecBole.

```bash
python run_model.py all                    # train + export every model in configs/models/
python run_model.py UserKNN ItemKNN BPR    # or just some of them
python run_model.py UserKNN --set k 20     # one-off override, e.g. for a quick experiment
python run_model.py BPR --no-tuned         # use the untuned defaults

python run_tuning.py UserKNN ItemKNN BPR   # grid search on the validation set (see below)
```

Rough CPU run times: KNN models and Pop take a few seconds each, and BPR takes about 10 s.
Tuning takes 20 s (UserKNN), about 1 min (ItemKNN) and about 5 min (BPR).

### Models

| Experiment    | RecBole model | What it is |
|---------------|---------------|------------|
| `Random`      | Random        | naive baseline |
| `Pop`         | TrainPop      | naive baseline: most-interacted items in the training set |
| `UserKNN`     | ItemKNN, `knn_method: user` | user-based neighbourhood CF, cosine similarity |
| `ItemKNN`     | ItemKNN, `knn_method: item` | item-based neighbourhood CF, cosine similarity |
| `AsymUserKNN` | AsymKNN, `knn_method: user` | user-based CF with asymmetric cosine |
| `AsymItemKNN` | AsymKNN, `knn_method: item` | item-based CF with asymmetric cosine |
| `BPR`         | BPR           | matrix factorisation trained with a pairwise ranking loss |
| `SLIMElastic` | SLIMElastic   | sparse item-item weights learned by ElasticNet regression (closed form, no epochs) |
| `RecVAE`      | RecVAE        | variational autoencoder over each user's interaction vector |
| `LightGCN`    | LightGCN      | embeddings smoothed over the user-item graph, BPR loss |
| `SimpleX`     | SimpleX       | user = own embedding + mean of history; cosine contrastive loss with 50 negatives |

To add a model, drop a YAML into `configs/models/` (it needs at least `model: <RecBole model name>`)
and optionally a grid in `configs/hyper/`.

### Configuration

Configs are applied in this order; later files override earlier ones:

1. `configs/base.yaml`: shared by every model. It sets the dataset, a random 80/10/10 split per user
   (seed 2020), full ranking evaluation and NDCG@10 as the validation metric.
   **Don't change this after models have been run**, or they will no longer share a split.
   `run_model.py` warns you if that happens.
2. `configs/models/<experiment>.yaml`: the model and its default hyperparameters.
3. `configs/tuned/<experiment>.yaml`: written by `run_tuning.py`, and used automatically when present.

Only ratings of **3 stars or more** are kept, and each one counts as a positive interaction
(`val_interval` in `configs/base.yaml`). 1- and 2-star ratings (17% of ML-100K) are dropped before
splitting, so they appear in no split and are never treated as hits.

### Tuning

`run_tuning.py <experiment>` tries every combination in `configs/hyper/<experiment>.yaml`, trains on
*train*, and selects by NDCG@10 on *valid*. The test set is never used for selection; test scores are
only logged in the CSV for reference. It writes:

- `outputs/tuning/<experiment>.csv`: every combination tried, best first
- `configs/tuned/<experiment>.yaml`: the best parameters

### Outputs (`outputs/`, not committed)

| File | Contents |
|------|----------|
| `split/{train,valid,test}.tsv` | `user_id, item_id, rating, timestamp`, with original MovieLens ids. Identical for all models. |
| `<experiment>/scores.npz` | `scores` [943 users x 1574 items; items with only 1-2 star ratings are dropped], `user_ids`, `item_ids`. Raw model scores for **every** pair; seen items are not masked. |
| `<experiment>/recs_valid.tsv` | top-100 per user, `user_id, item_id, rank, score`, with training items removed |
| `<experiment>/recs_test.tsv` | top-100 per user, with training **and** validation items removed |
| `<experiment>/metrics.json` | RecBole's own valid/test metrics, **as a sanity check only**. The project asks us to report our own metrics. |
| `<experiment>/config.yaml` | the hyperparameters actually used |

Models are trained on *train* only. Use `recs_valid` / validation scores to fit anything
(e.g. hybrid weights), and `recs_test` for final evaluation.

### Loading outputs in your own code

```python
from recbole.utils.experiment import load_split, load_scores, load_recs

train, valid, test = load_split()
S_user, user_ids, item_ids = load_scores("UserKNN")   # rows/cols are identical across models,
S_bpr, _, _ = load_scores("BPR")                      # so matrices can be combined directly
recs = load_recs("ItemKNN", "test")                   # DataFrame: user_id, item_id, rank, score
```

`recbole.utils.experiment` also exposes `CONFIG_DIR`, `OUTPUT_DIR` and `REPO_DIR`.

Scores are on different scales for different models (KNN: sums of similarities; BPR: dot
products; Pop: count / max count), so normalise them (e.g. per-user min-max or z-score) before
combining them in a *weighted / regression* hybrid.

### Changes to RecBole for this part

- **`Pop` uses our `TrainPop`** (`recbole/model/general_recommender/trainpop.py`), not RecBole's `Pop`.
  RecBole's Pop counts an item at most once per training batch (`cnt[item] = cnt[item] + 1` ignores
  repeats), and it also counts randomly sampled negatives, so its "popularity" took only about 50
  distinct values. TrainPop reads the true counts from the training interaction matrix and scores
  items the same way (count / max count). RecBole's `pop.py` is unchanged.
- **NumPy 2 support:** `recbole/config/configurator.py` referred to aliases removed in NumPy 2.0
  (`np.float_`, `np.complex_`, `np.unicode_`), so importing RecBole crashed.
- **LightGCN with recent SciPy:** `recbole/model/general_recommender/lightgcn.py` built its graph with
  `dok_matrix._update`, which SciPy >= 1.15 removed. It now builds the identical matrix as COO.
  NGCF, NCL, SpectralCF, GCMC, KGIN and MCCLK use the same call and are not fixed.
- `recbole/utils/experiment.py` restores the best checkpoint with `weights_only=False`, because
  `torch.load`'s default changed in PyTorch 2.6.

## Hybrid model: Mixed Hybrid (list mixing)

> This section covers **only the Mixed Hybrid**. The project has several hybrid models (for
> example the Logistic Regression weighted hybrid, `run_hybrid.py`); each is a separate
> implementation with its own scripts, config and results. The files of this one are
> `recbole/model/general_recommender/mixed_hybrid.py`, `tune_mixed_hybrid.py`,
> `run_mixed_hybrid.py`, `configs/mixed_hybrid_best.yaml`, `tests/hybrid/test_mixed_hybrid.py`
> and `outputs/MixedHybrid/`.

The Mixed Hybrid does **not** combine raw scores. It interleaves already-ranked recommendation
lists, deduplicates per user, and keeps the first `K` unique items.

Candidate sources: every model with tuned hyperparameters in `configs/tuned/`.

| Kind | Models |
|------|--------|
| neighbourhood / item-item | `ItemKNN`, `UserKNN`, `EASE`, `SLIMElastic` |
| latent factor | `BPR`, `LightGCN`, `SimpleX`, `RecVAE` |
| content | `ContentBased` (`CB` model, run through `run_model.py`) |

### Mixed Hybrid: strategy

Two mixers are implemented in `recbole/model/general_recommender/mixed_hybrid.py`:

1. **RoundRobin** — take rank 1 from each model, then rank 2, … skipping duplicates.
2. **Quota round-robin** — in each round take up to `quotas[model]` unseen items from that
   model's list, then repeat until `top_k` unique items are filled.

Example quotas `ItemKNN=2 BPR=1 ContentBased=1`:

```text
Round 1: ItemKNN, ItemKNN, BPR, ContentBased
Round 2: ItemKNN, ItemKNN, BPR, ContentBased
...
```

Deduplication is **per user**. If one source runs out of candidates, the others continue.


### Mixed Hybrid: tuning on validation, evaluating once on test

Primary selection metric: **NDCG@10** (same as `configs/base.yaml`). The test set is never used
for any choice.

```bash
python run_model.py RecVAE SLIMElastic EASE LightGCN BPR SimpleX ItemKNN UserKNN ContentBased
python tune_mixed_hybrid.py      # ~15 min with 4 worker processes
python run_mixed_hybrid.py
```

`tune_mixed_hybrid.py` does the following on the validation lists:

1. **Standalone + overlap analysis.** For every pair of models: the mean number of shared
   top-10 items per user, and the mean number of correct items one model finds that the other
   misses. Also, per model, the correct items no other model finds.
2. **Exhaustive quota search** over all 9 candidates: every quota vector with 0..10 items per
   model and a total of 1..10 (a total above 10 behaves like 10), i.e. 92,377 configurations,
   with models in order of standalone validation NDCG@10. A quota of 0 drops a model, so every
   subset of models is included, including several models of the same kind.
3. **Order search.** A model earlier in a round fills higher ranks, so for a chosen set of
   models every ordering x every positive quota vector is tried (up to 6 models).
4. **Choice of selection method.** Four methods pick a configuration from steps 2-3:
   `BestSingle` (best single model), `OnePerKind` (best model of each kind, then their best
   mix), `Greedy` (add models one at a time while NDCG@10 improves), `Exhaustive` (best
   overall). Searching more configurations fits more validation noise, so validation users are
   split into random halves 10 times; each method selects on one half and is scored on the
   other (20 folds). The method with the best mean held-out NDCG@10 is applied to the whole
   validation set to give the final configuration.

The search uses `mix_arrays` / `evaluate_arrays`, a vectorised version of
`mix_recommendations` / `evaluate_recommendations` (a test checks they give identical results).
The final configuration is also re-evaluated with the reference implementation.

Outputs (`outputs/MixedHybrid/`):

| File | Contents |
|---|---|
| `standalone_valid.csv` | validation metrics of every candidate |
| `overlap_shared_top10_valid.csv` | mean shared top-10 items per user, per pair of models |
| `overlap_extra_hits_valid.csv` | mean correct items the row model finds that the column model misses |
| `exclusive_hits_valid.csv` | correct items per user, and those no other model finds |
| `search_valid.csv.gz` | all 92,377 configurations of the exhaustive search |
| `greedy_path_valid.csv` | models added by the greedy method and the gain of each step |
| `order_search_valid.csv` | every ordering x quota vector for the final set of models |
| `split_half_valid.csv`, `split_half_summary_valid.csv` | the selection-method comparison |
| `recs_valid.tsv` | validation recommendations of the final configuration |
| `recs_test.tsv`, `test_comparison.csv`, `test_metrics.json` | written by `run_mixed_hybrid.py` |

`configs/mixed_hybrid_best.yaml` holds the final configuration, plus the configuration each
other method picked (`alternatives`), which `run_mixed_hybrid.py` reports on test for comparison
only.

### Mixed Hybrid: results (seed 2020, ML-100K, ratings >= 3)

**Overlap (validation).** The eight collaborative models share 4.7-7.6 of their top-10 items
per user (most similar: EASE/SLIMElastic 7.6, LightGCN/BPR 7.2). ContentBased shares only
about 0.5 with any of them; it finds the most items no other model finds (0.147 per user),
but few correct items in total (0.31 per user vs 1.35-1.53). Among the collaborative models,
LightGCN (0.116) and SimpleX (0.090) find the most items no other model finds; EASE finds
almost none (0.012), as SLIMElastic and RecVAE cover nearly the same items.

**Exhaustive search (validation NDCG@10, best configuration per number of models):**

| Models | Best configuration | NDCG@10 |
|---|---|---|
| 1 | RecVAE | 0.2500 |
| 2 | RecVAE=3 SLIMElastic=1 | 0.2539 |
| 3 | RecVAE=2 SLIMElastic=5 BPR=1 | 0.2540 |
| 4 | RecVAE=3 SLIMElastic=2 BPR=1 SimpleX=1 | 0.2537 |
| 5 | RecVAE=2 SLIMElastic=2 EASE=1 BPR=1 SimpleX=1 | 0.2537 |
| 6 | + LightGCN, ItemKNN | 0.2519 |
| 9 | all models (ContentBased=1) | 0.2380 |

Beyond 3 models nothing improves, and every best configuration with ContentBased is worse than
the same one without it.

**Selection methods (20 validation half splits, held-out NDCG@10):**

| Method | Held-out | Selection half | Optimism | vs BestSingle (paired) |
|---|---|---|---|---|
| BestSingle | 0.2466 | 0.2509 | 0.0043 | |
| OnePerKind | 0.2471 | 0.2548 | 0.0078 | +0.0005, better in 10/20 |
| Greedy | 0.2475 | 0.2584 | 0.0108 | +0.0010, better in 12/20 |
| **Exhaustive** | **0.2478** | 0.2586 | 0.0107 | +0.0013, better in 13/20 |

Exhaustive generalises best, so it is used, but the gain over a single model is small and
within the fold-to-fold noise. The larger "optimism" (selection minus held-out score) of the
searching methods shows how much of their validation gain is fitted noise.

**Final Mixed Hybrid configuration** (Exhaustive on the whole validation set): `BPR=1 SLIMElastic=4 RecVAE=5`,
in that order: BPR's top item, then SLIMElastic's top 4 new items, then RecVAE's top 5 new
items. Validation NDCG@10 = 0.2563 (RecVAE alone: 0.2500).

**Test (final Mixed Hybrid configuration evaluated once; the other rows are for comparison only):**

| Model | Recall@10 | MRR@10 | NDCG@10 | Hit@10 | Precision@10 |
|---|---|---|---|---|---|
| SLIMElastic | 0.2661 | 0.4593 | **0.2825** | 0.7381 | 0.1775 |
| EASE | 0.2640 | 0.4593 | 0.2813 | 0.7359 | 0.1757 |
| RecVAE | 0.2561 | 0.4421 | 0.2696 | 0.7222 | 0.1690 |
| SimpleX | 0.2479 | 0.4353 | 0.2585 | 0.7222 | 0.1595 |
| BPR | 0.2434 | 0.4277 | 0.2578 | 0.7190 | 0.1628 |
| LightGCN | 0.2509 | 0.4207 | 0.2571 | 0.7232 | 0.1641 |
| UserKNN | 0.2344 | 0.3972 | 0.2400 | 0.6925 | 0.1487 |
| ItemKNN | 0.2266 | 0.3966 | 0.2389 | 0.6903 | 0.1524 |
| ContentBased | 0.0417 | 0.0977 | 0.0450 | 0.2460 | 0.0323 |
| **Mixed Hybrid `BPR=1 SLIMElastic=4 RecVAE=5`** | **0.2667** | 0.4470 | 0.2778 | **0.7444** | 0.1742 |
| OnePerKind pick `RecVAE=3 SLIMElastic=1` | 0.2650 | 0.4482 | 0.2766 | 0.7402 | 0.1736 |

The Mixed Hybrid has the best Recall@10 and Hit@10 on test and beats RecVAE, the best single model on
validation (+0.008 NDCG@10). It does not beat SLIMElastic or EASE on NDCG@10/MRR@10: these rank
below RecVAE on validation but above it on test, so the validation-best model is not the
test-best one. The Mixed Hybrid sits between its sources.

### Mixed Hybrid: tests

```bash
python -m unittest discover -s tests/hybrid -p test_mixed_hybrid.py -v
```

---

# About RecBole

![RecBole Logo](asset/logo.png)

--------------------------------------------------------------------------------

# RecBole (伯乐)

*“世有伯乐，然后有千里马。千里马常有，而伯乐不常有。”——韩愈《马说》*

[![PyPi Latest Release](https://img.shields.io/pypi/v/recbole)](https://pypi.org/project/recbole/)
[![Conda Latest Release](https://anaconda.org/aibox/recbole/badges/version.svg)](https://anaconda.org/aibox/recbole)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-RecBole-%23B21B1B)](https://arxiv.org/abs/2011.01731)


[HomePage] | [Docs] | [Datasets] | [Paper] | [Blogs] | [Models] | [中文版]

[HomePage]: https://recbole.io/
[Docs]: https://recbole.io/docs/
[Datasets]: https://github.com/RUCAIBox/RecDatasets
[Paper]: https://arxiv.org/abs/2011.01731
[Blogs]: https://blog.csdn.net/Turinger_2000/article/details/111182852
[Models]: https://github.com/RUCAIBox/RecBole2.0/blob/main/model_list.md
[中文版]: README_CN.md

RecBole is developed based on Python and PyTorch for reproducing and developing recommendation algorithms in a unified,
comprehensive and efficient framework for research purpose.
Our library includes 94 recommendation algorithms, covering four major categories:

+ General Recommendation
+ Sequential Recommendation
+ Context-aware Recommendation
+ Knowledge-based Recommendation

We design a unified and flexible data file format, and provide the support for 44 benchmark recommendation datasets.
A user can apply the provided script to process the original data copy, or simply download the processed datasets
by our team.


<p align="center">
  <img src="asset/framework.png" alt="RecBole v0.1 architecture" width="600">
  <br>
  <b>Figure</b>: RecBole Overall Architecture
</p>

In order to support the study of recent advances in recommender systems, we construct an extended recommendation library [RecBole2.0](https://github.com/RUCAIBox/RecBole2.0) consisting of 8 packages for up-to-date topics and architectures (e.g., debiased, fairness and GNNs). 

## Feature
+ **General and extensible data structure.** We design general and extensible data structures to unify the formatting and
usage of various recommendation datasets.

+ **Comprehensive benchmark models and datasets.** We implement 94 commonly used recommendation algorithms, and provide
the formatted copies of 44 recommendation datasets.

+ **Efficient GPU-accelerated execution.** We optimize the efficiency of our library with a number of improved techniques
oriented to the GPU environment.

+ **Extensive and standard evaluation protocols.** We support a series of widely adopted evaluation protocols or settings
for testing and comparing recommendation algorithms.


## RecBole News
![new](/asset/new.gif) **02/23/2025**: We release RecBole [v1.2.1](https://github.com/RUCAIBox/RecBole/releases/tag/v1.2.1).

![new](/asset/new.gif) **11/01/2023**: We release RecBole [v1.2.0](https://github.com/RUCAIBox/RecBole/releases/tag/v1.2.0).

**11/06/2022**: We release [the optimal hyperparameters of the model and their tuning ranges](https://recbole.io/hyperparameters/index.html).

**10/05/2022**: We release RecBole [v1.1.1](https://github.com/RUCAIBox/RecBole/releases/tag/v1.1.1).

**06/28/2022**: We release [**RecBole2.0**](https://github.com/RUCAIBox/RecBole2.0) with **8 packages** consisting of **65 newly implement models**. 

**02/25/2022**: We release RecBole [v1.0.1](https://github.com/RUCAIBox/RecBole/releases/tag/v1.0.1).

**09/17/2021**: We release RecBole [v1.0.0](https://github.com/RUCAIBox/RecBole/releases/tag/v1.0.0).

**03/22/2021**: We release RecBole [v0.2.1](https://github.com/RUCAIBox/RecBole/releases/tag/v0.2.1).

**01/15/2021**: We release RecBole [v0.2.0](https://github.com/RUCAIBox/RecBole/releases/tag/v0.2.0).

**12/10/2020**: 我们发布了[RecBole小白入门系列中文博客（持续更新中）](https://blog.csdn.net/Turinger_2000/article/details/111182852) 。

**12/06/2020**: We release RecBole [v0.1.2](https://github.com/RUCAIBox/RecBole/releases/tag/v0.1.2).

**11/29/2020**: We constructed preliminary experiments to test the time and memory cost on three
different-sized datasets and provided the [test result](https://github.com/RUCAIBox/RecBole#time-and-memory-costs)
for reference.

**11/03/2020**: We release the first version of RecBole **v0.1.1**.

### Latest Update for SIGIR 2023 Submission

To better meet the user requirements and contribute to the research community, we present a significant update of RecBole in the latest version, making it more user-friendly and easy-to-use as a comprehensive benchmark library for recommendation. We summarize these updates in "**Towards a More User-Friendly and Easy-to-Use Benchmark Library for Recommender Systems**" and submit the paper to **SIGIR 2023**. The main contribution in this update is introduced below.

Our extensions are made in three major aspects, namely the models/datasets, the framework, and the configurations. Furthermore, we provide more comprehensive documentation and well-organized FAQ for the usage of our library, which largely improves the user experience. More specifically, the highlights of this update are summarized as: 

1. We introduce more operations and settings to help benchmarking the recommendation domain.

2. We improve the user friendliness of our library by providing more detailed documentation and well-organized frequently asked questions. 
3. We point out several development guidelines for the open-source library developers. 

These extensions make it much easier to reproduce the benchmark results and stay up-to-date with the recent advances on recommender systems. The datailed comparison between this update and previous versions is listed below.

|          Aspect           |            RecBole 1.0             |          RecBole 2.0           |                   This update                    |
| :-----------------------: | :--------------------------------: | :----------------------------: | :----------------------------------------------: |
|   Recommendation tasks    |            4 categories            |    3 topics and 5 packages     |                   4 categories                   |
|    Models and datasets    |     73 models and 28 datasets      |  65 models and 8 new datasets  |            94 models and 43 datasets             |
|      Data structure       | Implemented Dataset and Dataloader |         Task-oriented          |  Compatible data module inherited from PyTorch   |
|    Continuous features    |          Field embedding           |        Field embedding         |        Field embedding and discretization        |
| GPU-accelerated execution |       Single-GPU utilization       |     Single-GPU utilization     |      Multi-GPU and mixed precision training      |
|  Hyper-parameter tuning   |       Serial gradient search       |     Serial gradient search     | Three search methods in both serial and parallel |
|     Significance test     |                 -                  |               -                |               Available interface                |
|     Benchmark results     |                 -                  | Partially public (GNN and CDR) |      Benchmark configurations on 94 models       |
|      Friendly usage       |           Documentation            |         Documentation          |       Improved documentation and FAQ page        |


## Installation
RecBole works with the following operating systems:

* Linux
* Windows 10
* macOS X

RecBole requires Python version 3.7 or later.

RecBole requires torch version 1.7.0 or later. If you want to use RecBole with GPU,
please ensure that CUDA or cudatoolkit version is 9.2 or later.
This requires NVIDIA driver version >= 396.26 (for Linux) or >= 397.44 (for Windows10).

### Install from conda

```bash
conda install -c aibox recbole
```

### Install from pip

```bash
pip install recbole
```

### Install from source
```bash
git clone https://github.com/RUCAIBox/RecBole.git && cd RecBole
pip install -e . --verbose
```

## Quick-Start
With the source code, you can use the provided script for initial usage of our library:

```bash
python run_recbole.py
```

This script will run the BPR model on the ml-100k dataset.

Typically, this example takes less than one minute. We will obtain some output like:

```
INFO ml-100k
The number of users: 944
Average actions of users: 106.04453870625663
The number of items: 1683
Average actions of items: 59.45303210463734
The number of inters: 100000
The sparsity of the dataset: 93.70575143257098%
INFO Evaluation Settings:
Group by user_id
Ordering: {'strategy': 'shuffle'}
Splitting: {'strategy': 'by_ratio', 'ratios': [0.8, 0.1, 0.1]}
Negative Sampling: {'strategy': 'full', 'distribution': 'uniform'}
INFO BPRMF(
    (user_embedding): Embedding(944, 64)
    (item_embedding): Embedding(1683, 64)
    (loss): BPRLoss()
)
Trainable parameters: 168128
INFO epoch 0 training [time: 0.27s, train loss: 27.7231]
INFO epoch 0 evaluating [time: 0.12s, valid_score: 0.021900]
INFO valid result:
recall@10: 0.0073  mrr@10: 0.0219  ndcg@10: 0.0093  hit@10: 0.0795  precision@10: 0.0088
...
INFO epoch 63 training [time: 0.19s, train loss: 4.7660]
INFO epoch 63 evaluating [time: 0.08s, valid_score: 0.394500]
INFO valid result:
recall@10: 0.2156  mrr@10: 0.3945  ndcg@10: 0.2332  hit@10: 0.7593  precision@10: 0.1591
INFO Finished training, best eval result in epoch 52
INFO Loading model structure and parameters from saved/***.pth
INFO best valid result:
recall@10: 0.2169  mrr@10: 0.4005  ndcg@10: 0.235  hit@10: 0.7582  precision@10: 0.1598
INFO test result:
recall@10: 0.2368  mrr@10: 0.4519  ndcg@10: 0.2768  hit@10: 0.7614  precision@10: 0.1901
```

If you want to change the parameters, such as ``learning_rate``, ``embedding_size``, just set the additional command
parameters as you need:

```bash
python run_recbole.py --learning_rate=0.0001 --embedding_size=128
```

If you want to change the models, just run the script by setting additional command parameters:

```bash
python run_recbole.py --model=[model_name]
```

### Auto-tuning Hyperparameter 
Open `RecBole/hyper.test` and set several hyperparameters to auto-searching in parameter list. The following has two ways to search best hyperparameter:
* **loguniform**: indicates that the parameters obey the uniform distribution, randomly taking values from e^{-8} to e^{0}.
* **choice**: indicates that the parameter takes discrete values from the setting list.

Here is an example for `hyper.test`: 
```
learning_rate loguniform -8, 0
embedding_size choice [64, 96 , 128]
train_batch_size choice [512, 1024, 2048]
mlp_hidden_size choice ['[64, 64, 64]','[128, 128]']
```
Set training command parameters as you need to run:
```
python run_hyper.py --model=[model_name] --dataset=[data_name] --config_files=xxxx.yaml --params_file=hyper.test
e.g.
python run_hyper.py --model=BPR --dataset=ml-100k --config_files=test.yaml --params_file=hyper.test
```
Note that `--config_files=test.yaml` is optional, if you don't have any customize config settings, this parameter can be empty.

This processing maybe take a long time to output best hyperparameter and result:
```
running parameters:                                                                                                                    
{'embedding_size': 64, 'learning_rate': 0.005947474154838498, 'mlp_hidden_size': '[64,64,64]', 'train_batch_size': 512}                
  0%|                                                                                           | 0/18 [00:00<?, ?trial/s, best loss=?]
```

More information about parameter tuning can be found in our [docs](https://recbole.io/docs/user_guide/usage/parameter_tuning.html).


## Time and Memory Costs
We constructed preliminary experiments to test the time and memory cost on three different-sized datasets 
(small, medium and large). For detailed information, you can click the following links.

* [General recommendation models](asset/time_test_result/General_recommendation.md)
* [Sequential recommendation models](asset/time_test_result/Sequential_recommendation.md)
* [Context-aware recommendation models](asset/time_test_result/Context-aware_recommendation.md)
* [Knowledge-based recommendation models](asset/time_test_result/Knowledge-based_recommendation.md)

NOTE: Our test results only gave the approximate time and memory cost of our implementations in the RecBole library
(based on our machine server).  Any feedback or suggestions about the implementations and test are welcome. 
We will keep improving our implementations, and update these test results.


## RecBole Major Releases
| Releases | Date       |
|----------|------------|
| v1.2.1   | 02/23/2025 |
| v1.2.0   | 11/01/2023 |
| v1.1.1   | 10/05/2022 |
| v1.0.0   | 09/17/2021 |
| v0.2.0   | 01/15/2021 |
| v0.1.1   | 11/03/2020 |


## Open Source Contributions
As a one-stop framework from data processing, model development, algorithm training to scientific evaluation, RecBole has a total of **11** related GitHub projects including 
- two versions of RecBole ([RecBole 1.0](https://github.com/RUCAIBox/RecBole) and [RecBole 2.0](https://github.com/RUCAIBox/RecBole2.0));
- 8 benchmarking packages ([RecBole-MetaRec](https://github.com/nuster1128/RecBole-MetaRec), [RecBole-DA](https://github.com/RUCAIBox/RecBole-DA), [RecBole-Debias](https://github.com/JingsenZhang/RecBole-Debias), [RecBole-FairRec](https://github.com/TangJiakai/RecBole-FairRec), [RecBole-CDR](https://github.com/RUCAIBox/RecBole-CDR), [RecBole-TRM](https://github.com/RUCAIBox/RecBole-TRM), [RecBole-GNN](https://github.com/RUCAIBox/RecBole-GNN) and [RecBole-PJF](https://github.com/RUCAIBox/RecBole-PJF));
- dataset repository (<a href="https://github.com/RUCAIBox/RecSysDatasets">RecSysDatasets</a>).

In the following table, we summarize the open source contributions of GitHub projects based on RecBole.

| **Projects**                                                 | **Stars**                                                    | **Forks**                                                    | **Issues**                                                   | **Pull requests**                                            |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| [**RecBole**](https://github.com/RUCAIBox/RecBole)           | [![Stars](https://img.shields.io/github/stars/RUCAIBox/RecBole?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBox/RecBole/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecBole?style=social&logo=github)](https://github.com/RUCAIBox/RecBole/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecBole?style=social&logo=git)](https://github.com/RUCAIBox/RecBole/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecBole?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecBole/pulls) |
| [**RecBole2.0**](https://github.com/RUCAIBox/RecBole2.0)     | [![Stars](https://img.shields.io/github/stars/RUCAIBox/RecBole2.0?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBox/RecBole2.0/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecBole2.0?style=social&logo=github)](https://github.com/RUCAIBox/RecBole2.0/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecBole2.0?style=social&logo=git)](https://github.com/RUCAIBox/RecBole2.0/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecBole2.0?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecBole2.0/pulls) |
| [**RecBole-DA**](https://github.com/RUCAIBox/RecBole-DA)     | [![Stars](https://img.shields.io/github/stars/RUCAIBox/RecBole-DA?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBox/RecBole-DA/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecBole-DA?style=social&logo=github)](https://github.com/RUCAIBox/RecBole-DA/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecBole-DA?style=social&logo=git)](https://github.com/RUCAIBox/RecBole-DA/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecBole-DA?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecBole-DA/pulls) |
| [**RecBole-MetaRec**](https://github.com/nuster1128/RecBole-MetaRec) | [![Stars](https://img.shields.io/github/stars/nuster1128/RecBole-MetaRec?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/nuster1128/RecBole-MetaRec/stargazers) | [![Forks](https://img.shields.io/github/forks/nuster1128/RecBole-MetaRec?style=social&logo=github)](https://github.com/nuster1128/RecBole-MetaRec/network/members) | [![Issues](https://img.shields.io/github/issues-closed/nuster1128/RecBole-MetaRec?style=social&logo=git)](https://github.com/nuster1128/RecBole-MetaRec/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/nuster1128/RecBole-MetaRec?style=social&logo=githubactions)](https://github.com/nuster1128/RecBole-MetaRec/pulls) |
| [**RecBole-Debias**](https://github.com/JingsenZhang/RecBole-Debias) | [![Stars](https://img.shields.io/github/stars/JingsenZhang/RecBole-Debias?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/JingsenZhang/RecBole-Debias/stargazers) | [![Forks](https://img.shields.io/github/forks/JingsenZhang/RecBole-Debias?style=social&logo=github)](https://github.com/JingsenZhang/RecBole-Debias/network/members) | [![Issues](https://img.shields.io/github/issues-closed/JingsenZhang/RecBole-Debias?style=social&logo=git)](https://github.com/JingsenZhang/RecBole-Debias/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/JingsenZhang/RecBole-Debias?style=social&logo=githubactions)](https://github.com/JingsenZhang/RecBole-Debias/pulls) |
| [**RecBole-FairRec**](https://github.com/TangJiakai/RecBole-FairRec) | [![Stars](https://img.shields.io/github/stars/TangJiakai/RecBole-FairRec?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/TangJiakai/RecBole-FairRec/stargazers) | [![Forks](https://img.shields.io/github/forks/TangJiakai/RecBole-FairRec?style=social&logo=github)](https://github.com/TangJiakai/RecBole-FairRec/network/members) | [![Issues](https://img.shields.io/github/issues-closed/TangJiakai/RecBole-FairRec?style=social&logo=git)](https://github.com/TangJiakai/RecBole-FairRec/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/TangJiakai/RecBole-FairRec?style=social&logo=githubactions)](https://github.com/TangJiakai/RecBole-FairRec/pulls) |
| [**RecBole-CDR**](https://github.com/RUCAIBox/RecBole-CDR)   | [![Stars](https://img.shields.io/github/stars/RUCAIBox/RecBole-CDR?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBox/RecBole-CDR/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecBole-CDR?style=social&logo=github)](https://github.com/RUCAIBox/RecBole-CDR/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecBole-CDR?style=social&logo=git)](https://github.com/RUCAIBox/RecBole-CDR/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecBole-CDR?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecBole-CDR/pulls) |
| [**RecBole-GNN**](https://github.com/RUCAIBox/RecBole-GNN)   | [![Stars](https://img.shields.io/github/stars/RUCAIBox/RecBole-GNN?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBox/RecBole-GNN/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecBole-GNN?style=social&logo=github)](https://github.com/RUCAIBox/RecBole-GNN/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecBole-GNN?style=social&logo=git)](https://github.com/RUCAIBox/RecBole-GNN/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecBole-GNN?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecBole-GNN/pulls) |
| [**RecBole-TRM**](https://github.com/RUCAIBox/RecBole-TRM)   | [![Stars](https://img.shields.io/github/stars/RUCAIBOX/RecBole-TRM?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBOX/RecBole-TRM/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecBole-TRM?style=social&logo=github)](https://github.com/RUCAIBox/RecBole-TRM/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecBole-TRM?style=social&logo=git)](https://github.com/RUCAIBox/RecBole-TRM/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecBole-TRM?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecBole-TRM/pulls) |
| [**RecBole-PJF**](https://github.com/RUCAIBox/RecBole-PJF)   | [![Stars](https://img.shields.io/github/stars/RUCAIBox/RecBole-PJF?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBox/RecBole-PJF/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecBole-PJF?style=social&logo=github)](https://github.com/RUCAIBox/RecBole-PJF/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecBole-PJF?style=social&logo=git)](https://github.com/RUCAIBox/RecBole-PJF/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecBole-PJF?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecBole-PJF/pulls) |
| [**RecSysDatasets**](https://github.com/RUCAIBox/RecSysDatasets) | [![Stars](https://img.shields.io/github/stars/RUCAIBox/RecSysDatasets?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBox/RecSysDatasets/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecSysDatasets?style=social&logo=github)](https://github.com/RUCAIBox/RecSysDatasets/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecSysDatasets?style=social&logo=git)](https://github.com/RUCAIBox/RecSysDatasets/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecSysDatasets?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecSysDatasets/pulls) |


## Contributing

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/RUCAIBox/RecBole/issues).

We welcome all contributions from bug fixes to new features and extensions.

We expect all contributions discussed in the issue tracker and going through PRs.

We thank the insightful suggestions from [@tszumowski](https://github.com/tszumowski), [@rowedenny](https://github.com/rowedenny), [@deklanw](https://github.com/deklanw) et.al.

We thank the nice contributions through PRs from [@rowedenny](https://github.com/rowedenny)，[@deklanw](https://github.com/deklanw) et.al.


## Cite
If you find RecBole useful for your research or development, please cite the following papers: [RecBole[1.0]](https://arxiv.org/abs/2011.01731), [RecBole[2.0]](https://dl.acm.org/doi/abs/10.1145/3459637.3482016) and [RecBole[1.2.1]](https://dl.acm.org/doi/10.1145/3539618.3591889).

```bibtex
@inproceedings{recbole[1.0],
  author    = {Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Yushuo Chen and Xingyu Pan and Kaiyuan Li and Yujie Lu and Hui Wang and Changxin Tian and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji{-}Rong Wen},
  title     = {RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms},
  booktitle = {{CIKM}},
  pages     = {4653--4664},
  publisher = {{ACM}},
  year      = {2021}
}
@inproceedings{recbole[2.0],
  author    = {Wayne Xin Zhao and Yupeng Hou and Xingyu Pan and Chen Yang and Zeyu Zhang and Zihan Lin and Jingsen Zhang and Shuqing Bian and Jiakai Tang and Wenqi Sun and Yushuo Chen and Lanling Xu and Gaowei Zhang and Zhen Tian and Changxin Tian and Shanlei Mu and Xinyan Fan and Xu Chen and Ji{-}Rong Wen},
  title     = {RecBole 2.0: Towards a More Up-to-Date Recommendation Library},
  booktitle = {{CIKM}},
  pages     = {4722--4726},
  publisher = {{ACM}},
  year      = {2022}
}
@inproceedings{recbole[1.2.1],
  author    = {Lanling Xu and Zhen Tian and Gaowei Zhang and Junjie Zhang and Lei Wang and Bowen Zheng and Yifan Li and Jiakai Tang and Zeyu Zhang and Yupeng Hou and Xingyu Pan and Wayne Xin Zhao and Xu Chen and Ji{-}Rong Wen},
  title     = {Towards a More User-Friendly and Easy-to-Use Benchmark Library for Recommender Systems},
  booktitle = {{SIGIR}},
  pages     = {2837--2847},
  publisher = {{ACM}},
  year      = {2023}
}
```


## The Team

RecBole is developed by [RUC, BUPT, ECNU](https://www.recbole.io/about.html), and maintained by RUC.

Here is the list of our lead developers in each development phase. They are the souls of RecBole and have made outstanding contributions.

|         Time          |        Version         |                Lead Developers                 |                Paper            |
| :-------------------: | :--------------------: | :--------------------------------------------: | ---------------------------------------------- |
| June 2020<br> ~<br> Nov. 2020 |        v0.1.1         |  Shanlei Mu ([@ShanleiMu](https://github.com/ShanleiMu)), Yupeng Hou ([@hyp1231](https://github.com/hyp1231)),<br> Zihan Lin ([@linzihan-backforward](https://github.com/linzihan-backforward)), Kaiyuan Li ([@tsotfsk](https://github.com/tsotfsk))| [PDF](https://dl.acm.org/doi/abs/10.1145/3459637.3482016) |
|    Nov. 2020<br> ~ <br> Jul. 2022    | v0.1.2 ~ v1.0.1 |      Yushuo Chen ([@chenyushuo](https://github.com/chenyushuo)), Xingyu Pan ([@2017pxy](https://github.com/2017pxy))    | [PDF](https://dl.acm.org/doi/abs/10.1145/3459637.3482016)  |
| Jul. 2022<br/> ~ <br/> Nov. 2023 | v1.1.0 ~ v1.1.1 | Lanling Xu ([@Sherry-XLL](https://github.com/Sherry-XLL)), Zhen Tian ([@chenyuwuxin](https://github.com/chenyuwuxin)), Gaowei Zhang ([@Wicknight](https://github.com/Wicknight)), Lei Wang ([@Paitesanshi](https://github.com/Paitesanshi)), Junjie Zhang ([@leoleojie](https://github.com/leoleojie)) | [PDF](https://dl.acm.org/doi/10.1145/3539618.3591889) |
| Nov. 2023<br/> ~ <br/> Feb. 2025 | v1.2.0 | Bowen Zheng ([@zhengbw0324](https://github.com/zhengbw0324)), Chen Ma ([@Yilu114](https://github.com/Yilu114)) | [PDF](https://dl.acm.org/doi/10.1145/3539618.3591889) |
| Feb. 2025<br/> ~ <br/> now | v1.2.1 | Enze Liu ([@BishopLiu](https://github.com/BishopLiu)), Kesha Ou ([@TayTroye](https://github.com/TayTroye)), Bingqian Li ([@Fotiligner](https://github.com/Fotiligner)) | [PDF](https://dl.acm.org/doi/10.1145/3539618.3591889) |


## License
RecBole uses [MIT License](./LICENSE). All data and code in this project can only be used for academic purposes.

## Acknowledgments

This project was supported by National Natural Science Foundation of China (No. 61832017).