"""Tune the Feature Augmentation hybrid on VALIDATION scores. The test set is never opened.

A configuration is a target model with its hyperparameters, a set of source models, and
(n_pseudo, pseudo_weight): each source's top-n_pseudo unseen items per user are added to the
training matrix with value pseudo_weight, and the target is refitted on that matrix
(HybridFeatureAugmentation.fit_augmentation). No sources is the plain target.

Sources are the eight tuned models below, read from outputs/<model>/scores.npz. Targets are
the closed-form models ItemKNN, EASE and SLIMElastic, refitted for every configuration.

Steps, all on validation:
1. Standalone metrics of every source, top-10 overlap, and the correct items one model finds
   that another misses.
2. Exhaustive search: every subset of sources (including none) x n_pseudo x pseudo_weight x
   target x target hyperparameters, each fitted on train and scored on all validation users.
3. Choice of selection method. Five methods pick a configuration from that search:
       NoAugmentation  - the best target alone (no sources)
       ContentBoosted  - ContentBased as the only source on that target; only n_pseudo and
                         pseudo_weight are tuned (content-boosted CF, Melville et al., 2002)
       OnePerKind      - the best source of each kind, then the best subset of those
       Greedy          - start from NoAugmentation and add the source that most improves
                         NDCG@10, retuning everything else at each step; stop when nothing helps
       Exhaustive      - the best configuration overall
   Picking the best of thousands of configurations on the same users overfits them, so the
   validation users are split into random halves (N_REPEATS times). Each method selects on one
   half and is scored on the other. The method with the best mean held-out NDCG@10 is applied
   to the full validation set. Ties go to the simpler method.
4. The final configuration is refitted with a fresh HybridFeatureAugmentation on RecBole's own
   dataset, its recommendations are evaluated with the reference (pandas) metrics, and the
   result must equal the search. It is written to configs/hybrid_feature_aug_best.yaml.

The search reuses the hybrid's own code: source rankings come from _rank_candidates (computed
once), and every configuration is fitted with fit_augmentation and scored with
full_sort_predict. Results are cached per target in outputs/HybridFeatureAug/, so an
interrupted run resumes; the cache is ignored when the sources, the split or the grid change.

Prerequisite (step 1, train every source once and export its scores):
    python run_model.py SLIMElastic EASE ItemKNN RecVAE SimpleX BPR LightGCN ContentBased

Examples:
    python tune_hybrid_feature_aug.py                      # all targets: roughly 1.5-2 h with 10 workers
    python tune_hybrid_feature_aug.py --targets ItemKNN EASE   # skip SLIMElastic (12 s per fit)
"""

import argparse
import copy
import hashlib
import itertools
import os
from multiprocessing import get_context

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from recbole.data.interaction import Interaction
from recbole.model.general_recommender.hybrid_feature_aug import HybridFeatureAugmentation
from recbole.model.general_recommender.hybrid_regression import (
    METRIC_NAMES,
    PRIMARY_METRIC,
    TOP_K,
    evaluate_ranking,
    load_score_stack,
)
from tune_hybrid_regression import _plain, best_index, overlap_tables, recommendation_frame

# Source models and their kind. ContentBased must not be first: load_score_stack aligns the
# other matrices onto the first model's items, and ContentBased knows 108 extra items.
KINDS = {
    "SLIMElastic": "neighbourhood",
    "EASE": "neighbourhood",
    "ItemKNN": "neighbourhood",
    "RecVAE": "latent factor",
    "SimpleX": "latent factor",
    "BPR": "latent factor",
    "LightGCN": "latent factor",
    "ContentBased": "content",
}
CONTENT_SOURCE = "ContentBased"

# Target hyperparameters, centred on configs/tuned/<target>.yaml. Anything not listed keeps
# RecBole's default (recbole/properties/model/<target>.yaml). SLIMElastic takes about 12 s per
# fit, so only its tuned values are searched: it already needs one fit per source subset,
# n_pseudo and pseudo_weight (3,061 fits, about 10 CPU hours).
TARGET_GRIDS = {
    "ItemKNN": {"k": [100, 200, 400], "shrink": [0.0, 10.0]},
    "EASE": {"reg_weight": [250.0, 500.0, 1000.0, 2000.0]},
    "SLIMElastic": {"alpha": [0.1], "l1_ratio": [0.005]},
}
N_PSEUDO_GRID = (2, 5, 10, 20)
PSEUDO_WEIGHTS = (0.25, 0.5, 1.0)

METHODS = ["NoAugmentation", "ContentBoosted", "OnePerKind", "Greedy", "Exhaustive"]  # simplest first
N_REPEATS = 10  # random half splits -> 2 * N_REPEATS selection folds
SEED = 2020
N_WORKERS = 10
NDCG_COL = METRIC_NAMES.index(PRIMARY_METRIC)

_state = {}


# --------------------------------------------------------------------------- #
# Stand-ins for the RecBole objects the hybrid reads
# --------------------------------------------------------------------------- #
class TrainMatrix:
    """The parts of a RecBole Dataset that HybridFeatureAugmentation and its targets read."""

    def __init__(self, matrix, uid_field, iid_field):
        self._matrix = matrix.tocsr()
        self.user_num, self.item_num = matrix.shape
        self.uid_field, self.iid_field = uid_field, iid_field

    def inter_matrix(self, form="coo", value_field=None):
        return self._matrix.copy() if form == "csr" else self._matrix.tocoo()

    def num(self, field):
        return self.user_num if field == self.uid_field else self.item_num


class TargetConfig(dict):
    """Dict with attribute access: EASE reads `config.device`."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class FixedScores(torch.nn.Module):
    """A source model that returns its exported scores (outputs/<model>/scores.npz).

    scores has no padding row/column; RecBole's internal ids are the score indices + 1.
    """

    def __init__(self, scores, user_field):
        super().__init__()
        padded = np.full((scores.shape[0] + 1, scores.shape[1] + 1), -np.inf, dtype=np.float32)
        padded[1:, 1:] = scores
        self.scores = torch.from_numpy(padded)
        self.user_field = user_field

    def full_sort_predict(self, interaction):
        return self.scores[interaction[self.user_field]].flatten()


# --------------------------------------------------------------------------- #
# Fitting and scoring one configuration
# --------------------------------------------------------------------------- #
def full_scores(hybrid):
    """Scores of every user for every item [n_users - 1, n_items - 1], without RecBole's padding."""
    users = torch.arange(1, hybrid.n_users)
    with torch.no_grad():
        scores = hybrid.full_sort_predict(Interaction({hybrid.USER_ID: users}))
    return scores.reshape(len(users), hybrid.n_items)[:, 1:].float().cpu().numpy()


def fit_configuration(hybrid, rankings, sources, n_pseudo, pseudo_weight, target_params):
    """Fit one configuration with the hybrid's own fit_augmentation, reusing cached rankings.

    rankings[m] is _rank_candidates of source m for the largest n_pseudo; fit_augmentation uses
    its first n_pseudo columns, as it does when it caches rankings itself. The source models
    are not called again, so only their indices are passed on.
    """
    hybrid.source_models = list(sources)
    hybrid._rankings = [rankings[m] for m in sources] if sources else None
    np.random.seed(SEED)  # SLIMElastic's ElasticNet(selection="random") draws from the global RNG
    hybrid.fit_augmentation(
        n_pseudo=n_pseudo if sources else 0, pseudo_weight=pseudo_weight, target_params=target_params
    )
    return full_scores(hybrid)


def _init_worker(matrix, fields, rankings, seen, relevant, targets, param_lists):
    from threadpoolctl import threadpool_limits

    _state["limits"] = threadpool_limits(1)  # one BLAS thread per worker process
    torch.set_num_threads(1)
    config = TargetConfig(device=torch.device("cpu"), **fields)
    dataset = TrainMatrix(matrix, fields["USER_ID_FIELD"], fields["ITEM_ID_FIELD"])
    _state.update(
        hybrids={t: HybridFeatureAugmentation(config, dataset, [], target_model=t) for t in targets},
        rankings=rankings,
        seen=seen,
        relevant=relevant,
        param_lists=param_lists,
    )


def _evaluate_task(task):
    """(target, source indices, n_pseudo, pseudo_weight) -> metrics [P, 5], per-user NDCG [P, n_users]."""
    target, sources, n_pseudo, pseudo_weight = task
    hybrid = _state["hybrids"][target]
    params = _state["param_lists"][target]
    metrics = np.empty((len(params), len(METRIC_NAMES)))
    ndcg = np.empty((len(params), _state["seen"].shape[0]), dtype=np.float32)
    for p, target_params in enumerate(params):
        scores = fit_configuration(hybrid, _state["rankings"], sources, n_pseudo, pseudo_weight, target_params)
        metric, per_user = evaluate_ranking(scores, _state["seen"], _state["relevant"])
        metrics[p] = [metric[name] for name in METRIC_NAMES]
        ndcg[p] = per_user
    return metrics, ndcg


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #
def param_grid(target):
    """Every hyperparameter combination of one target, as dicts."""
    grid = TARGET_GRIDS[target]
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def build_search_space(targets, n_sources):
    """Every configuration, simplest first so that a tie keeps the simpler one.

    Order: the plain targets, then source subsets by size, then n_pseudo, pseudo_weight,
    target and target hyperparameters. Returns the configuration arrays and, per target,
    the fitting tasks with the configuration rows each task fills.
    """
    param_lists = {t: param_grid(t) for t in targets}
    rows = {"target": [], "param": [], "mask": [], "n_pseudo": [], "weight": []}
    tasks = {t: [] for t in targets}

    def add(mask, n_pseudo, weight):
        for t, target in enumerate(targets):
            start = len(rows["target"])
            for p in range(len(param_lists[target])):
                for key, value in zip(rows, (t, p, mask, n_pseudo, weight)):
                    rows[key].append(value)
            sources = tuple(m for m in range(n_sources) if mask >> m & 1)
            tasks[target].append(((target, sources, n_pseudo, weight), np.arange(start, len(rows["target"]))))

    add(0, 0, 0.0)
    for mask in sorted(range(1, 1 << n_sources), key=lambda m: (bin(m).count("1"), m)):
        for n_pseudo in N_PSEUDO_GRID:
            for weight in PSEUDO_WEIGHTS:
                add(mask, n_pseudo, weight)
    space = {
        "target": np.asarray(rows["target"], dtype=np.int64),
        "param": np.asarray(rows["param"], dtype=np.int64),
        "mask": np.asarray(rows["mask"], dtype=np.int64),
        "n_pseudo": np.asarray(rows["n_pseudo"], dtype=np.int64),
        "weight": np.asarray(rows["weight"], dtype=np.float64),
    }
    return space, tasks, param_lists


def params_text(params):
    return " ".join(f"{key}={value:g}" for key, value in params.items())


def config_label(target, params, sources, n_pseudo, weight):
    head = f"{target}[{params_text(params)}]"
    if not sources:
        return f"{head} (no augmentation)"
    return f"{head} <- {'+'.join(sources)} | N={n_pseudo} w={weight:g}"


# --------------------------------------------------------------------------- #
# Selection methods (operate on one score per configuration)
# --------------------------------------------------------------------------- #
def no_augmentation_index(scores, masks):
    return best_index(scores, masks == 0)


def content_boosted_index(scores, masks, targets, params, content_bit):
    """ContentBased as the only source on the NoAugmentation target; only N and the weight vary."""
    base = no_augmentation_index(scores, masks)
    eligible = (masks == content_bit) & (targets == targets[base]) & (params == params[base])
    return best_index(scores, eligible)


def one_per_kind_index(scores, masks, names, kinds, source_scores):
    """Best source of each kind by its own NDCG, then the best non-empty subset of those."""
    allowed = 0
    for kind in sorted({kinds[name] for name in names}):
        members = [i for i, name in enumerate(names) if kinds[name] == kind]
        allowed |= 1 << max(members, key=lambda i: (source_scores[i], -i))
    eligible = (masks != 0) & ((masks & ~np.int64(allowed)) == 0)
    return best_index(scores, eligible)


def greedy_index(scores, masks, n_sources):
    """Start from NoAugmentation; add the source whose best configuration scores highest.

    Everything except the source set (N, weight, target, its hyperparameters) is retuned at
    each step. Stop when no source improves. Returns (chosen index, indices along the path).
    """
    chosen = no_augmentation_index(scores, masks)
    support, score, path = 0, float(scores[chosen]), []
    while True:
        best_i, best_s = None, -np.inf
        for source in range(n_sources):
            if support >> source & 1:
                continue
            eligible = masks == (support | (1 << source))
            if not eligible.any():
                continue
            i = best_index(scores, eligible)
            if scores[i] > best_s:
                best_i, best_s = i, float(scores[i])
        if best_i is None or best_s <= score:
            return chosen, path
        path.append(best_i)
        chosen, support, score = best_i, int(masks[best_i]), best_s


def exhaustive_index(scores):
    return int(np.argmax(scores))


def select_index(method, scores, space, names, source_scores):
    masks = space["mask"]
    if method == "NoAugmentation":
        return no_augmentation_index(scores, masks)
    if method == "ContentBoosted":
        bit = 1 << names.index(CONTENT_SOURCE)
        return content_boosted_index(scores, masks, space["target"], space["param"], bit)
    if method == "OnePerKind":
        return one_per_kind_index(scores, masks, names, KINDS, source_scores)
    if method == "Greedy":
        return greedy_index(scores, masks, len(names))[0]
    if method == "Exhaustive":
        return exhaustive_index(scores)
    raise ValueError(method)


# --------------------------------------------------------------------------- #
# Search with a per-target cache
# --------------------------------------------------------------------------- #
def fingerprint(*arrays, extra=""):
    h = hashlib.md5(extra.encode())
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def run_search(pool, tasks, n_configs, n_users, out_dir, key, use_cache):
    """Fit every configuration. Returns metrics [n_configs, 5] and per-user NDCG [n_configs, n_users]."""
    metrics = np.full((n_configs, len(METRIC_NAMES)), np.nan)
    ndcg = np.full((n_configs, n_users), np.nan, dtype=np.float32)
    for target, target_tasks in tasks.items():
        rows = np.concatenate([r for _, r in target_tasks])
        cache = os.path.join(out_dir, f"search_cache_{target}.npz")
        if use_cache and os.path.exists(cache):
            data = np.load(cache)
            if str(data["key"]) == key and len(data["metrics"]) == len(rows):
                metrics[rows], ndcg[rows] = data["metrics"], data["ndcg"]
                print(f"{target}: {len(rows)} configurations loaded from {cache}", flush=True)
                continue
        # Longest tasks (most sources) first, for a better load balance across workers.
        order = sorted(range(len(target_tasks)), key=lambda i: -len(target_tasks[i][0][1]))
        specs = [target_tasks[i][0] for i in order]
        with tqdm(total=len(rows), desc=f"{target} ({len(specs)} matrices)", unit="cfg") as bar:
            for i, (m, n) in zip(order, pool.imap(_evaluate_task, specs, chunksize=1)):
                metrics[target_tasks[i][1]] = m
                ndcg[target_tasks[i][1]] = n
                bar.update(len(m))
        np.savez(cache, key=key, metrics=metrics[rows], ndcg=ndcg[rows])
    return metrics, ndcg


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def load_recbole_train(target):
    """RecBole's config and training data for one target (the same split as outputs/split)."""
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed
    from recbole.utils.experiment import build_config

    config = build_config(target)
    init_seed(config["seed"], config["reproducibility"])
    train_data, _, _ = data_preparation(config, create_dataset(config))
    return config, train_data


def main():
    from recbole.model.general_recommender.mixed_hybrid import evaluate_recommendations
    from recbole.utils.experiment import CONFIG_DIR, OUTPUT_DIR, SPLIT_DIR

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="+", choices=list(TARGET_GRIDS), default=list(TARGET_GRIDS))
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--no-cache", action="store_true", help="refit every configuration")
    args = parser.parse_args()
    targets = [t for t in TARGET_GRIDS if t in args.targets]

    out_dir = os.path.join(OUTPUT_DIR, "HybridFeatureAug")
    os.makedirs(out_dir, exist_ok=True)

    def read_split(name):
        # The test file is intentionally not read.
        return pd.read_csv(os.path.join(SPLIT_DIR, f"{name}.tsv"), sep="\t", dtype={"user_id": str, "item_id": str})

    train, valid = read_split("train"), read_split("valid")
    names = list(KINDS)
    scores, user_ids, item_ids = load_score_stack(names)
    if np.isnan(scores).any():
        raise RuntimeError("a score matrix contains NaN")
    user_index = {user: i for i, user in enumerate(user_ids)}
    item_index = {item: i for i, item in enumerate(item_ids)}
    seen = np.zeros(scores.shape[1:], dtype=bool)
    seen[train["user_id"].map(user_index), train["item_id"].map(item_index)] = True
    relevant = np.zeros_like(seen)
    relevant[valid["user_id"].map(user_index), valid["item_id"].map(item_index)] = True

    # RecBole's training matrix, in the same id order as the exported scores (internal id = index + 1).
    config, train_data = load_recbole_train(targets[0])
    dataset = train_data._dataset
    uid, iid = dataset.uid_field, dataset.iid_field
    if (list(dataset.id2token(uid, np.arange(1, dataset.user_num))) != list(user_ids)
            or list(dataset.id2token(iid, np.arange(1, dataset.item_num))) != list(item_ids)):
        raise RuntimeError("RecBole's ids differ from outputs/<model>/scores.npz; re-run run_model.py")
    matrix = dataset.inter_matrix(form="csr").astype(np.float32)
    matrix.data[:] = 1.0
    if not np.array_equal(matrix[1:, 1:].toarray().astype(bool), seen):
        raise RuntimeError("RecBole's training split differs from outputs/split/train.tsv")
    fields = {key: config[key] for key in ("USER_ID_FIELD", "ITEM_ID_FIELD", "NEG_PREFIX")}

    # 1. Standalone metrics and overlap of the sources; sources are ordered by standalone NDCG@10.
    standalone = [evaluate_ranking(scores[m], seen, relevant) for m in range(len(names))]
    order = sorted(range(len(names)), key=lambda i: -standalone[i][0][PRIMARY_METRIC])
    names = [names[i] for i in order]
    scores = scores[order]
    standalone = [standalone[i] for i in order]
    table = pd.DataFrame([{"model": n, "kind": KINDS[n], **s[0]} for n, s in zip(names, standalone)])
    table.to_csv(os.path.join(out_dir, "standalone_valid.csv"), index=False)
    print("--- Standalone validation metrics of the sources ---")
    print(table.to_string(index=False, float_format="%.4f"))

    shared, extra, exclusive = overlap_tables(scores, seen, relevant, names)
    shared.to_csv(os.path.join(out_dir, "overlap_shared_top10_valid.csv"))
    extra.to_csv(os.path.join(out_dir, "overlap_extra_hits_valid.csv"))
    exclusive.to_csv(os.path.join(out_dir, "exclusive_hits_valid.csv"), index=False)
    print("\n--- Mean number of shared top-10 items per validation user ---")
    print(shared.to_string(float_format="%.1f"))
    print("\n--- Correct top-10 items the row model finds and the column model misses ---")
    print(extra.to_string(float_format="%.2f"))
    print("\n--- Correct items per user, and those found by no other model ---")
    print(exclusive.to_string(index=False, float_format="%.3f"))

    # Each source's top unseen items, once, with the hybrid's own ranking code.
    sources = [FixedScores(scores[m], fields["USER_ID_FIELD"]) for m in range(len(names))]
    ranker = HybridFeatureAugmentation(
        TargetConfig(device=torch.device("cpu"), **fields), TrainMatrix(matrix, uid, iid), sources
    )
    rankings = [ranker._rank_candidates(source, max(N_PSEUDO_GRID)) for source in sources]

    # 2. Exhaustive search.
    space, tasks, param_lists = build_search_space(targets, len(names))
    n_configs = len(space["target"])
    print(f"\n--- Exhaustive search: {n_configs} configurations ---")
    for target in targets:
        print(f"{target}: {len(tasks[target])} augmented matrices x {len(param_lists[target])} "
              f"hyperparameter settings {TARGET_GRIDS[target]}")
    key = fingerprint(
        matrix.indptr, matrix.indices, relevant, *rankings,
        extra=repr((names, TARGET_GRIDS, N_PSEUDO_GRID, PSEUDO_WEIGHTS, SEED)),
    )
    workers = max(1, min(args.workers, os.cpu_count() or 1))
    print(f"Fitting with {workers} workers", flush=True)
    with get_context("spawn").Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(matrix, fields, rankings, seen, relevant, targets, param_lists),
    ) as pool:
        metrics, ndcg = run_search(pool, tasks, n_configs, len(user_ids), out_dir, key, not args.no_cache)

    def choice_params(i):
        return param_lists[targets[space["target"][i]]][space["param"][i]]

    def choice_sources(i):
        return [names[m] for m in range(len(names)) if space["mask"][i] >> m & 1]

    labels = [
        config_label(targets[space["target"][i]], choice_params(i), choice_sources(i),
                     space["n_pseudo"][i], space["weight"][i])
        for i in range(n_configs)
    ]
    search = pd.DataFrame(metrics, columns=METRIC_NAMES)
    search.insert(0, "pseudo_weight", space["weight"])
    search.insert(0, "n_pseudo", space["n_pseudo"])
    search.insert(0, "n_sources", [bin(m).count("1") for m in space["mask"]])
    search.insert(0, "sources", ["+".join(choice_sources(i)) for i in range(n_configs)])
    search.insert(0, "target_params", [params_text(choice_params(i)) for i in range(n_configs)])
    search.insert(0, "target", [targets[t] for t in space["target"]])
    search.insert(0, "configuration", labels)
    search = search.sort_values(PRIMARY_METRIC, ascending=False, kind="mergesort")
    search.to_csv(os.path.join(out_dir, "search_valid.csv.gz"), index=False)
    shown = ["configuration"] + METRIC_NAMES
    print(search[shown].head(15).to_string(index=False, float_format="%.4f"))
    print("\nBest configuration per target, without and with augmentation:")
    best_rows = search.groupby(["target", search["n_sources"] > 0], sort=False).head(1)
    print(best_rows[shown].to_string(index=False, float_format="%.4f"))
    print("\nBest configuration per number of sources:")
    per_size = search.groupby("n_sources", sort=True).head(1).sort_values("n_sources")
    print(per_size[["n_sources"] + shown].to_string(index=False, float_format="%.4f"))

    # 3. Selection methods on validation half splits.
    valid_users = np.flatnonzero(relevant.any(axis=1))
    folds = np.zeros((len(user_ids), 2 * N_REPEATS))
    rng = np.random.default_rng(SEED)
    for repeat in range(N_REPEATS):
        perm = rng.permutation(valid_users)
        folds[perm[: len(perm) // 2], 2 * repeat] = 1
        folds[perm[len(perm) // 2:], 2 * repeat + 1] = 1
    fold_scores = np.nan_to_num(ndcg).astype(np.float64) @ folds / folds.sum(axis=0)
    source_ndcg = np.stack([s[1] for s in standalone])
    source_fold_scores = np.nan_to_num(source_ndcg) @ folds / folds.sum(axis=0)

    print(f"\n--- Selection methods on {2 * N_REPEATS} validation half splits ---")
    rows = []
    for fold in range(2 * N_REPEATS):
        select, held = fold_scores[:, fold], fold_scores[:, fold ^ 1]
        for method in METHODS:
            chosen = select_index(method, select, space, names, source_fold_scores[:, fold])
            rows.append({
                "fold": fold,
                "method": method,
                "configuration": labels[chosen],
                "selection_half_NDCG@10": float(select[chosen]),
                "held_out_half_NDCG@10": float(held[chosen]),
            })
    split_half = pd.DataFrame(rows)
    split_half.to_csv(os.path.join(out_dir, "split_half_valid.csv"), index=False)
    summary = split_half.groupby("method", sort=False).agg(
        held_out_mean=("held_out_half_NDCG@10", "mean"),
        held_out_std=("held_out_half_NDCG@10", "std"),
        selection_mean=("selection_half_NDCG@10", "mean"),
        distinct_configurations=("configuration", "nunique"),
    ).reindex(METHODS)
    summary["optimism"] = summary["selection_mean"] - summary["held_out_mean"]
    held_by_method = split_half.pivot(index="fold", columns="method", values="held_out_half_NDCG@10")
    delta = held_by_method.sub(held_by_method["NoAugmentation"], axis=0)
    summary["mean_delta_vs_no_augmentation"] = delta.mean().reindex(METHODS)
    summary["folds_better_than_no_augmentation"] = (delta > 0).sum().reindex(METHODS)
    summary.to_csv(os.path.join(out_dir, "split_half_summary_valid.csv"))
    print(summary.to_string(float_format="%.4f"))
    chosen_method = METHODS[0]
    for method in METHODS[1:]:
        if summary.loc[method, "held_out_mean"] > summary.loc[chosen_method, "held_out_mean"]:
            chosen_method = method
    print(f"Selection method with the best held-out {PRIMARY_METRIC}: {chosen_method}")

    # Every method on the full validation set; the chosen one is the final configuration.
    full = metrics[:, NDCG_COL]
    source_full = np.array([s[0][PRIMARY_METRIC] for s in standalone])
    _, greedy_path = greedy_index(full, space["mask"], len(names))
    path_rows, previous = [], float(full[no_augmentation_index(full, space["mask"])])
    for step, i in enumerate(greedy_path, start=1):
        path_rows.append({
            "step": step,
            "sources": "+".join(choice_sources(i)),
            "configuration": labels[i],
            PRIMARY_METRIC: float(full[i]),
            "gain": float(full[i]) - previous,
        })
        previous = float(full[i])
    pd.DataFrame(path_rows).to_csv(os.path.join(out_dir, "greedy_path_valid.csv"), index=False)
    print("\n--- Greedy path (full validation) ---")
    print(pd.DataFrame(path_rows).to_string(index=False, float_format="%.4f") if path_rows
          else "no source improves on the plain target")

    alternatives = {}
    for method in METHODS:
        i = select_index(method, full, space, names, source_full)
        target = targets[space["target"][i]]
        alternatives[method] = {
            "target": target,
            "target_params": {**HybridFeatureAugmentation._default_params(target), **choice_params(i)},
            "sources": choice_sources(i),
            "n_pseudo": int(space["n_pseudo"][i]),
            "pseudo_weight": float(space["weight"][i]),
            "valid_metrics": dict(zip(METRIC_NAMES, map(float, metrics[i]))),
        }
    final = copy.deepcopy(alternatives[chosen_method])

    # 4. Refit the final configuration from scratch on RecBole's own data and check it.
    print("\nCross-checking the final configuration with a fresh HybridFeatureAugmentation...", flush=True)
    config, train_data = load_recbole_train(final["target"])
    reference_sources = [FixedScores(scores[names.index(n)], fields["USER_ID_FIELD"]) for n in final["sources"]]
    hybrid = HybridFeatureAugmentation(
        config, train_data, reference_sources, final["target"], final["target_params"]
    )
    np.random.seed(SEED)
    hybrid.fit_augmentation(n_pseudo=final["n_pseudo"], pseudo_weight=final["pseudo_weight"])
    recs = recommendation_frame(full_scores(hybrid), seen, user_ids, item_ids, relevant.any(axis=1))
    reference = evaluate_recommendations(recs, valid, k=TOP_K)
    for name in METRIC_NAMES:
        if abs(reference[name] - final["valid_metrics"][name]) > 1e-9:
            raise AssertionError((name, reference[name], final["valid_metrics"][name]))
    recs_path = os.path.join(out_dir, "recs_valid.tsv")
    recs.to_csv(recs_path, sep="\t", index=False)

    def describe(alt):
        return config_label(alt["target"], {k: alt["target_params"][k] for k in TARGET_GRIDS[alt["target"]]},
                            alt["sources"], alt["n_pseudo"], alt["pseudo_weight"])

    print("\n--- Best configuration of each selection method (full validation) ---")
    for method in METHODS:
        alt = alternatives[method]
        print(f"{method:<15} {PRIMARY_METRIC}={alt['valid_metrics'][PRIMARY_METRIC]:.4f}  {describe(alt)}")

    payload = {
        **final,
        "primary_metric": PRIMARY_METRIC,
        "selection_method": chosen_method,
        "held_out_half_ndcg": {m: float(summary.loc[m, "held_out_mean"]) for m in METHODS},
        "alternatives": alternatives,
    }
    best_path = os.path.join(CONFIG_DIR, "hybrid_feature_aug_best.yaml")
    with open(best_path, "w", encoding="utf-8") as f:
        f.write(f"# Selected by {chosen_method}, valid {PRIMARY_METRIC} = "
                f"{final['valid_metrics'][PRIMARY_METRIC]:.4f} (written by tune_hybrid_feature_aug.py)\n")
        yaml.safe_dump(_plain(payload), f, sort_keys=False)
    print(f"\nFinal ({chosen_method}): {describe(final)}")
    print(f"Wrote {best_path}")
    print(f"Wrote {recs_path} and the tables in {out_dir}")


if __name__ == "__main__":
    main()
