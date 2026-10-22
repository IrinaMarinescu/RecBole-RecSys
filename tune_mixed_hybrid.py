"""Tune the Mixed Hybrid on the VALIDATION recommendation lists.

The test set is never read here. The primary metric is NDCG@10 (same as configs/base.yaml).

A configuration is an ordered list of models and a quota per model (see mix_recommendations):
in each round every model, in order, adds its next `quota` not-yet-chosen items, until 10 are
chosen. Steps:

1. Standalone + overlap analysis of the candidate models: how many top-10 items each pair of
   models shares, and how many correct items one model finds that another misses.
2. Exhaustive quota search: every quota vector over ALL candidates (0..10 per model, total
   1..10; a total above 10 behaves the same as 10), with models ordered by standalone
   validation NDCG@10. This covers every subset of models, so mixes with several models of
   the same kind are included.
3. Order search: for a chosen set of models, every ordering x every positive quota vector
   (skipped for more than MAX_ORDER_MODELS models).
4. Choosing the selection method: four methods pick a configuration from steps 2-3:
       BestSingle  - the best single model
       OnePerKind  - the best model of each kind, then the best mix of those
       Greedy      - add models one at a time while validation NDCG@10 improves
       Exhaustive  - the best configuration overall
   A method that searches more configurations fits validation noise more. To measure that,
   validation users are split into random halves (N_REPEATS times); each method selects on
   one half and is scored on the other. The method with the best held-out NDCG@10 is applied
   to the full validation set to produce the final configuration.

Prerequisite:
    python run_model.py RecVAE SLIMElastic EASE LightGCN BPR SimpleX ItemKNN UserKNN ContentBased

Example:
    python tune_mixed_hybrid.py
"""

import copy
import itertools
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml

from recbole.utils.experiment import CONFIG_DIR, EXPORT_TOP_N, OUTPUT_DIR, load_split
from recbole.model.general_recommender.mixed_hybrid import (
    PRIMARY_METRIC,
    TOP_K,
    evaluate_arrays,
    evaluate_recommendations,
    load_sources,
    mix_arrays,
    mix_recommendations,
    save_recommendations,
    to_arrays,
)

# Every model with tuned hyperparameters in configs/tuned/, and its kind.
KINDS = {
    "ItemKNN": "neighbourhood",
    "UserKNN": "neighbourhood",
    "EASE": "neighbourhood",          # learned item-item weights
    "SLIMElastic": "neighbourhood",   # learned item-item weights
    "BPR": "latent factor",
    "LightGCN": "latent factor",
    "SimpleX": "latent factor",
    "RecVAE": "latent factor",
    "ContentBased": "content",
}
METHODS = ["BestSingle", "OnePerKind", "Greedy", "Exhaustive"]  # simplest first: ties go to the simpler one
N_REPEATS = 10           # random half splits of the validation users -> 2 * N_REPEATS selection folds
SEED = 2020
MAX_ORDER_MODELS = 6     # order search for k models evaluates k! * C(10, k) configurations
N_WORKERS = 4
METRIC_NAMES = [f"{m}@{TOP_K}" for m in ("Recall", "MRR", "NDCG", "Hit", "Precision")]
NDCG = METRIC_NAMES.index(PRIMARY_METRIC)
N_METRICS = len(METRIC_NAMES)


# --------------------------------------------------------------------------- #
# Parallel evaluation
# --------------------------------------------------------------------------- #
_worker = {}


def _init_worker(lists, relevant, folds):
    _worker.update(lists=lists, relevant=relevant, folds=folds)


def _evaluate_chunk(configs):
    """[(order, quotas), ...] -> array [n, N_METRICS + n_folds]: full-validation metrics, then mean NDCG per fold."""
    lists, relevant, folds = _worker["lists"], _worker["relevant"], _worker["folds"]
    out = np.empty((len(configs), N_METRICS + folds.shape[1]))
    for i, (order, quotas) in enumerate(configs):
        metrics, ndcg = evaluate_arrays(mix_arrays(lists[list(order)], quotas), relevant)
        out[i, :N_METRICS] = [metrics[name] for name in METRIC_NAMES]
        out[i, N_METRICS:] = ndcg @ folds / folds.sum(axis=0)
    return out


def evaluate_all(pool, configs, chunk=400):
    chunks = [configs[i:i + chunk] for i in range(0, len(configs), chunk)]
    return np.vstack(pool.map(_evaluate_chunk, chunks))


def quota_vectors(n_models, total_max):
    """Every vector of non-negative ints of length n_models with 1 <= sum <= total_max."""
    for total in range(1, total_max + 1):
        for bars in itertools.combinations(range(total + n_models - 1), n_models - 1):
            edges = (-1,) + bars + (total + n_models - 1,)
            yield tuple(edges[i + 1] - edges[i] - 1 for i in range(n_models))


def positive_quota_vectors(n_models, total_max):
    """Every vector of positive ints of length n_models with sum <= total_max: all ones plus a non-negative extra."""
    if n_models > total_max:
        return []
    ones = (1,) * n_models
    return [ones] + [tuple(q + 1 for q in extra) for extra in quota_vectors(n_models, total_max - n_models)]


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #
def overlap_tables(lists, relevant, names):
    """Mean shared top-10 items per user, and mean correct items the row model finds that the column model misses."""
    top = lists[:, :, :TOP_K]
    users = np.arange(top.shape[1])[:, None]
    hits = relevant[users, np.maximum(top, 0)] & (top >= 0)
    n = len(names)
    shared, extra = np.zeros((n, n)), np.zeros((n, n))
    in_other = np.zeros((n, n) + top.shape[1:], dtype=bool)
    for a in range(n):
        for b in range(n):
            in_other[a, b] = (top[a][:, :, None] == top[b][:, None, :]).any(axis=2) & (top[a] >= 0)
            shared[a, b] = in_other[a, b].sum(axis=1).mean()
            extra[a, b] = (hits[a] & ~in_other[a, b]).sum(axis=1).mean()
    others = [in_other[a, [b for b in range(n) if b != a]].any(axis=0) for a in range(n)]
    exclusive = [float((hits[a] & ~others[a]).sum(axis=1).mean()) for a in range(n)]
    hits_per_user = hits.sum(axis=2).mean(axis=1)
    return (
        pd.DataFrame(shared, index=names, columns=names),
        pd.DataFrame(extra, index=names, columns=names),
        pd.DataFrame({"model": names, "hits_per_user": hits_per_user, "exclusive_hits_per_user": exclusive}),
    )


class Search:
    """Holds the evaluated configurations and implements the selection methods."""

    def __init__(self, pool, names, configs, results):
        self.pool, self.names = pool, names
        self.configs, self.results = configs, results  # step 2: fixed order, all models
        self.bits = np.array([sum(1 << m for m, q in enumerate(quotas) if q) for _, quotas in configs])
        self.by_support = pd.Series(np.arange(len(configs))).groupby(self.bits).apply(np.array).to_dict()
        self._ordered = {}

    def label(self, order, quotas):
        return " ".join(f"{self.names[m]}={q}" for m, q in zip(order, quotas) if q)

    def ordered(self, support):
        """Step 3 for one set of models (bitmask), cached: (configs, results)."""
        if support not in self._ordered:
            models = [m for m in range(len(self.names)) if support >> m & 1]
            if len(models) > MAX_ORDER_MODELS:
                idx = self.by_support[support]
                self._ordered[support] = ([self.configs[i] for i in idx], self.results[idx])
            else:
                configs = [
                    (order, quotas)
                    for order in itertools.permutations(models)
                    for quotas in positive_quota_vectors(len(models), TOP_K)
                ]
                print(f"  order search over {', '.join(self.names[m] for m in models)}: {len(configs)} configurations", flush=True)
                self._ordered[support] = (configs, evaluate_all(self.pool, configs))
        return self._ordered[support]

    def _finish(self, support, col):
        configs, results = self.ordered(support)
        i = int(np.argmax(results[:, col]))
        return configs[i], results[i]

    def _support(self, i):
        return int(self.bits[i])

    def best_single(self, col):
        singles = np.concatenate([self.by_support[1 << m] for m in range(len(self.names))])
        return self._finish(self._support(singles[np.argmax(self.results[singles, col])]), col)

    def one_per_kind(self, col):
        allowed = 0
        for kind in sorted(set(KINDS.values())):
            members = [m for m, name in enumerate(self.names) if KINDS[name] == kind]
            scores = [self.results[self.by_support[1 << m], col].max() for m in members]
            allowed |= 1 << members[int(np.argmax(scores))]
        idx = np.flatnonzero((self.bits & ~allowed) == 0)
        return self._finish(self._support(idx[np.argmax(self.results[idx, col])]), col)

    def greedy(self, col, path=None):
        support, score = 0, -np.inf
        while True:
            best = None
            for m in range(len(self.names)):
                if support >> m & 1:
                    continue
                idx = self.by_support[support | 1 << m]
                i = idx[np.argmax(self.results[idx, col])]
                if best is None or self.results[i, col] > self.results[best, col]:
                    best = i
            if best is None or self.results[best, col] <= score:
                break
            if path is not None:
                added = self.names[int(np.log2(self._support(best) ^ support))]
                path.append({
                    "step": len(path) + 1,
                    "added": added,
                    "configuration": self.label(*self.configs[best]),
                    PRIMARY_METRIC: self.results[best, col],
                    "gain": self.results[best, col] - score if np.isfinite(score) else np.nan,
                })
            support, score = self._support(best), self.results[best, col]
        return self._finish(support, col)

    def exhaustive(self, col):
        return self._finish(self._support(int(np.argmax(self.results[:, col]))), col)

    def select(self, method, col):
        return {
            "BestSingle": self.best_single,
            "OnePerKind": self.one_per_kind,
            "Greedy": self.greedy,
            "Exhaustive": self.exhaustive,
        }[method](col)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    out_dir = os.path.join(OUTPUT_DIR, "MixedHybrid")
    os.makedirs(out_dir, exist_ok=True)

    _, valid, _ = load_split()
    users = sorted(valid["user_id"].astype(str).unique())
    sources = load_sources(list(KINDS), stage="valid")
    lists, item_ids = to_arrays(sources, list(KINDS), users)
    index = {item: i for i, item in enumerate(item_ids)}
    relevant = np.zeros((len(users), len(item_ids) + 1), dtype=bool)  # last column: items no model recommends
    user_index = {user: u for u, user in enumerate(users)}
    for user, item in zip(valid["user_id"].astype(str), valid["item_id"].astype(str)):
        relevant[user_index[user], index.get(item, len(item_ids))] = True

    # Random halves of the validation users: fold 2r and 2r+1 are the two halves of repeat r.
    rng = np.random.default_rng(SEED)
    folds = np.zeros((len(users), 2 * N_REPEATS))
    for r in range(N_REPEATS):
        perm = rng.permutation(len(users))
        folds[perm[: len(users) // 2], 2 * r] = 1
        folds[perm[len(users) // 2:], 2 * r + 1] = 1

    # 1. Standalone metrics; models are ordered by them from here on.
    standalone = {}
    for m, name in enumerate(KINDS):
        metrics, _ = evaluate_arrays(mix_arrays(lists[[m]], [TOP_K]), relevant)
        standalone[name] = metrics
    names = sorted(KINDS, key=lambda name: -standalone[name][PRIMARY_METRIC])
    lists = lists[[list(KINDS).index(name) for name in names]]
    standalone_table = pd.DataFrame([{"model": n, "kind": KINDS[n], **standalone[n]} for n in names])
    standalone_table.to_csv(os.path.join(out_dir, "standalone_valid.csv"), index=False)
    print("--- Standalone validation metrics ---")
    print(standalone_table.to_string(index=False, float_format="%.4f"))

    shared, extra, exclusive = overlap_tables(lists, relevant, names)
    shared.to_csv(os.path.join(out_dir, "overlap_shared_top10_valid.csv"))
    extra.to_csv(os.path.join(out_dir, "overlap_extra_hits_valid.csv"))
    exclusive.to_csv(os.path.join(out_dir, "exclusive_hits_valid.csv"), index=False)
    print("\n--- Mean number of shared top-10 items per user (row vs column) ---")
    print(shared.to_string(float_format="%.1f"))
    print("\n--- Mean correct items per user in the row model's top-10 that the column model misses ---")
    print(extra.to_string(float_format="%.2f"))
    print("\n--- Correct items per user, and those found by no other model ---")
    print(exclusive.to_string(index=False, float_format="%.3f"))

    # Each worker imports recbole (and so torch), which costs a few hundred MB.
    with Pool(processes=min(N_WORKERS, os.cpu_count() or 1), initializer=_init_worker,
              initargs=(lists, relevant, folds)) as pool:
        # 2. Exhaustive search.
        order = tuple(range(len(names)))
        configs = [(order, quotas) for quotas in quota_vectors(len(names), TOP_K)]
        print(f"\n--- Exhaustive search: {len(configs)} configurations ---", flush=True)
        results = evaluate_all(pool, configs)
        search = Search(pool, names, configs, results)

        table = pd.DataFrame(results[:, :N_METRICS], columns=METRIC_NAMES)
        table.insert(0, "n_models", [bin(b).count("1") for b in search.bits])
        table.insert(0, "configuration", [search.label(*c) for c in configs])
        table = table.sort_values(PRIMARY_METRIC, ascending=False, kind="mergesort")
        table.to_csv(os.path.join(out_dir, "search_valid.csv.gz"), index=False)
        print(table.head(15).to_string(index=False, float_format="%.4f"))
        print("\nBest configuration per number of models:")
        print(table.groupby("n_models").head(1).sort_values("n_models").to_string(index=False, float_format="%.4f"))

        # 4. Which selection method generalises best? Select on one half, score on the other.
        print(f"\n--- Selection methods on {2 * N_REPEATS} validation half splits ---", flush=True)
        rows = []
        for fold in range(2 * N_REPEATS):
            select_col, held_col = N_METRICS + fold, N_METRICS + (fold ^ 1)
            for method in METHODS:
                (cfg_order, cfg_quotas), res = search.select(method, select_col)
                rows.append({
                    "fold": fold, "method": method, "configuration": search.label(cfg_order, cfg_quotas),
                    "selection_half_NDCG@10": res[select_col], "held_out_half_NDCG@10": res[held_col],
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
        summary.to_csv(os.path.join(out_dir, "split_half_summary_valid.csv"))
        print(summary.to_string(float_format="%.4f"))
        chosen_method = METHODS[0]
        for method in METHODS[1:]:
            if summary.loc[method, "held_out_mean"] > summary.loc[chosen_method, "held_out_mean"]:
                chosen_method = method
        print(f"Selection method with the best held-out {PRIMARY_METRIC}: {chosen_method}")

        # Apply every method to the full validation set (the chosen one becomes the final config).
        path = []
        alternatives = {}
        for method in METHODS:
            (cfg_order, cfg_quotas), res = (
                search.greedy(NDCG, path) if method == "Greedy" else search.select(method, NDCG)
            )
            used = [(names[m], int(q)) for m, q in zip(cfg_order, cfg_quotas) if q]
            alternatives[method] = {
                "models": [name for name, _ in used],
                "quotas": dict(used),
                "valid_metrics": dict(zip(METRIC_NAMES, map(float, res[:N_METRICS]))),
            }
        pd.DataFrame(path).to_csv(os.path.join(out_dir, "greedy_path_valid.csv"), index=False)
        print("\n--- Greedy path (full validation) ---")
        print(pd.DataFrame(path).to_string(index=False, float_format="%.4f"))

        final = copy.deepcopy(alternatives[chosen_method])  # a copy, so the YAML has no &id anchors
        support = sum(1 << names.index(n) for n in final["models"])
        order_configs, order_results = search.ordered(support)
        order_table = pd.DataFrame(order_results[:, :N_METRICS], columns=METRIC_NAMES)
        order_table.insert(0, "configuration", [search.label(*c) for c in order_configs])
        order_table = order_table.sort_values(PRIMARY_METRIC, ascending=False, kind="mergesort")
        order_table.to_csv(os.path.join(out_dir, "order_search_valid.csv"), index=False)

    # Cross-check the final configuration with the reference (pandas) implementation.
    # Export EXPORT_TOP_N items (the rerankers need candidates beyond the top-10); mixing is prefix-consistent,
    # so the top-10, and the metrics checked here, are the same as with top_k = 10.
    mixed = mix_recommendations(sources, final["models"], final["quotas"], top_k=EXPORT_TOP_N)
    reference = evaluate_recommendations(mixed, valid)
    for name in METRIC_NAMES:
        assert abs(reference[name] - final["valid_metrics"][name]) < 1e-9, (name, reference[name], final["valid_metrics"][name])
    save_recommendations(mixed, os.path.join(out_dir, "recs_valid.tsv"))

    print("\n--- Best configuration of each selection method (full validation) ---")
    for method in METHODS:
        alt = alternatives[method]
        print(f"{method:<11} {PRIMARY_METRIC}={alt['valid_metrics'][PRIMARY_METRIC]:.4f}  "
              + " ".join(f"{n}={q}" for n, q in alt["quotas"].items()))

    best_path = os.path.join(CONFIG_DIR, "mixed_hybrid_best.yaml")
    payload = {
        "models": final["models"],
        "quotas": final["quotas"],
        "primary_metric": PRIMARY_METRIC,
        "selection_method": chosen_method,
        "valid_metrics": final["valid_metrics"],
        "held_out_half_ndcg": {m: float(summary.loc[m, "held_out_mean"]) for m in METHODS},
        "alternatives": alternatives,
    }
    with open(best_path, "w", encoding="utf-8") as f:
        f.write(f"# Selected by {chosen_method}, valid {PRIMARY_METRIC} = "
                f"{final['valid_metrics'][PRIMARY_METRIC]:.4f} (written by tune_mixed_hybrid.py)\n")
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"\nFinal ({chosen_method}): " + " ".join(f"{n}={q}" for n, q in final["quotas"].items()))
    print(f"Wrote {best_path} and the tables in {out_dir}")


if __name__ == "__main__":
    main()
