"""Tune Hybrid Regression on VALIDATION scores. The test set is never opened.

The meta-learner is the one in HybridRegression: a StandardScaler and an L1/L2 logistic
regression on the base models' scores. Ranking uses the linear score, which orders items
the same way as the logistic probability.

Candidates are the nine tuned models. A configuration is a subset of at least two of them
plus (C, penalty, num_negatives). Single models are the BestSingle baseline, not hybrid
subsets. Model order does not matter: logistic regression has no round-robin.

Steps, all on validation:
1. Standalone metrics, top-10 overlap, and score correlation.
2. Exhaustive search: every subset of at least two models x every hyperparameter, fit on
   all validation users.
3. Choice of selection method. Four methods pick a configuration from that search:
   BestSingle, OnePerKind, Greedy, Exhaustive. Fitting and then picking the best
   validation score overfits those users, so validation users are split into random
   halves (N_REPEATS times). Each method refits on one half and is scored on the other.
   The method with the best mean held-out NDCG@10 is applied to the full validation set.

Prerequisite:
    python run_model.py SLIMElastic EASE RecVAE SimpleX BPR LightGCN UserKNN ItemKNN ContentBased

Example:
    python tune_hybrid_regression.py
"""

import copy
import os
from multiprocessing import get_context

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from recbole.model.general_recommender.hybrid_regression import (
    C_GRID,
    METRIC_NAMES,
    NUM_NEGATIVES_GRID,
    PENALTIES,
    PRIMARY_METRIC,
    TOP_K,
    combine_scores,
    design_matrix,
    evaluate_ranking,
    fit_ranking_weights,
    load_score_stack,
    sample_pairs,
    top_k_items,
)

# Same nine models as the mixed-hybrid retune, in the comparison-table order.
KINDS = {
    "SLIMElastic": "neighbourhood",
    "EASE": "neighbourhood",
    "RecVAE": "latent factor",
    "SimpleX": "latent factor",
    "BPR": "latent factor",
    "LightGCN": "latent factor",
    "UserKNN": "neighbourhood",
    "ItemKNN": "neighbourhood",
    "ContentBased": "content",
}
METHODS = ["BestSingle", "OnePerKind", "Greedy", "Exhaustive"]  # simplest first; ties keep the simpler one
N_REPEATS = 10  # random half splits -> 2 * N_REPEATS selection folds
SEED = 2020
N_WORKERS = 8
NDCG_COL = METRIC_NAMES.index(PRIMARY_METRIC)

_state = {}


# --------------------------------------------------------------------------- #
# Selection methods (operate on one already-fitted score per configuration)
# --------------------------------------------------------------------------- #
def best_index(scores, eligible):
    """Index of the best score among eligible rows. The earliest row wins a tie."""
    if not np.any(eligible):
        raise ValueError("no configuration is eligible")
    masked = np.where(eligible, scores, -np.inf)
    return int(np.argmax(masked))


def best_single_index(scores, masks):
    return best_index(scores, (masks & (masks - 1)) == 0)


def at_least_two(masks):
    """True for masks that contain at least two models. A single set bit is one model."""
    return (masks & (masks - 1)) != 0


def one_per_kind_index(scores, masks, names, kinds, standalone):
    """Best model of each kind by standalone NDCG, then the best mix of at least two of those."""
    allowed = 0
    for kind in sorted({kinds[name] for name in names}):
        members = [i for i, name in enumerate(names) if kinds[name] == kind]
        best = max(members, key=lambda i: (standalone[i], -i))
        allowed |= 1 << best
    eligible = ((masks & ~np.int64(allowed)) == 0) & at_least_two(masks)
    return best_index(scores, eligible)


def greedy_index(scores, masks, n_models, start_support=0, start_score=-np.inf):
    """Add the model that most improves NDCG, retuning hyperparameters at each step.

    start_support/start_score let the search begin from the best single model without that
    single model being a hybrid configuration. Stop when the next model does not improve.
    """
    support, score, chosen = int(start_support), float(start_score), None
    path = []
    while True:
        best_i, best_s = None, -np.inf
        for model in range(n_models):
            if support >> model & 1:
                continue
            eligible = masks == (support | (1 << model))
            if not eligible.any():
                continue
            i = best_index(scores, eligible)
            if scores[i] > best_s:
                best_i, best_s = i, float(scores[i])
        if best_i is None or best_s <= score:
            break
        path.append(best_i)
        support, score, chosen = int(masks[best_i]), best_s, best_i
    if chosen is None:
        chosen = best_index(scores, at_least_two(masks))
        path = [chosen]
    return chosen, path


def exhaustive_index(scores, masks=None):
    if masks is None:
        return int(np.argmax(scores))
    return best_index(scores, at_least_two(masks))


def select_index(method, scores, masks, names, kinds, standalone, single_scores=None):
    if method == "BestSingle":
        return best_single_index(scores, masks)
    if method == "OnePerKind":
        return one_per_kind_index(scores, masks, names, kinds, standalone)
    if method == "Greedy":
        baseline = standalone if single_scores is None else single_scores
        start = int(np.argmax(baseline))
        return greedy_index(
            scores, masks, len(names), start_support=1 << start, start_score=float(baseline[start])
        )[0]
    if method == "Exhaustive":
        return exhaustive_index(scores, masks)
    raise ValueError(method)


# --------------------------------------------------------------------------- #
# Parallel fitting
# --------------------------------------------------------------------------- #
def _init_worker(scores, designs, seen, relevant, select_masks, held_masks, columns, design_ids, Cs, penalties):
    _state.update(
        scores=scores,
        designs=designs,
        seen=seen,
        relevant=relevant,
        select_masks=select_masks,
        held_masks=held_masks,
        columns=columns,
        design_ids=design_ids,
        Cs=Cs,
        penalties=penalties,
    )


def _fit_chunk(spec):
    """(fold_id, start, stop) -> weights [n, M], metrics [n, 5], selection NDCG, held-out NDCG."""
    fold_id, start, stop = spec
    scores = _state["scores"]
    seen, relevant = _state["seen"], _state["relevant"]
    select_users = _state["select_masks"][fold_id]
    held_users = _state["held_masks"][fold_id]
    n_models = scores.shape[0]
    n = stop - start
    weights = np.zeros((n, n_models), dtype=np.float32)
    metrics = np.empty((n, len(METRIC_NAMES)), dtype=np.float64)
    selected = np.empty(n, dtype=np.float64)
    held = np.empty(n, dtype=np.float64)
    cached_rows = {}
    for offset, config_index in enumerate(range(start, stop)):
        design_id = int(_state["design_ids"][config_index])
        if design_id not in cached_rows:
            features, labels, users = _state["designs"][design_id]
            cached_rows[design_id] = (features, labels, select_users[users])
        features, labels, rows = cached_rows[design_id]
        cols = _state["columns"][config_index]
        ranking = fit_ranking_weights(
            features[rows][:, cols], labels[rows], _state["Cs"][config_index], _state["penalties"][config_index]
        )
        weights[offset, cols] = ranking
        combined = combine_scores(scores, weights[offset])
        metric, ndcg = evaluate_ranking(combined, seen, relevant)
        metrics[offset] = [metric[name] for name in METRIC_NAMES]
        selected[offset] = float(np.nanmean(ndcg[select_users]))
        held[offset] = float(np.nanmean(ndcg[held_users])) if held_users.any() else np.nan
    return weights, metrics, selected, held


def fit_fold(pool, fold_id, n_configs, desc, chunk=48, position=0):
    specs = [(fold_id, start, min(start + chunk, n_configs)) for start in range(0, n_configs, chunk)]
    parts = [[] for _ in range(4)]
    with tqdm(total=n_configs, desc=desc, unit="cfg", position=position, leave=False) as bar:
        for part in pool.imap(_fit_chunk, specs, chunksize=1):
            for bucket, array in zip(parts, part):
                bucket.append(array)
            bar.update(len(part[0]))
    return np.vstack(parts[0]), np.vstack(parts[1]), np.concatenate(parts[2]), np.concatenate(parts[3])


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def _pairs(frame, user_index, item_index):
    users = np.fromiter((user_index[u] for u in frame["user_id"].astype(str)), dtype=np.int32, count=len(frame))
    items = np.fromiter((item_index[i] for i in frame["item_id"].astype(str)), dtype=np.int32, count=len(frame))
    return users, items


def _mask(users, items, shape):
    mask = np.zeros(shape, dtype=bool)
    mask[users, items] = True
    return mask


def overlap_tables(scores, seen, relevant, names):
    """Shared top-10 items, and correct items the row model finds that the column model misses."""
    users = np.flatnonzero(relevant.any(axis=1))
    tops = np.stack([top_k_items(scores[m], seen, TOP_K)[0][users] for m in range(len(names))])
    hits = np.stack([relevant[users[:, None], tops[m]] for m in range(len(names))])
    n = len(names)
    shared, extra = np.zeros((n, n)), np.zeros((n, n))
    in_other = np.empty((n, n) + tops.shape[1:], dtype=bool)
    for a in range(n):
        for b in range(n):
            in_other[a, b] = (tops[a][:, :, None] == tops[b][:, None, :]).any(axis=2)
            shared[a, b] = in_other[a, b].sum(axis=1).mean()
            extra[a, b] = (hits[a] & ~in_other[a, b]).sum(axis=1).mean()
    exclusive, hits_per_user = [], []
    for a in range(n):
        others = in_other[a, [b for b in range(n) if b != a]].any(axis=0)
        exclusive.append(float((hits[a] & ~others).sum(axis=1).mean()))
        hits_per_user.append(float(hits[a].sum(axis=1).mean()))
    return (
        pd.DataFrame(shared, index=names, columns=names),
        pd.DataFrame(extra, index=names, columns=names),
        pd.DataFrame({"model": names, "hits_per_user": hits_per_user, "exclusive_hits_per_user": exclusive}),
    )


def score_correlation(scores, seen, relevant, names):
    """Pearson correlation of raw scores on validation users' unseen items."""
    users = np.flatnonzero(relevant.any(axis=1))
    unseen = np.zeros(seen.shape, dtype=bool)
    unseen[users] = ~seen[users]
    corr = np.corrcoef(scores[:, unseen])
    return pd.DataFrame(corr, index=names, columns=names)


def config_label(names, columns, C, penalty, num_negatives):
    models = "+".join(names[i] for i in columns)
    return f"{models} | C={C:g} penalty={penalty} neg={num_negatives}"


def _plain(value):
    """Convert NumPy scalars so PyYAML can dump the configuration."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def pack_choice(names, columns, C, penalty, num_negatives, weight_row, metric_row):
    raw = {names[i]: float(weight_row[i]) for i in columns}
    total = sum(raw.values())
    normalized = {key: value / total for key, value in raw.items()} if total > 0 else dict(raw)
    return {
        "models": [names[i] for i in columns],
        "hyperparams": {"C": float(C), "penalty": str(penalty), "num_negatives": int(num_negatives)},
        "ranking_weights": raw,
        "normalized_weights": normalized,
        "valid_metrics": {name: float(metric_row[i]) for i, name in enumerate(METRIC_NAMES)},
    }


def recommendation_frame(combined, seen, user_ids, item_ids, user_mask):
    chosen, chosen_scores = top_k_items(combined, seen, TOP_K)
    rows = []
    for user in np.flatnonzero(user_mask):
        for rank, (item, score) in enumerate(zip(chosen[user], chosen_scores[user]), start=1):
            if not np.isfinite(score):
                continue
            rows.append((user_ids[user], item_ids[item], rank, float(score)))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rank", "score"])


def main():
    from recbole.model.general_recommender.mixed_hybrid import evaluate_recommendations
    from recbole.utils.experiment import CONFIG_DIR, OUTPUT_DIR, SPLIT_DIR

    out_dir = os.path.join(OUTPUT_DIR, "HybridRegression")
    os.makedirs(out_dir, exist_ok=True)

    def read_split(name):
        # The test file is intentionally not read.
        return pd.read_csv(
            os.path.join(SPLIT_DIR, f"{name}.tsv"), sep="\t", dtype={"user_id": str, "item_id": str}
        )

    train, valid = read_split("train"), read_split("valid")
    names = list(KINDS)
    scores, user_ids, item_ids = load_score_stack(names)
    # ContentBased stores -inf on training items (filter_interacted). Those cells are masked
    # again at ranking time, and the regression samples never use them.
    if np.isnan(scores).any():
        raise RuntimeError("a score matrix contains NaN")
    user_index = {user: i for i, user in enumerate(user_ids)}
    item_index = {item: i for i, item in enumerate(item_ids)}
    train_u, train_i = _pairs(train, user_index, item_index)
    valid_u, valid_i = _pairs(valid, user_index, item_index)
    seen = _mask(train_u, train_i, scores.shape[1:])  # hidden at validation ranking time
    relevant = _mask(valid_u, valid_i, scores.shape[1:])
    forbidden = seen | relevant  # negatives are items in neither train nor validation

    standalone = [evaluate_ranking(scores[m], seen, relevant)[0] for m in range(len(names))]
    order = sorted(range(len(names)), key=lambda i: -standalone[i][PRIMARY_METRIC])
    names = [names[i] for i in order]
    scores = scores[order]
    standalone = [standalone[i] for i in order]
    table = pd.DataFrame([{"model": name, "kind": KINDS[name], **metrics} for name, metrics in zip(names, standalone)])
    table.to_csv(os.path.join(out_dir, "standalone_valid.csv"), index=False)
    print("--- Standalone validation metrics ---")
    print(table.to_string(index=False, float_format="%.4f"))

    shared, extra, exclusive = overlap_tables(scores, seen, relevant, names)
    shared.to_csv(os.path.join(out_dir, "overlap_shared_top10_valid.csv"))
    extra.to_csv(os.path.join(out_dir, "overlap_extra_hits_valid.csv"))
    exclusive.to_csv(os.path.join(out_dir, "exclusive_hits_valid.csv"), index=False)
    correlation = score_correlation(scores, seen, relevant, names)
    correlation.to_csv(os.path.join(out_dir, "score_correlation_valid.csv"))
    print("\n--- Mean number of shared top-10 items per validation user ---")
    print(shared.to_string(float_format="%.1f"))
    print("\n--- Correct top-10 items the row model finds and the column model misses ---")
    print(extra.to_string(float_format="%.2f"))
    print("\n--- Correct items per user, and those found by no other model ---")
    print(exclusive.to_string(index=False, float_format="%.3f"))
    print("\n--- Correlation of raw scores on validation users' unseen items ---")
    print(correlation.to_string(float_format="%.2f"))

    designs = []
    for num_negatives in NUM_NEGATIVES_GRID:
        rng = np.random.default_rng(SEED + num_negatives)
        users, items, labels = sample_pairs(
            valid_u, valid_i, forbidden, num_negatives, scores.shape[2], rng
        )
        designs.append((design_matrix(scores, users, items), labels, users))
        print(f"Sampled {len(labels)} pairs for num_negatives={num_negatives} "
              f"({int(labels.sum())} positives)", flush=True)

    columns, design_ids, Cs, penalties, masks = [], [], [], [], []
    neg_to_design = {n: i for i, n in enumerate(NUM_NEGATIVES_GRID)}
    for mask in range(1, 1 << len(names)):
        if (mask & (mask - 1)) == 0:
            continue  # a single model is the BestSingle baseline, not a hybrid subset
        cols = tuple(i for i in range(len(names)) if mask >> i & 1)
        for num_negatives in NUM_NEGATIVES_GRID:
            for C in C_GRID:
                for penalty in PENALTIES:
                    columns.append(cols)
                    design_ids.append(neg_to_design[num_negatives])
                    Cs.append(C)
                    penalties.append(penalty)
                    masks.append(mask)
    design_ids = np.asarray(design_ids, dtype=np.int16)
    Cs = np.asarray(Cs, dtype=np.float64)
    penalties = np.asarray(penalties)
    masks = np.asarray(masks, dtype=np.int16)
    n_configs = len(columns)
    n_subsets = (1 << len(names)) - 1 - len(names)
    print(f"\n--- Exhaustive search: {n_configs} configurations "
          f"({n_subsets} subsets of at least 2 models x "
          f"{len(NUM_NEGATIVES_GRID) * len(C_GRID) * len(PENALTIES)} hyperparams) ---",
          flush=True)

    valid_users = np.flatnonzero(relevant.any(axis=1))
    n_folds = 2 * N_REPEATS
    select_masks = np.zeros((1 + n_folds, scores.shape[1]), dtype=bool)
    held_masks = np.zeros_like(select_masks)
    select_masks[0, valid_users] = True
    rng = np.random.default_rng(SEED)
    for repeat in range(N_REPEATS):
        perm = rng.permutation(valid_users)
        mid = len(perm) // 2
        left, right = perm[:mid], perm[mid:]
        select_masks[1 + 2 * repeat, left] = True
        held_masks[1 + 2 * repeat, right] = True
        select_masks[2 + 2 * repeat, right] = True
        held_masks[2 + 2 * repeat, left] = True

    standalone_ndcg = np.array([metrics[PRIMARY_METRIC] for metrics in standalone])
    single_ndcg = np.stack([evaluate_ranking(scores[m], seen, relevant)[1] for m in range(len(names))])

    def single_means(user_mask):
        return np.array([float(np.nanmean(single_ndcg[m, user_mask])) for m in range(len(names))])
    workers = min(N_WORKERS, os.cpu_count() or 1)
    with get_context("spawn").Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(scores, designs, seen, relevant, select_masks, held_masks, columns, design_ids, Cs, penalties),
    ) as pool:
        print(f"Fitting on all validation users with {workers} workers", flush=True)
        weights, metrics, _, _ = fit_fold(pool, 0, n_configs, desc="full validation")
        labels = [
            config_label(names, columns[i], Cs[i], penalties[i], NUM_NEGATIVES_GRID[design_ids[i]])
            for i in range(n_configs)
        ]
        search = pd.DataFrame(metrics, columns=METRIC_NAMES)
        search.insert(0, "num_negatives", [NUM_NEGATIVES_GRID[i] for i in design_ids])
        search.insert(0, "penalty", penalties)
        search.insert(0, "C", Cs)
        search.insert(0, "n_models", [len(cols) for cols in columns])
        search.insert(0, "configuration", labels)
        for model, column in enumerate(names):
            search[f"w_{column}"] = weights[:, model]
        search = search.sort_values(PRIMARY_METRIC, ascending=False, kind="mergesort")
        search_path = os.path.join(out_dir, "search_valid.csv.gz")
        search.to_csv(search_path, index=False)
        print(search.head(15).to_string(index=False, float_format="%.4f"))
        print("\nBest configuration per number of models:")
        per_size = search.groupby("n_models", sort=True).head(1).sort_values("n_models")
        print(per_size.to_string(index=False, float_format="%.4f"))

        print(f"\n--- Selection methods on {n_folds} validation half splits ---", flush=True)
        rows = []
        for fold in tqdm(range(1, 1 + n_folds), desc="validation halves", unit="fold", position=0):
            _, _, selected, held_out = fit_fold(
                pool, fold, n_configs, desc=f"fold {fold}/{n_folds}", position=1
            )
            sel_means = single_means(select_masks[fold])
            held_means = single_means(held_masks[fold])
            for method in METHODS:
                if method == "BestSingle":
                    model = int(np.argmax(sel_means))
                    rows.append({
                        "fold": fold,
                        "method": method,
                        "configuration": names[model],
                        "selection_half_NDCG@10": sel_means[model],
                        "held_out_half_NDCG@10": held_means[model],
                    })
                    continue
                chosen = select_index(
                    method, selected, masks, names, KINDS, standalone_ndcg, single_scores=sel_means
                )
                rows.append({
                    "fold": fold,
                    "method": method,
                    "configuration": labels[chosen],
                    "selection_half_NDCG@10": float(selected[chosen]),
                    "held_out_half_NDCG@10": float(held_out[chosen]),
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
        base = split_half.loc[split_half["method"] == "BestSingle"].set_index("fold")["held_out_half_NDCG@10"]
        deltas, wins = [], []
        for method in METHODS:
            other = split_half.loc[split_half["method"] == method].set_index("fold")["held_out_half_NDCG@10"]
            delta = other - base
            deltas.append(float(delta.mean()))
            wins.append(int((delta > 0).sum()))
        summary["mean_delta_vs_best_single"] = deltas
        summary["folds_better_than_best_single"] = wins
        summary.to_csv(os.path.join(out_dir, "split_half_summary_valid.csv"))
        print(summary.to_string(float_format="%.4f"))
        chosen_method = METHODS[0]
        for method in METHODS[1:]:
            if summary.loc[method, "held_out_mean"] > summary.loc[chosen_method, "held_out_mean"]:
                chosen_method = method
        print(f"Selection method with the best held-out {PRIMARY_METRIC}: {chosen_method}")

    full_scores = metrics[:, NDCG_COL]
    alternatives = {}
    start = int(np.argmax(standalone_ndcg))
    greedy_chosen, greedy_path = greedy_index(
        full_scores, masks, len(names),
        start_support=1 << start, start_score=float(standalone_ndcg[start]),
    )
    path_rows = []
    previous = float(standalone_ndcg[start])
    previous_mask = 1 << start
    for step, index in enumerate(greedy_path, start=1):
        added_bit = int(masks[index]) ^ previous_mask
        added = "+".join(names[i] for i in range(len(names)) if added_bit >> i & 1)
        path_rows.append({
            "step": step,
            "added": added,
            "configuration": labels[index],
            PRIMARY_METRIC: float(full_scores[index]),
            "gain": float(full_scores[index] - previous),
        })
        previous = float(full_scores[index])
        previous_mask = int(masks[index])
    pd.DataFrame(path_rows).to_csv(os.path.join(out_dir, "greedy_path_valid.csv"), index=False)
    print("\n--- Greedy path (full validation) ---")
    print(pd.DataFrame(path_rows).to_string(index=False, float_format="%.4f"))

    for method in METHODS:
        if method == "BestSingle":
            alternatives[method] = {
                "models": [names[start]],
                "hyperparams": None,
                "ranking_weights": {names[start]: 1.0},
                "normalized_weights": {names[start]: 1.0},
                "valid_metrics": {key: float(value) for key, value in standalone[start].items()},
            }
            continue
        index = greedy_chosen if method == "Greedy" else select_index(
            method, full_scores, masks, names, KINDS, standalone_ndcg
        )
        alternatives[method] = pack_choice(
            names, columns[index], Cs[index], penalties[index],
            NUM_NEGATIVES_GRID[design_ids[index]], weights[index], metrics[index],
        )

    final = copy.deepcopy(alternatives[chosen_method])
    weight_vector = np.zeros(len(names), dtype=np.float32)
    for name, weight in final["ranking_weights"].items():
        weight_vector[names.index(name)] = weight
    combined = combine_scores(scores, weight_vector)
    user_mask = relevant.any(axis=1)
    mixed = recommendation_frame(combined, seen, user_ids, item_ids, user_mask)
    reference = evaluate_recommendations(mixed, valid, k=TOP_K)
    for name in METRIC_NAMES:
        if abs(reference[name] - final["valid_metrics"][name]) > 1e-8:
            raise AssertionError((name, reference[name], final["valid_metrics"][name]))
    recs_path = os.path.join(out_dir, "recs_valid.tsv")
    mixed.to_csv(recs_path, sep="\t", index=False)

    print("\n--- Best configuration of each selection method (full validation) ---")
    for method in METHODS:
        alt = alternatives[method]
        weights_text = " ".join(f"{name}={weight:.3f}" for name, weight in alt["normalized_weights"].items())
        hp = alt["hyperparams"]
        label = alt["models"][0] if hp is None else config_label(
            names, [names.index(m) for m in alt["models"]], hp["C"], hp["penalty"], hp["num_negatives"]
        )
        print(f"{method:<11} {PRIMARY_METRIC}={alt['valid_metrics'][PRIMARY_METRIC]:.4f}  "
              f"{label}  [{weights_text}]")

    payload = {
        "models": final["models"],
        "hyperparams": final["hyperparams"],
        "ranking_weights": final["ranking_weights"],
        "normalized_weights": final["normalized_weights"],
        "primary_metric": PRIMARY_METRIC,
        "selection_method": chosen_method,
        "valid_metrics": final["valid_metrics"],
        "held_out_half_ndcg": {method: float(summary.loc[method, "held_out_mean"]) for method in METHODS},
        "alternatives": alternatives,
    }
    best_path = os.path.join(CONFIG_DIR, "hybrid_regression_best.yaml")
    with open(best_path, "w", encoding="utf-8") as handle:
        handle.write(
            f"# Selected by {chosen_method}, valid {PRIMARY_METRIC} = "
            f"{final['valid_metrics'][PRIMARY_METRIC]:.4f} (written by tune_hybrid_regression.py)\n"
        )
        yaml.safe_dump(_plain(payload), handle, sort_keys=False)
    print(f"\nFinal ({chosen_method}): {payload['models']} {payload['hyperparams']}")
    print(f"Wrote {best_path}")
    print(f"Wrote {recs_path} and the tables in {out_dir}")


if __name__ == "__main__":
    main()
