"""Analyse how the base recommenders contribute to each tuned hybrid.

Reads the frozen configurations (configs/mixed_hybrid_best.yaml, configs/hybrid_regression_best.yaml,
configs/hybrid_feature_aug_best.yaml) and the base models' exported outputs. Nothing here selects or
changes a configuration: where something is re-tuned (the ablations), it is re-tuned on VALIDATION
with the same procedure as the hybrid's own tuning, and only then evaluated on test.

Mixed Hybrid
    attribution   which model supplied each recommended item, and which of them are hits
    ablation      drop one model, re-tune order + quotas of the rest on validation, evaluate on test

Hybrid Regression
    coefficients  ranking weights, standardised coefficients, and each model's share of the
                  recommended items' scores
    ablation      drop one model, re-tune (C, penalty, num_negatives) on validation, evaluate on test
    stability     standardised coefficients refitted on the 20 selection halves of the tuner
    complementarity  overlap between the hybrid's top-10 and each component's top-10

Feature Augmentation
    ablation      drop one source (or all), or replace the target; re-tune everything else
                  (target, its hyperparameters, n_pseudo, pseudo_weight) on validation, evaluate on test
    complementarity  overlap between the hybrid's top-10 and the top-10 of each source and of the
                  plain target (same target and hyperparameters, no pseudo-interactions)
    pseudo-items  how many of the recommended items were added as pseudo-interactions, by which
                  source, and how often they are hits; which items augmentation adds to and removes
                  from the plain target's top-10

Prerequisite: the tuned outputs of the base models (python run_model.py ...).

Examples:
    python analyze_hybrids.py                          # all hybrids (~20 min, mostly Feature Augmentation)
    python analyze_hybrids.py --hybrids feature_aug
"""

import argparse
import itertools
import os
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from recbole.utils.experiment import CONFIG_DIR, OUTPUT_DIR, load_recs, load_split
from recbole.model.general_recommender import hybrid_regression as hr
from recbole.model.general_recommender.mixed_hybrid import (
    PRIMARY_METRIC,
    TOP_K,
    evaluate_arrays,
    evaluate_recommendations,
    mix_arrays,
    mix_recommendations,
    to_arrays,
)
from tune_hybrid_regression import KINDS as REGRESSION_CANDIDATES
from tune_hybrid_regression import N_REPEATS, SEED, score_correlation
import tune_hybrid_feature_aug as fa

OUT_DIR = os.path.join(OUTPUT_DIR, "HybridAnalysis")
HYBRIDS = ("mixed", "regression", "feature_aug")
METRICS = [f"{m}@{TOP_K}" for m in ("Recall", "MRR", "NDCG", "Hit", "Precision")]


def save(table, name, title):
    table.to_csv(os.path.join(OUT_DIR, name), index=False)
    print(f"\n--- {title} ({name}) ---")
    print(table.to_string(index=False, float_format="%.4f"))


def load_config(name):
    with open(os.path.join(CONFIG_DIR, name), encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Mixed Hybrid
# --------------------------------------------------------------------------- #
def mixed_attribution(cfg, split):
    """Per source model: share of slots and of hits, hit rate of its slots, unique hits, own rank."""
    models, quotas = cfg["models"], cfg["quotas"]
    rows, by_rank = [], []
    for stage, truth in (("valid", split["valid"]), ("test", split["test"])):
        sources = {m: load_recs(m, stage) for m in models}
        mixed = mix_recommendations(sources, models, quotas)
        relevant = set(zip(truth["user_id"].astype(str), truth["item_id"].astype(str)))
        pairs = list(zip(mixed["user_id"].astype(str), mixed["item_id"].astype(str)))
        mixed["hit"] = [pair in relevant for pair in pairs]
        own_rank = {m: dict(zip(zip(s["user_id"].astype(str), s["item_id"].astype(str)), s["rank"]))
                    for m, s in sources.items()}
        top10 = {m: set(zip(s.loc[s["rank"] <= TOP_K, "user_id"].astype(str), s.loc[s["rank"] <= TOP_K, "item_id"].astype(str)))
                 for m, s in sources.items()}
        mixed["source_rank"] = [own_rank[s][p] for s, p in zip(mixed["source"], pairs)]
        mixed["unique_hit"] = [
            h and not any(p in top10[o] for o in models if o != s)
            for h, s, p in zip(mixed["hit"], mixed["source"], pairs)
        ]
        n_hits = mixed["hit"].sum()
        for m in models:
            part = mixed[mixed["source"] == m]
            rows.append({
                "stage": stage, "model": m, "quota": quotas[m],
                "slot_share": len(part) / len(mixed),
                "hits": int(part["hit"].sum()), "hit_share": part["hit"].sum() / n_hits,
                "hit_rate_of_slots": part["hit"].mean(),
                "unique_hits": int(part["unique_hit"].sum()),
                "mean_rank_in_own_list": part["source_rank"].mean(),
            })
        rate = mixed.groupby("rank")["hit"].mean()
        by_rank.append(pd.DataFrame({"stage": stage, "rank": rate.index, "hit_rate": rate.values}))
    save(pd.DataFrame(rows), "mixed_attribution.csv", "Mixed Hybrid: contribution of each source model")
    save(pd.concat(by_rank), "mixed_hit_rate_by_rank.csv", "Mixed Hybrid: hit rate by final rank")


def _best_mix_on_valid(names, valid):
    """Best order + positive quotas (sum <= 10) of the given models on validation, as in tune_mixed_hybrid.py."""
    users = sorted(valid["user_id"].astype(str).unique())
    lists, item_ids = to_arrays({m: load_recs(m, "valid") for m in names}, names, users)
    index = {item: i for i, item in enumerate(item_ids)}
    relevant = np.zeros((len(users), len(item_ids) + 1), dtype=bool)
    user_index = {u: i for i, u in enumerate(users)}
    for u, i in zip(valid["user_id"].astype(str), valid["item_id"].astype(str)):
        relevant[user_index[u], index.get(i, len(item_ids))] = True
    best = None
    for order in itertools.permutations(range(len(names))):
        for quotas in itertools.product(range(1, TOP_K + 1), repeat=len(names)):
            if sum(quotas) > TOP_K:
                continue
            metrics, _ = evaluate_arrays(mix_arrays(lists[list(order)], quotas), relevant)
            if best is None or metrics[PRIMARY_METRIC] > best[0]:
                best = (metrics[PRIMARY_METRIC], [names[o] for o in order], {names[o]: int(q) for o, q in zip(order, quotas)})
    return best


def mixed_ablation(cfg, split):
    models, quotas = cfg["models"], cfg["quotas"]
    full_test = evaluate_recommendations(
        mix_recommendations({m: load_recs(m, "test") for m in models}, models, quotas), split["test"])
    rows = [{"removed": "none (full hybrid)", "configuration": " ".join(f"{m}={quotas[m]}" for m in models),
             f"valid_{PRIMARY_METRIC}": cfg["valid_metrics"][PRIMARY_METRIC], **{f"test_{k}": v for k, v in full_test.items()},
             f"delta_test_{PRIMARY_METRIC}": 0.0}]
    for removed in models:
        rest = [m for m in models if m != removed]
        valid_score, order, q = _best_mix_on_valid(rest, split["valid"])
        test = evaluate_recommendations(mix_recommendations({m: load_recs(m, "test") for m in order}, order, q), split["test"])
        rows.append({"removed": removed, "configuration": " ".join(f"{m}={q[m]}" for m in order),
                     f"valid_{PRIMARY_METRIC}": valid_score, **{f"test_{k}": v for k, v in test.items()},
                     f"delta_test_{PRIMARY_METRIC}": test[PRIMARY_METRIC] - full_test[PRIMARY_METRIC]})
    save(pd.DataFrame(rows), "mixed_ablation.csv", "Mixed Hybrid: leave-one-out ablation (re-tuned on validation)")


# --------------------------------------------------------------------------- #
# Hybrid Regression
# --------------------------------------------------------------------------- #
class RegressionData:
    """Score matrices, masks and sampled training pairs, built exactly as in tune_hybrid_regression.py."""

    def __init__(self, split):
        self.names = list(REGRESSION_CANDIDATES)
        self.scores, self.user_ids, self.item_ids = hr.load_score_stack(self.names)
        user_index = {u: i for i, u in enumerate(self.user_ids)}
        item_index = {i: k for k, i in enumerate(self.item_ids)}

        def mask(frame):
            m = np.zeros(self.scores.shape[1:], dtype=bool)
            m[[user_index[u] for u in frame["user_id"].astype(str)], [item_index[i] for i in frame["item_id"].astype(str)]] = True
            return m

        def pairs(frame):
            return (np.array([user_index[u] for u in frame["user_id"].astype(str)], dtype=np.int32),
                    np.array([item_index[i] for i in frame["item_id"].astype(str)], dtype=np.int32))

        self.train, self.valid, self.test = mask(split["train"]), mask(split["valid"]), mask(split["test"])
        valid_u, valid_i = pairs(split["valid"])
        forbidden = self.train | self.valid
        self.designs = {}
        for n in hr.NUM_NEGATIVES_GRID:
            users, items, labels = hr.sample_pairs(valid_u, valid_i, forbidden, n, self.scores.shape[2],
                                                   np.random.default_rng(SEED + n))
            self.designs[n] = (users, items, labels)

        # The tuner's half splits: fold 2r+1 selects on `left`, fold 2r+2 on `right`.
        valid_users = np.flatnonzero(self.valid.any(axis=1))
        rng = np.random.default_rng(SEED)
        self.selection_halves = []
        for _ in range(N_REPEATS):
            perm = rng.permutation(valid_users)
            mid = len(perm) // 2
            for half in (perm[:mid], perm[mid:]):
                m = np.zeros(self.scores.shape[1], dtype=bool)
                m[half] = True
                self.selection_halves.append(m)

    def columns(self, models):
        return [self.names.index(m) for m in models]

    def fit(self, models, C, penalty, num_negatives, user_mask=None):
        """(ranking weights, standardised coefficients, scaler) for these models, as in fit_ranking_weights."""
        users, items, labels = self.designs[num_negatives]
        rows = np.ones(len(users), dtype=bool) if user_mask is None else user_mask[users]
        features = hr.design_matrix(self.scores[self.columns(models)], users[rows], items[rows]).astype(np.float64)
        scaler = StandardScaler().fit(features)
        model = hr._logistic(C, penalty)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(scaler.transform(features), labels[rows])
        scale = scaler.scale_.copy()
        scale[scale == 0] = 1.0
        standardised = model.coef_.ravel()
        return standardised / scale, standardised, scaler

    def combined(self, models, weights):
        return hr.combine_scores(self.scores[self.columns(models)], np.asarray(weights, dtype=np.float32))

    def evaluate(self, combined, stage):
        seen, relevant = (self.train, self.valid) if stage == "valid" else (self.train | self.valid, self.test)
        return hr.evaluate_ranking(combined, seen, relevant)[0]

    def top10(self, combined, stage):
        seen = self.train if stage == "valid" else self.train | self.valid
        return hr.top_k_items(combined, seen, TOP_K)[0]


def regression_coefficients(cfg, data):
    models, hp = cfg["models"], cfg["hyperparams"]
    ranking, standardised, scaler = data.fit(models, hp["C"], hp["penalty"], hp["num_negatives"])
    stored = np.array([cfg["ranking_weights"][m] for m in models])
    if not np.allclose(ranking, stored, rtol=1e-3, atol=1e-4):
        print(f"WARNING: refitted weights {np.round(ranking, 4)} differ from the stored ones {np.round(stored, 4)}; "
              "the base model outputs probably differ from the ones the hybrid was tuned on.")

    # Contribution of each model to the score of every recommended test item:
    # term_m = standardised coefficient_m * (score_m - mean_m) / scale_m.
    combined = data.combined(models, stored)
    chosen = data.top10(combined, "test")
    users = np.repeat(np.arange(chosen.shape[0]), TOP_K)
    items = chosen.ravel()
    features = hr.design_matrix(data.scores[data.columns(models)], users, items).astype(np.float64)
    terms = scaler.transform(features) * standardised
    share = np.abs(terms) / np.abs(terms).sum(axis=1, keepdims=True)
    hits = data.test[users, items]
    rows = []
    for k, m in enumerate(models):
        rows.append({
            "model": m,
            "ranking_weight": stored[k],
            "standardised_coefficient": standardised[k],
            "mean_share_of_score": share[:, k].mean(),
            "mean_term": terms[:, k].mean(),
            "mean_term_on_hits": terms[hits, k].mean(),
            "mean_term_on_misses": terms[~hits, k].mean(),
            "largest_term_share": float((np.abs(terms).argmax(axis=1) == k).mean()),
        })
    save(pd.DataFrame(rows), "regression_coefficients.csv",
         "Hybrid Regression: coefficients and contribution to recommended test items")
    # Over all unseen items, as in the tuner (restricting to recommended items would bias it,
    # since those items were chosen for having high scores).
    corr = score_correlation(data.scores[data.columns(models)], data.train | data.valid, data.test, models)
    save(corr.reset_index().rename(columns={"index": "model"}), "regression_score_correlation.csv",
         "Hybrid Regression: score correlation on unseen test items")
    return standardised


def _best_hyperparams_on_valid(data, models):
    best = None
    for n in hr.NUM_NEGATIVES_GRID:
        for C in hr.C_GRID:
            for penalty in hr.PENALTIES:
                weights, _, _ = data.fit(models, C, penalty, n)
                score = data.evaluate(data.combined(models, weights), "valid")[PRIMARY_METRIC]
                if best is None or score > best[0]:
                    best = (score, {"C": C, "penalty": penalty, "num_negatives": n}, weights)
    return best


def regression_ablation(cfg, data, standardised):
    models, hp = cfg["models"], cfg["hyperparams"]
    full = data.evaluate(data.combined(models, [cfg["ranking_weights"][m] for m in models]), "test")
    rows = [{"removed": "none (full hybrid)", "standardised_coefficient": np.nan,
             "hyperparams": f"C={hp['C']:g} {hp['penalty']} neg={hp['num_negatives']}",
             f"valid_{PRIMARY_METRIC}": cfg["valid_metrics"][PRIMARY_METRIC],
             **{f"test_{k}": v for k, v in full.items()}, f"delta_test_{PRIMARY_METRIC}": 0.0}]
    for k, removed in enumerate(models):
        rest = [m for m in models if m != removed]
        valid_score, best_hp, weights = _best_hyperparams_on_valid(data, rest)
        test = data.evaluate(data.combined(rest, weights), "test")
        rows.append({"removed": removed, "standardised_coefficient": standardised[k],
                     "hyperparams": f"C={best_hp['C']:g} {best_hp['penalty']} neg={best_hp['num_negatives']}",
                     f"valid_{PRIMARY_METRIC}": valid_score, **{f"test_{m}": v for m, v in test.items()},
                     f"delta_test_{PRIMARY_METRIC}": test[PRIMARY_METRIC] - full[PRIMARY_METRIC]})
    save(pd.DataFrame(rows), "regression_ablation.csv",
         "Hybrid Regression: leave-one-out ablation (re-tuned on validation)")


def regression_stability(cfg, data, standardised):
    models, hp = cfg["models"], cfg["hyperparams"]
    folds = np.array([data.fit(models, hp["C"], hp["penalty"], hp["num_negatives"], half)[1]
                      for half in data.selection_halves])
    rows = []
    for k, m in enumerate(models):
        signs = np.sign(folds[:, k])
        rows.append({
            "model": m, "full_validation": standardised[k],
            "fold_mean": folds[:, k].mean(), "fold_std": folds[:, k].std(ddof=1),
            "fold_min": folds[:, k].min(), "fold_max": folds[:, k].max(),
            "folds_same_sign_as_full": int((signs == np.sign(standardised[k])).sum()),
            "n_folds": len(folds),
        })
    save(pd.DataFrame(rows), "regression_coefficient_stability.csv",
         "Hybrid Regression: standardised coefficients on the 20 selection halves (final configuration)")


def regression_complementarity(cfg, data):
    """How much of the hybrid's top-10, and of its hits, each component's own top-10 contains."""
    models = cfg["models"]
    hybrid = data.top10(data.combined(models, [cfg["ranking_weights"][m] for m in models]), "test")
    users = np.arange(hybrid.shape[0])[:, None]
    hybrid_hits = data.test[users, hybrid]
    active = data.test.any(axis=1)
    in_any = np.zeros_like(hybrid_hits)
    rows = []
    for m in models:
        own = data.top10(data.scores[data.names.index(m)], "test")
        shared = (hybrid[:, :, None] == own[:, None, :]).any(axis=2)
        in_any |= shared
        rows.append({
            "component": m,
            "shared_top10_items": shared[active].sum(axis=1).mean(),
            "share_of_hybrid_hits_in_its_top10": (hybrid_hits & shared).sum() / hybrid_hits.sum(),
            "its_own_hits": int(data.test[users, own][active].sum()),
        })
    rows.append({
        "component": "none of them",
        "shared_top10_items": (~in_any)[active].sum(axis=1).mean(),
        "share_of_hybrid_hits_in_its_top10": (hybrid_hits & ~in_any).sum() / hybrid_hits.sum(),
        "its_own_hits": int(hybrid_hits.sum()),
    })
    save(pd.DataFrame(rows), "regression_complementarity.csv",
         "Hybrid Regression: overlap of the hybrid's test top-10 with each component's top-10 "
         "('none of them': items in no component's top-10; its_own_hits = the hybrid's hits)")


# --------------------------------------------------------------------------- #
# Feature Augmentation
# --------------------------------------------------------------------------- #
class FeatureAugData:
    """Sources, training matrix and fitting code exactly as in tune_hybrid_feature_aug.py.

    Every configuration is fitted with the tuner's own worker code (fa._init_worker,
    fa._evaluate_task, fa.fit_configuration), in this process.
    """

    def __init__(self, split):
        self.names = list(fa.KINDS)
        self.scores, self.user_ids, self.item_ids = hr.load_score_stack(self.names)
        user_index = {u: i for i, u in enumerate(self.user_ids)}
        item_index = {i: k for k, i in enumerate(self.item_ids)}

        def mask(frame):
            m = np.zeros(self.scores.shape[1:], dtype=bool)
            m[frame["user_id"].astype(str).map(user_index), frame["item_id"].astype(str).map(item_index)] = True
            return m

        self.train, self.valid, self.test = mask(split["train"]), mask(split["valid"]), mask(split["test"])

        self.targets = list(fa.TARGET_GRIDS)
        config, train_data = fa.load_recbole_train(self.targets[0])
        dataset = train_data._dataset
        uid, iid = dataset.uid_field, dataset.iid_field
        if (list(dataset.id2token(uid, np.arange(1, dataset.user_num))) != list(self.user_ids)
                or list(dataset.id2token(iid, np.arange(1, dataset.item_num))) != list(self.item_ids)):
            raise RuntimeError("RecBole's ids differ from outputs/<model>/scores.npz; re-run run_model.py")
        matrix = dataset.inter_matrix(form="csr").astype(np.float32)
        matrix.data[:] = 1.0
        if not np.array_equal(matrix[1:, 1:].toarray().astype(bool), self.train):
            raise RuntimeError("RecBole's training split differs from outputs/split/train.tsv")
        fields = {key: config[key] for key in ("USER_ID_FIELD", "ITEM_ID_FIELD", "NEG_PREFIX")}

        sources = [fa.FixedScores(self.scores[m], fields["USER_ID_FIELD"]) for m in range(len(self.names))]
        ranker = fa.HybridFeatureAugmentation(
            fa.TargetConfig(device=fa.torch.device("cpu"), **fields), fa.TrainMatrix(matrix, uid, iid), sources)
        self.rankings = [ranker._rank_candidates(s, max(fa.N_PSEUDO_GRID)) for s in sources]
        self.param_lists = {t: fa.param_grid(t) for t in self.targets}
        # Validation is the selection data of the workers, as in the tuner.
        fa._init_worker(matrix, fields, self.rankings, self.train, self.valid, self.targets, self.param_lists)

    def fit(self, choice):
        """Score matrix [users, items] of one configuration (dict as in hybrid_feature_aug_best.yaml)."""
        hybrid = fa._state["hybrids"][choice["target"]]
        sources = tuple(self.names.index(s) for s in choice["sources"])
        return fa.fit_configuration(hybrid, self.rankings, sources, choice["n_pseudo"],
                                    choice["pseudo_weight"], choice["target_params"])

    def evaluate(self, scores, stage):
        seen, relevant = (self.train, self.valid) if stage == "valid" else (self.train | self.valid, self.test)
        return hr.evaluate_ranking(scores, seen, relevant)[0]

    def top10(self, scores, stage):
        seen = self.train if stage == "valid" else self.train | self.valid
        return hr.top_k_items(scores, seen, TOP_K)[0]

    def pseudo_items(self, source, n_pseudo):
        """Boolean [users, items]: the items `source` added as pseudo-interactions (its top-n unseen)."""
        top = self.rankings[self.names.index(source)][1:, :n_pseudo] - 1  # internal id -> score index
        m = np.zeros(self.scores.shape[1:], dtype=bool)
        rows, cols = np.nonzero(top >= 0)
        m[rows, top[rows, cols]] = True
        return m

    def best_on_valid(self, sources, targets=None):
        """Best configuration on validation over targets x hyperparameters (x n_pseudo x pseudo_weight
        when there are sources), as the tuner's search. Ties keep the first (simplest) one."""
        source_idx = tuple(self.names.index(s) for s in sources)
        grid = [(n, w) for n in fa.N_PSEUDO_GRID for w in fa.PSEUDO_WEIGHTS] if sources else [(0, 0.0)]
        best = None
        for target in targets or self.targets:
            for n, w in grid:
                metrics, _ = fa._evaluate_task((target, source_idx, n, w))
                for p, params in enumerate(self.param_lists[target]):
                    score = metrics[p, fa.NDCG_COL]
                    if best is None or score > best[0]:
                        best = (score, {"target": target, "target_params": params, "sources": list(sources),
                                        "n_pseudo": n, "pseudo_weight": w})
        return best


def _fa_label(choice):
    params = {k: choice["target_params"][k] for k in fa.TARGET_GRIDS[choice["target"]]}
    return fa.config_label(choice["target"], params, choice["sources"], choice["n_pseudo"], choice["pseudo_weight"])


def _plain_target(cfg):
    """The final target with its hyperparameters, without pseudo-interactions."""
    return {**{k: cfg[k] for k in ("target", "target_params")}, "sources": [], "n_pseudo": 0, "pseudo_weight": 0.0}


def feature_aug_check(cfg, data):
    """Refit the final configuration; it must reproduce the validation metrics stored by the tuner."""
    scores = data.fit(cfg)
    valid = data.evaluate(scores, "valid")
    if any(abs(valid[m] - cfg["valid_metrics"][m]) > 1e-9 for m in METRICS):
        print(f"WARNING: refitted validation {PRIMARY_METRIC} {valid[PRIMARY_METRIC]:.6f} differs from the stored "
              f"{cfg['valid_metrics'][PRIMARY_METRIC]:.6f}; the source outputs probably differ from the tuned ones.")
    return scores


def feature_aug_ablation(cfg, data, final_scores):
    full = data.evaluate(final_scores, "test")
    rows = [{"change": "none (full hybrid)", "configuration": _fa_label(cfg),
             f"valid_{PRIMARY_METRIC}": cfg["valid_metrics"][PRIMARY_METRIC],
             **{f"test_{k}": v for k, v in full.items()}, f"delta_test_{PRIMARY_METRIC}": 0.0}]

    def add(change, valid_score, choice):
        test = data.evaluate(data.fit(choice), "test")
        rows.append({"change": change, "configuration": _fa_label(choice), f"valid_{PRIMARY_METRIC}": valid_score,
                     **{f"test_{k}": v for k, v in test.items()},
                     f"delta_test_{PRIMARY_METRIC}": test[PRIMARY_METRIC] - full[PRIMARY_METRIC]})

    sources = cfg["sources"]
    for removed in sources:
        print(f"Feature Augmentation ablation: re-tuning without {removed}...", flush=True)
        add(f"without {removed}", *data.best_on_valid([s for s in sources if s != removed]))
    if len(sources) > 1:
        print("Feature Augmentation ablation: re-tuning without any source...", flush=True)
        add("without all sources (no augmentation)", *data.best_on_valid([]))
    # Same target and hyperparameters, pseudo-interactions removed (not re-tuned): the effect of augmentation alone.
    plain = _plain_target(cfg)
    add("without pseudo-interactions (same target, not re-tuned)", data.evaluate(data.fit(plain), "valid")[PRIMARY_METRIC], plain)
    for target in data.targets:
        if target != cfg["target"]:
            print(f"Feature Augmentation ablation: same sources on {target}...", flush=True)
            add(f"target replaced by {target}", *data.best_on_valid(sources, [target]))
    save(pd.DataFrame(rows), "feature_aug_ablation.csv",
         "Feature Augmentation: ablation (re-tuned on validation unless stated)")


def feature_aug_complementarity(cfg, data, final_scores):
    """Overlap of the hybrid's test top-10 with each source's and the plain target's top-10."""
    hybrid = data.top10(final_scores, "test")
    users = np.arange(hybrid.shape[0])[:, None]
    hybrid_hits = data.test[users, hybrid]
    active = data.test.any(axis=1)
    components = {s: data.scores[data.names.index(s)] for s in cfg["sources"]}
    components[f"{cfg['target']} (plain target)"] = data.fit(_plain_target(cfg))
    in_any = np.zeros_like(hybrid_hits)
    rows = []
    for name, scores in components.items():
        own = data.top10(scores, "test")
        shared = (hybrid[:, :, None] == own[:, None, :]).any(axis=2)
        in_any |= shared
        rows.append({
            "component": name,
            "shared_top10_items": shared[active].sum(axis=1).mean(),
            "share_of_hybrid_hits_in_its_top10": (hybrid_hits & shared).sum() / hybrid_hits.sum(),
            "its_own_hits": int(data.test[users, own][active].sum()),
        })
    rows.append({
        "component": "none of them",
        "shared_top10_items": (~in_any)[active].sum(axis=1).mean(),
        "share_of_hybrid_hits_in_its_top10": (hybrid_hits & ~in_any).sum() / hybrid_hits.sum(),
        "its_own_hits": int(hybrid_hits.sum()),
    })
    save(pd.DataFrame(rows), "feature_aug_complementarity.csv",
         "Feature Augmentation: overlap of the hybrid's test top-10 with each component's top-10 "
         "('none of them': items in no component's top-10; its_own_hits = the hybrid's hits)")


def feature_aug_pseudo_items(cfg, data, final_scores):
    """Which recommended items were pseudo-interactions, from which source, and how often they are hits.

    The plain target (same target, no pseudo-interactions) is the reference: some pseudo-items are
    recommended anyway, because the sources and the target agree on them.
    """
    sources, n = cfg["sources"], cfg["n_pseudo"]
    pseudo = {s: data.pseudo_items(s, n) for s in sources}
    plain_scores = data.fit(_plain_target(cfg))
    groups, changes = [], []
    for stage in ("valid", "test"):
        relevant = data.valid if stage == "valid" else data.test
        active = relevant.any(axis=1)
        users = np.arange(len(active))[active][:, None]
        lists = {"hybrid": data.top10(final_scores, stage)[active], "plain target": data.top10(plain_scores, stage)[active]}
        for name, top in lists.items():
            hit = relevant[users, top]
            flags = {s: pseudo[s][users, top] for s in sources}
            any_pseudo = np.logical_or.reduce(list(flags.values()))
            categories = {f"pseudo from {s} only": flags[s] & (np.add.reduce([flags[o] for o in sources]) == 1)
                          for s in sources}
            if len(sources) > 1:
                categories["pseudo from several sources"] = np.add.reduce(list(flags.values())) > 1
            categories["not a pseudo-item"] = ~any_pseudo
            for category, member in categories.items():
                groups.append({"stage": stage, "list": name, "items": category,
                               "share_of_slots": member.mean(), "hits": int(hit[member].sum()),
                               "share_of_hits": hit[member].sum() / hit.sum(),
                               "hit_rate": hit[member].mean() if member.any() else np.nan})
        # Items augmentation brings into, and pushes out of, the plain target's top-10.
        hybrid, plain = lists["hybrid"], lists["plain target"]
        entered = ~(hybrid[:, :, None] == plain[:, None, :]).any(axis=2)
        left = ~(plain[:, :, None] == hybrid[:, None, :]).any(axis=2)
        any_pseudo_hybrid = np.logical_or.reduce([pseudo[s][users, hybrid] for s in sources])
        changes.append({"stage": stage,
                        "changed_items_per_user": entered.sum(axis=1).mean(),
                        "entered_hit_rate": relevant[users, hybrid][entered].mean(),
                        "left_hit_rate": relevant[users, plain][left].mean(),
                        "entered_that_are_pseudo": any_pseudo_hybrid[entered].mean(),
                        "net_hits": int(relevant[users, hybrid][entered].sum() - relevant[users, plain][left].sum())})
    union = np.logical_or.reduce(list(pseudo.values()))
    print(f"\nPseudo-interactions per user: {union.sum(axis=1).mean():.1f} "
          f"({' + '.join(f'{s} {pseudo[s].sum(axis=1).mean():.1f}' for s in sources)}; "
          f"{np.logical_and.reduce(list(pseudo.values())).sum(axis=1).mean():.1f} added by all of them)")
    save(pd.DataFrame(groups), "feature_aug_pseudo_items.csv",
         "Feature Augmentation: recommended items that were pseudo-interactions (hybrid vs plain target)")
    save(pd.DataFrame(changes), "feature_aug_changes_vs_plain_target.csv",
         "Feature Augmentation: items entering / leaving the plain target's top-10")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hybrids", nargs="+", choices=HYBRIDS, default=list(HYBRIDS))
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    train, valid, test = load_split()
    split = {"train": train, "valid": valid, "test": test}

    if "mixed" in args.hybrids:
        mixed = load_config("mixed_hybrid_best.yaml")
        mixed_attribution(mixed, split)
        mixed_ablation(mixed, split)

    if "regression" in args.hybrids:
        regression = load_config("hybrid_regression_best.yaml")
        data = RegressionData(split)
        standardised = regression_coefficients(regression, data)
        regression_ablation(regression, data, standardised)
        regression_stability(regression, data, standardised)
        regression_complementarity(regression, data)

    if "feature_aug" in args.hybrids:
        feature_aug = load_config("hybrid_feature_aug_best.yaml")
        data = FeatureAugData(split)
        final_scores = feature_aug_check(feature_aug, data)
        feature_aug_pseudo_items(feature_aug, data, final_scores)
        feature_aug_complementarity(feature_aug, data, final_scores)
        feature_aug_ablation(feature_aug, data, final_scores)
    print(f"\nWrote the tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
