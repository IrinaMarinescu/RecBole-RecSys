"""Evaluate the frozen Hybrid Regression configuration once on the TEST set.

Loads configs/hybrid_regression_best.yaml (written by tune_hybrid_regression.py) and
compares it with every candidate model, and with the configurations the other selection
methods picked on validation. Does not refit.

Example:
    python run_hybrid_regression.py
"""

import json
import os

import numpy as np
import pandas as pd
import yaml

from recbole.model.general_recommender.hybrid_regression import (
    METRIC_NAMES,
    PRIMARY_METRIC,
    TOP_K,
    combine_scores,
    evaluate_ranking,
    load_score_stack,
)
from recbole.model.general_recommender.mixed_hybrid import evaluate_recommendations
from recbole.utils.experiment import CONFIG_DIR, EXPORT_TOP_N, OUTPUT_DIR, load_split
from tune_hybrid_regression import KINDS, recommendation_frame


def _mask(frame, user_index, item_index, shape):
    mask = np.zeros(shape, dtype=bool)
    users = [user_index[user] for user in frame["user_id"].astype(str)]
    items = [item_index[item] for item in frame["item_id"].astype(str)]
    mask[users, items] = True
    return mask


def _weight_vector(names, chosen_models, ranking_weights):
    weights = np.zeros(len(names), dtype=np.float32)
    for name in chosen_models:
        weights[names.index(name)] = ranking_weights[name]
    return weights


def _row(label, role, metrics):
    return {"model": label, "role": role, **metrics}


def _hybrid_label(models, hyperparams):
    if not hyperparams:
        return models[0]
    return (
        f"Hybrid Regression: {'+'.join(models)} | "
        f"C={hyperparams['C']:g} penalty={hyperparams['penalty']} neg={hyperparams['num_negatives']}"
    )


def main():
    config_path = os.path.join(CONFIG_DIR, "hybrid_regression_best.yaml")
    if not os.path.exists(config_path):
        raise SystemExit(f"Missing {config_path}. Run tune_hybrid_regression.py first.")
    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    names = list(KINDS)
    scores, user_ids, item_ids = load_score_stack(names)
    user_index = {user: i for i, user in enumerate(user_ids)}
    item_index = {item: i for i, item in enumerate(item_ids)}
    train, valid, test = load_split()
    train_seen = _mask(train, user_index, item_index, scores.shape[1:])
    valid_seen = _mask(valid, user_index, item_index, scores.shape[1:])
    seen = train_seen | valid_seen
    relevant = _mask(test, user_index, item_index, scores.shape[1:])
    user_mask = relevant.any(axis=1)

    rows = []
    for model in names:
        rows.append(_row(model, "standalone", evaluate_ranking(scores[names.index(model)], seen, relevant)[0]))

    final_weights = _weight_vector(names, cfg["models"], cfg["ranking_weights"])
    combined = combine_scores(scores, final_weights)
    hybrid_metrics, _ = evaluate_ranking(combined, seen, relevant)
    final_label = _hybrid_label(cfg["models"], cfg["hyperparams"])
    rows.append(_row(final_label, f"FINAL ({cfg['selection_method']})", hybrid_metrics))

    for method, alt in cfg.get("alternatives", {}).items():
        if method == cfg["selection_method"]:
            continue
        label = _hybrid_label(alt["models"], alt["hyperparams"])
        weights = _weight_vector(names, alt["models"], alt["ranking_weights"])
        metrics, _ = evaluate_ranking(combine_scores(scores, weights), seen, relevant)
        rows.append(_row(label, f"not selected ({method})", metrics))

    mixed = recommendation_frame(combined, seen, user_ids, item_ids, user_mask, top_k=EXPORT_TOP_N)  # candidates for the rerankers
    reference = evaluate_recommendations(mixed, test, k=TOP_K)
    for name in METRIC_NAMES:
        if abs(reference[name] - hybrid_metrics[name]) > 1e-8:
            raise AssertionError((name, reference[name], hybrid_metrics[name]))

    table = pd.DataFrame(rows)
    print("--- Test metrics ---")
    print(table.to_string(index=False, float_format="%.4f"))
    print(f"\nFinal configuration selected on validation by {PRIMARY_METRIC} ({cfg['selection_method']}).")

    out_dir = os.path.join(OUTPUT_DIR, "HybridRegression")
    os.makedirs(out_dir, exist_ok=True)
    recs_path = os.path.join(out_dir, "recs_test.tsv")
    mixed.to_csv(recs_path, sep="\t", index=False)
    # Also the frozen configuration's validation list (training items hidden), for downstream tuning
    # such as the rerankers.
    valid_frame = recommendation_frame(combined, train_seen, user_ids, item_ids, valid_seen.any(axis=1), top_k=EXPORT_TOP_N)
    valid_frame.to_csv(os.path.join(out_dir, "recs_valid.tsv"), sep="\t", index=False)
    table.to_csv(os.path.join(out_dir, "test_comparison.csv"), index=False)
    metrics_path = os.path.join(out_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump({**cfg, "hybrid_regression_test": hybrid_metrics, "comparison_test": rows}, handle, indent=2)
    print(f"Wrote {recs_path} and recs_valid.tsv")
    print(f"Wrote {metrics_path} and test_comparison.csv")


if __name__ == "__main__":
    main()
