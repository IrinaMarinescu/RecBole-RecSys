"""Evaluate the frozen Feature Augmentation configuration once on the TEST set.

Loads configs/hybrid_feature_aug_best.yaml (written by tune_hybrid_feature_aug.py) and fits
that configuration exactly as on validation: pseudo-interactions from the sources' exported
scores (outputs/<model>/scores.npz), the target refitted on train + pseudo-interactions. No
hyperparameter is changed. Test lists hide training and validation items, as for every model
in run_model.py.

Before the test, the fit must reproduce the validation metrics stored in the YAML; otherwise
the sources were re-exported after tuning and the script stops.

Compares the hybrid with every source model, and with the configurations the other selection
methods picked on validation (for the report only; nothing is selected on test). Exports the
frozen configuration's test and validation lists (EXPORT_TOP_N items) as candidates for the rerankers.

Example:
    python run_hybrid_feature_aug.py
"""

import json
import os

import numpy as np
import pandas as pd
import yaml

from recbole.model.general_recommender.hybrid_regression import (
    METRIC_NAMES,
    TOP_K,
    evaluate_ranking,
    load_score_stack,
)
from recbole.model.general_recommender.mixed_hybrid import evaluate_recommendations
from recbole.utils.experiment import CONFIG_DIR, EXPORT_TOP_N, OUTPUT_DIR, load_split
from tune_hybrid_feature_aug import (
    KINDS,
    SEED,
    TARGET_GRIDS,
    FixedScores,
    config_label,
    full_scores,
    load_recbole_train,
    target_hybrid,
)
from tune_hybrid_regression import recommendation_frame


def _mask(frame, user_index, item_index, shape):
    mask = np.zeros(shape, dtype=bool)
    mask[frame["user_id"].map(user_index), frame["item_id"].map(item_index)] = True
    return mask


def _label(choice):
    params = {key: choice["target_params"][key] for key in TARGET_GRIDS[choice["target"]]}
    return "Feature Augmentation: " + config_label(
        choice["target"], params, choice["sources"], choice["n_pseudo"], choice["pseudo_weight"]
    )


def _key(choice):
    return json.dumps({k: v for k, v in choice.items() if k != "valid_metrics"}, sort_keys=True)


def main():
    config_path = os.path.join(CONFIG_DIR, "hybrid_feature_aug_best.yaml")
    if not os.path.exists(config_path):
        raise SystemExit(f"Missing {config_path}. Run tune_hybrid_feature_aug.py first.")
    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    # Every source, plus targets that are not sources (UserKNN), for the standalone rows.
    names = list(KINDS) + [t for t in TARGET_GRIDS if t not in KINDS]
    scores, user_ids, item_ids = load_score_stack(names)
    user_index = {user: i for i, user in enumerate(user_ids)}
    item_index = {item: i for i, item in enumerate(item_ids)}
    train, valid, test = load_split()
    train_seen = _mask(train, user_index, item_index, scores.shape[1:])
    valid_relevant = _mask(valid, user_index, item_index, scores.shape[1:])
    seen = train_seen | valid_relevant  # hidden from the test lists
    relevant = _mask(test, user_index, item_index, scores.shape[1:])

    recbole = {}  # target -> (config, train_data)
    fitted = {}   # configuration -> score matrix

    def fit(choice):
        """Score matrix of one configuration: the target refitted on train + pseudo-interactions."""
        key = _key(choice)
        if key not in fitted:
            target = choice["target"]
            if target not in recbole:
                recbole[target] = load_recbole_train(target)
            config, train_data = recbole[target]
            sources = [FixedScores(scores[names.index(n)], config["USER_ID_FIELD"]) for n in choice["sources"]]
            hybrid = target_hybrid(config, train_data, sources, target, choice["target_params"])
            np.random.seed(SEED)  # as in the tuner: SLIMElastic draws from the global RNG
            hybrid.fit_augmentation(n_pseudo=choice["n_pseudo"], pseudo_weight=choice["pseudo_weight"])
            fitted[key] = full_scores(hybrid)
        return fitted[key]

    final = {key: cfg[key] for key in ("target", "target_params", "sources", "n_pseudo", "pseudo_weight", "valid_metrics")}
    combined = fit(final)

    # The frozen configuration must reproduce its validation score before it is used on test.
    valid_metrics, _ = evaluate_ranking(combined, train_seen, valid_relevant)
    for name in METRIC_NAMES:
        if abs(valid_metrics[name] - final["valid_metrics"][name]) > 1e-9:
            raise RuntimeError(
                f"Validation {name} is {valid_metrics[name]:.6f}, but tuning recorded "
                f"{final['valid_metrics'][name]:.6f}. Were the sources re-exported after tuning? "
                "Re-run tune_hybrid_feature_aug.py."
            )

    rows = []
    for model in names:
        metrics, _ = evaluate_ranking(scores[names.index(model)], seen, relevant)
        rows.append({"model": model, "role": "standalone", **metrics})

    hybrid_metrics, _ = evaluate_ranking(combined, seen, relevant)
    rows.append({"model": _label(final), "role": f"FINAL ({cfg['selection_method']})", **hybrid_metrics})

    for method, alt in cfg.get("alternatives", {}).items():
        if method == cfg["selection_method"]:
            continue
        print(f"Fitting the {method} configuration...", flush=True)
        metrics, _ = evaluate_ranking(fit(alt), seen, relevant)
        rows.append({"model": _label(alt), "role": f"not selected ({method})", **metrics})

    recs = recommendation_frame(combined, seen, user_ids, item_ids, relevant.any(axis=1), top_k=EXPORT_TOP_N)  # candidates for the rerankers
    reference = evaluate_recommendations(recs, test, k=TOP_K)
    for name in METRIC_NAMES:
        if abs(reference[name] - hybrid_metrics[name]) > 1e-8:
            raise AssertionError((name, reference[name], hybrid_metrics[name]))

    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_colwidth", 120):
        print("--- Test metrics ---")
        print(table.to_string(index=False, float_format="%.4f"))
    print(f"\nFinal configuration selected on validation by {cfg['primary_metric']} ({cfg['selection_method']}).")

    out_dir = os.path.join(OUTPUT_DIR, "HybridFeatureAug")
    os.makedirs(out_dir, exist_ok=True)
    recs_path = os.path.join(out_dir, "recs_test.tsv")
    recs.to_csv(recs_path, sep="\t", index=False)
    # Also the frozen configuration's validation list (training items hidden), for downstream tuning
    # such as the rerankers.
    valid_frame = recommendation_frame(combined, train_seen, user_ids, item_ids, valid_relevant.any(axis=1), top_k=EXPORT_TOP_N)
    valid_frame.to_csv(os.path.join(out_dir, "recs_valid.tsv"), sep="\t", index=False)
    table.to_csv(os.path.join(out_dir, "test_comparison.csv"), index=False)
    metrics_path = os.path.join(out_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump({**cfg, "feature_aug_test": hybrid_metrics, "comparison_test": rows}, handle, indent=2)
    print(f"Wrote {recs_path} and recs_valid.tsv")
    print(f"Wrote {metrics_path} and test_comparison.csv")


if __name__ == "__main__":
    main()
