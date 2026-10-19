import argparse
import os
import time

import pandas as pd
import yaml

from recbole.model.general_recommender.mixed_hybrid import save_recommendations
from recbole.model.general_recommender.rerankers import (
    EXPOSURE_TARGETS,
    MAX_ACCURACY_LOSS,
    RERANKER_NAMES,
    build_rerankers,
    output_name,
    rerank,
    select_lambda,
)
from recbole.utils.evaluation import Evaluator, load_item_genres
from recbole.utils.experiment import CONFIG_DIR, OUTPUT_DIR, load_recs, load_split

DEFAULT_BASE = "RecVAE"
DEFAULT_LAMBDAS = [round(0.1 * i, 1) for i in range(11)]
DEFAULT_INACTIVE_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
SHOWN = ["NDCG", "Recall", "ILD", "Miscalibration", "Novelty", "APLT", "TailExposure", "UserNDCGGap", "Coverage", "Gini"]
FAIRNESS_PARAMS = ("inactive_weight", "exposure_target")  # Fairness attributes stored next to lambda in the config


def sweep(base_recs, reranker, settings, evaluator, k):
    """One row per setting ({'lambda': ..., <attribute>: ...}): the reranked validation list's metrics."""
    rows = []
    for setting in settings:
        for attr, value in setting.items():
            if attr != "lambda":
                setattr(reranker, attr, value)
        recs = rerank(base_recs, reranker, setting["lambda"], k)
        rows.append({**setting, **{m: float(v) for m, v in evaluator.evaluate(recs, k).items()}})
    return pd.DataFrame(rows)


def settings_for(name, lambdas, inactive_weights):
    """Fairness searches lambda x inactive_weight (lambda = 0 once: every weight gives the base list)."""
    if name != "Fairness":
        return [{"lambda": lam} for lam in lambdas]
    return [{"lambda": 0.0, "inactive_weight": 1.0}] + [
        {"lambda": lam, "inactive_weight": w} for lam in lambdas if lam > 0 for w in inactive_weights
    ]


def show(table, best, k, shown):
    """Print the sweep; Fairness as lambda x inactive_weight pivots, the others as one row per lambda."""
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        if "inactive_weight" in table.columns:
            for metric in (f"{table.attrs['target']}", f"NDCG@{k}", f"UserNDCGGap@{k}"):
                pivot = table.pivot_table(index="lambda", columns="inactive_weight", values=metric)
                print(f"{metric} (rows lambda, columns inactive_weight; lambda = 0 is the base list):")
                print(pivot.to_string(na_rep=""))
            print(f"selected: lambda = {best['lambda']}, inactive_weight = {best['inactive_weight']}")
        else:
            view = table.set_index("lambda")[shown]
            view.index = [f"{lam:.1f}{' *' if lam == best['lambda'] else ''}" for lam in view.index]
            print(view.to_string())
            print(f"selected lambda = {best['lambda']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=DEFAULT_BASE, help="experiment whose lists are reranked")
    parser.add_argument("--rerankers", nargs="+", default=list(RERANKER_NAMES), choices=RERANKER_NAMES)
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--inactive-weights", nargs="+", type=float, default=DEFAULT_INACTIVE_WEIGHTS,
                        help="Fairness: multipliers of lambda for inactive users")
    parser.add_argument("--fairness-target", choices=EXPOSURE_TARGETS, default="equity",
                        help="Fairness: exposure quota of the long tail, proportional to its relevance (equity) or 50%% (parity)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-loss", type=float, default=MAX_ACCURACY_LOSS,
                        help="largest accepted relative drop in NDCG@k vs the base list")
    args = parser.parse_args()
    lambdas = sorted(set(args.lambdas) | {0.0})  # lambda = 0 is the accuracy reference
    k, primary = args.k, f"NDCG@{args.k}"

    train, _, _ = load_split()
    evaluator = Evaluator.from_outputs("valid")
    rerankers = build_rerankers(train, load_item_genres(), evaluator.short_head, evaluator.active_users)
    rerankers["Fairness"].exposure_target = args.fairness_target
    base_recs = load_recs(args.base, "valid")
    base_metrics = {m: float(v) for m, v in evaluator.evaluate(base_recs, k).items()}
    shown = [f"{m}@{k}" for m in SHOWN]
    print(f"Base model {args.base} (valid): " + "  ".join(f"{m}={base_metrics[m]:.4f}" for m in shown))
    print(f"Selection: best target metric with {primary} >= {1 - args.max_loss:.2f} x base"
          f" (Fairness: and UserNDCGGap@{k} not above the base)\n")

    config = {
        "base_model": args.base, "k": k, "primary_metric": primary, "max_accuracy_loss": args.max_loss,
        "lambdas": lambdas, "inactive_weights": args.inactive_weights, "base_valid_metrics": base_metrics,
        "rerankers": {},
    }
    for name in args.rerankers:
        reranker, start = rerankers[name], time.time()
        table = sweep(base_recs, reranker, settings_for(name, lambdas, args.inactive_weights), evaluator, k)
        target = f"{reranker.target}@{k}"
        table.attrs["target"] = target
        no_worse = [(f"UserNDCGGap@{k}", False)] if name == "Fairness" else []
        best = select_lambda(table, target, reranker.higher_is_better, primary, args.max_loss, no_worse)
        chosen = {"lambda": float(best["lambda"])}
        if name == "Fairness":
            chosen["inactive_weight"] = float(best["inactive_weight"])
            chosen["exposure_target"] = args.fairness_target
            reranker.inactive_weight = chosen["inactive_weight"]

        out_dir = os.path.join(OUTPUT_DIR, output_name(name))
        os.makedirs(out_dir, exist_ok=True)
        table.to_csv(os.path.join(out_dir, "tuning_valid.csv"), index=False)
        save_recommendations(rerank(base_recs, reranker, chosen["lambda"], k), os.path.join(out_dir, "recs_valid.tsv"))
        metric_columns = [c for c in table.columns if c not in ("lambda", "inactive_weight")]
        config["rerankers"][name] = {**chosen, "target": target,
                                     "valid_metrics": {m: float(best[m]) for m in metric_columns}}

        arrow = "max" if reranker.higher_is_better else "min"
        print(f"=== {name}: target {target} ({arrow}), {time.time() - start:.0f}s ===")
        show(table, best, k, shown)
        print()

    best_path = os.path.join(CONFIG_DIR, "rerankers_best.yaml")
    with open(best_path, "w", encoding="utf-8") as f:
        f.write(f"# Selected on validation, base {args.base} (written by tune_rerankers.py)\n")
        yaml.safe_dump(config, f, sort_keys=False)
    print("Wrote outputs/Rerank_<reranker>/tuning_valid.csv and recs_valid.tsv")
    print(f"Wrote {best_path}")


if __name__ == "__main__":
    main()
