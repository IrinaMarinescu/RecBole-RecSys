"""Task 3.1: select each reranker's lambda on the validation lists of one or several base models.

For every base model and reranker, every lambda (Fairness: lambda x inactive_weight) is applied to the
base model's validation list and evaluated. The selected setting is the best target metric among those
that keep NDCG@k within `--max-loss` of the base list (Fairness: and do not widen the active/inactive gap).
The test split is never opened here.

    python tune_rerankers.py                      # every base with a tuned config, plus the hybrids
    python tune_rerankers.py --bases RecVAE BPR   # some of them

Writes configs/rerankers/<base>.yaml and outputs/Rerank_<reranker>/<base>/{tuning_valid.csv, recs_valid.tsv}.
"""

import argparse
import glob
import os
import time

import pandas as pd
import yaml

from recbole.model.general_recommender.mixed_hybrid import save_recommendations
from recbole.model.general_recommender.rerankers import (
    EXPOSURE_TARGETS,
    HYBRID_BASES,
    MAX_ACCURACY_LOSS,
    RERANKER_NAMES,
    base_relevance_frame,
    build_rerankers,
    output_name,
    rerank,
    select_lambda,
)
from recbole.utils.evaluation import Evaluator, load_item_genres
from recbole.utils.experiment import CONFIG_DIR, OUTPUT_DIR, load_recs, load_split

RERANK_CONFIG_DIR = os.path.join(CONFIG_DIR, "rerankers")
DEFAULT_LAMBDAS = [round(0.1 * i, 1) for i in range(11)]
DEFAULT_INACTIVE_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
SHOWN = ["NDCG", "Recall", "ILD", "Miscalibration", "Novelty", "APLT", "TailExposure", "UserNDCGGap", "Coverage", "Gini"]
FAIRNESS_PARAMS = ("inactive_weight", "exposure_target")  # Fairness attributes stored next to lambda in the config


def config_path(base):
    return os.path.join(RERANK_CONFIG_DIR, f"{base}.yaml")


def load_rerank_config(base):
    """The selection written for `base` by this script (configs/rerankers/<base>.yaml)."""
    with open(config_path(base), encoding="utf-8") as f:
        return yaml.safe_load(f)


def configured_bases():
    """Base models with a configs/rerankers/<base>.yaml."""
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(config_path("*")))


def default_bases(stage="valid"):
    """Every model with a tuned config, plus the hybrids, if its lists for `stage` are exported."""
    tuned = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(CONFIG_DIR, "tuned", "*.yaml")))
    candidates = tuned + [b for b in HYBRID_BASES if b not in tuned]
    bases = [b for b in candidates if os.path.exists(os.path.join(OUTPUT_DIR, b, f"recs_{stage}.tsv"))]
    missing = [b for b in candidates if b not in bases]
    if missing:
        print(f"Skipping bases without outputs/<base>/recs_{stage}.tsv: {', '.join(missing)}")
    return bases


def sweep(base_recs, reranker, settings, evaluator, k):
    """One row per setting ({'lambda': ..., <attribute>: ...}): the reranked list's metrics."""
    rows = []
    for setting in settings:
        for attr, value in setting.items():
            if attr != "lambda":
                setattr(reranker, attr, value)
        recs = rerank(base_recs, reranker, setting["lambda"], k)
        rows.append({**setting, **{m: float(v) for m, v in evaluator.evaluate(recs, k).items()}})
    return pd.DataFrame(rows)


def no_worse_for(name, k):
    """Extra selection constraints: Fairness may not widen the active/inactive accuracy gap."""
    return [(f"UserNDCGGap@{k}", False)] if name == "Fairness" else []


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


def relative_change(value, reference, higher_is_better=True):
    """Change in % vs the reference, signed so that positive = better."""
    if not reference:
        return float("nan")
    return (1.0 if higher_is_better else -1.0) * 100.0 * (value - reference) / abs(reference)


def tune_base(base, args, lambdas, evaluator, rerankers):
    """Sweep and select every reranker on `base`'s validation list; returns the config to write."""
    k, primary = args.k, f"NDCG@{args.k}"
    base_recs = base_relevance_frame(load_recs(base, "valid"))
    base_metrics = {m: float(v) for m, v in evaluator.evaluate(base_recs, k).items()}
    shown = [f"{m}@{k}" for m in SHOWN]
    print(f"##### Base model {base} (valid): " + "  ".join(f"{m}={base_metrics[m]:.4f}" for m in shown) + "\n")

    config = {
        "base_model": base, "k": k, "primary_metric": primary, "max_accuracy_loss": args.max_loss,
        "lambdas": lambdas, "inactive_weights": args.inactive_weights, "base_valid_metrics": base_metrics,
        "rerankers": {},
    }
    for name in args.rerankers:
        reranker, start = rerankers[name], time.time()
        table = sweep(base_recs, reranker, settings_for(name, lambdas, args.inactive_weights), evaluator, k)
        target = f"{reranker.target}@{k}"
        table.attrs["target"] = target
        best = select_lambda(table, target, reranker.higher_is_better, primary, args.max_loss, no_worse_for(name, k))
        chosen = {"lambda": float(best["lambda"])}
        if name == "Fairness":
            chosen["inactive_weight"] = float(best["inactive_weight"])
            chosen["exposure_target"] = args.fairness_target
            reranker.inactive_weight = chosen["inactive_weight"]

        out_dir = os.path.join(OUTPUT_DIR, output_name(name, base))
        os.makedirs(out_dir, exist_ok=True)
        table.to_csv(os.path.join(out_dir, "tuning_valid.csv"), index=False)
        save_recommendations(rerank(base_recs, reranker, chosen["lambda"], k), os.path.join(out_dir, "recs_valid.tsv"))
        metric_columns = [c for c in table.columns if c not in ("lambda", "inactive_weight")]
        config["rerankers"][name] = {**chosen, "target": target,
                                     "valid_metrics": {m: float(best[m]) for m in metric_columns}}

        arrow = "max" if reranker.higher_is_better else "min"
        print(f"=== {base} + {name}: target {target} ({arrow}), {time.time() - start:.0f}s ===")
        show(table, best, k, shown)
        print()
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bases", nargs="*", help="experiments whose lists are reranked (default: all with "
                                                   "a tuned config, plus the hybrids)")
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
    bases = args.bases or default_bases("valid")
    if not bases:
        raise SystemExit("No base model has outputs/<base>/recs_valid.tsv. Run run_model.py first.")

    stale = [n for n in RERANKER_NAMES if os.path.exists(os.path.join(OUTPUT_DIR, output_name(n), "recs_valid.tsv"))]
    if stale:
        print(f"Warning: outputs/Rerank_<reranker>/recs_*.tsv from an older single-base run exist for {', '.join(stale)}; "
              "delete them, or run_evaluation.py will list them as models.\n")

    train, _, _ = load_split()
    evaluator = Evaluator.from_outputs("valid")
    rerankers = build_rerankers(train, load_item_genres(), evaluator.short_head, evaluator.active_users)
    rerankers["Fairness"].exposure_target = args.fairness_target
    print(f"Selection: best target metric with NDCG@{args.k} >= {1 - args.max_loss:.2f} x base"
          f" (Fairness: and UserNDCGGap@{args.k} not above the base)\n")

    os.makedirs(RERANK_CONFIG_DIR, exist_ok=True)
    summary = []
    for base in bases:
        config = tune_base(base, args, lambdas, evaluator, rerankers)
        with open(config_path(base), "w", encoding="utf-8") as f:
            f.write(f"# Selected on validation, base {base} (written by tune_rerankers.py)\n")
            yaml.safe_dump(config, f, sort_keys=False)
        for name, chosen in config["rerankers"].items():
            ref, got = config["base_valid_metrics"], chosen["valid_metrics"]
            summary.append({
                "base": base, "reranker": name, "lambda": chosen["lambda"],
                "inactive_weight": chosen.get("inactive_weight", float("nan")),
                f"NDCG@{args.k} loss %": -relative_change(got[config["primary_metric"]], ref[config["primary_metric"]]),
                "target": chosen["target"],
                "target gain %": relative_change(got[chosen["target"]], ref[chosen["target"]], rerankers[name].higher_is_better),
            })

    with pd.option_context("display.width", 220, "display.float_format", "{:.2f}".format):
        print("=== Selected settings (validation) ===")
        print(pd.DataFrame(summary).to_string(index=False))
    print("\nWrote outputs/Rerank_<reranker>/<base>/tuning_valid.csv and recs_valid.tsv")
    print(f"Wrote {RERANK_CONFIG_DIR}/<base>.yaml")


if __name__ == "__main__":
    main()
