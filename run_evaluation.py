"""Evaluate exported top-N lists with our own metrics (Task 2.1) and write the tables to outputs/evaluation/.

Evaluates every model that has outputs/<model>/recs_<stage>.tsv (written by run_model.py), or
only the models given. Metric definitions are in recbole/utils/evaluation.py.
"""

import argparse
import os

import pandas as pd

from recbole.utils.evaluation import Evaluator
from recbole.utils.experiment import OUTPUT_DIR, load_recs, load_scores

ACCURACY = ["NDCG", "Recall", "Precision", "F1", "MAP", "MRR", "Hit"]
BEYOND = ["Coverage", "Gini", "Novelty", "ILD", "ARP", "APLT", "PopLift", "Miscalibration", "UserNDCGGap", "TailHitShare"]


def available_models(stage):
    return sorted(
        m for m in os.listdir(OUTPUT_DIR)
        if os.path.exists(os.path.join(OUTPUT_DIR, m, f"recs_{stage}.tsv"))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="models to evaluate (default: all with exported lists)")
    parser.add_argument("--stage", choices=["valid", "test"], default="test")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    models = args.models or available_models(args.stage)
    evaluator = Evaluator.from_outputs(args.stage)

    rows, groups = [], []
    for model in models:
        recs = load_recs(model, args.stage)
        # AUC needs the full score matrix; models/hybrids that only export lists get no AUC.
        has_scores = os.path.exists(os.path.join(OUTPUT_DIR, model, "scores.npz"))
        scores = load_scores(model) if has_scores else None
        rows.append({"model": model, **evaluator.evaluate(recs, args.k, scores)})
        groups.append(evaluator.group_breakdown(recs, args.k).reset_index().assign(model=model))
        print(f"evaluated {model}", flush=True)

    table = pd.DataFrame(rows).set_index("model").sort_values(f"NDCG@{args.k}", ascending=False)
    out_dir = os.path.join(OUTPUT_DIR, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, f"{args.stage}_metrics.csv"))
    group_table = pd.concat(groups).set_index(["model", "user_group"])
    group_table.to_csv(os.path.join(out_dir, f"{args.stage}_user_groups.csv"))

    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(f"\n=== Accuracy ({args.stage}, k={args.k}) ===")
        print(table[[f"{m}@{args.k}" for m in ACCURACY] + (["AUC"] if "AUC" in table else [])])
        print(f"\n=== Beyond accuracy ({args.stage}, k={args.k}) ===")
        print(table[[f"{m}@{args.k}" for m in BEYOND]])
    print(f"\nWrote {out_dir}/{args.stage}_metrics.csv and {args.stage}_user_groups.csv")


if __name__ == "__main__":
    main()
