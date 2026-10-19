import json
import os

import pandas as pd
import yaml

from recbole.model.general_recommender.mixed_hybrid import save_recommendations
from recbole.model.general_recommender.rerankers import build_rerankers, output_name, rerank
from recbole.utils.evaluation import Evaluator, load_item_genres
from recbole.utils.experiment import CONFIG_DIR, OUTPUT_DIR, load_recs, load_split

SHOWN = ["NDCG", "Recall", "ILD", "Miscalibration", "Novelty", "APLT", "TailExposure", "TailHitShare", "UserNDCGGap", "Coverage", "Gini"]
FAIRNESS_PARAMS = ("inactive_weight", "exposure_target")  # stored next to lambda in the config by tune_rerankers.py


def share_changed(recs, k):
    """Share of the reranked top-k that was not in the base model's top-k."""
    return float((recs["base_rank"] > k).mean())


def main():
    config_path = os.path.join(CONFIG_DIR, "rerankers_best.yaml")
    if not os.path.exists(config_path):
        raise SystemExit(f"Missing {config_path}. Run tune_rerankers.py first.")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base, k = cfg["base_model"], cfg["k"]

    train, _, _ = load_split()
    evaluator = Evaluator.from_outputs("test")
    rerankers = build_rerankers(train, load_item_genres(), evaluator.short_head, evaluator.active_users)
    base_recs = load_recs(base, "test")

    rows = [{"model": base, "lambda": 0.0, f"Changed@{k}": 0.0, **evaluator.evaluate(base_recs, k)}]
    for name, chosen in cfg["rerankers"].items():
        for attr in FAIRNESS_PARAMS:
            if attr in chosen:
                setattr(rerankers[name], attr, chosen[attr])
        recs = rerank(base_recs, rerankers[name], chosen["lambda"], k)
        metrics = {m: float(v) for m, v in evaluator.evaluate(recs, k).items()}
        out_dir = os.path.join(OUTPUT_DIR, output_name(name))
        save_recommendations(recs, os.path.join(out_dir, "recs_test.tsv"))
        with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"base_model": base, "k": k, **chosen, "test_metrics": metrics,
                       f"changed_share@{k}": share_changed(recs, k)}, f, indent=2)
        rows.append({"model": f"{base} + {name}", "lambda": chosen["lambda"], f"Changed@{k}": share_changed(recs, k), **metrics})
        params = ", ".join(f"{a} = {chosen[a]}" for a in ("lambda",) + FAIRNESS_PARAMS if a in chosen)
        print(f"reranked {base} with {name} ({params})", flush=True)

    table = pd.DataFrame(rows).set_index("model")
    out_dir = os.path.join(OUTPUT_DIR, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "rerank_test_comparison.csv"))

    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(f"\n=== Test, k = {k}: base model vs reranked (lambda selected on validation) ===")
        print(table[["lambda", f"Changed@{k}"] + [f"{m}@{k}" for m in SHOWN]].T)  # metrics as rows: fits the terminal
    print(f"\nWrote outputs/Rerank_<reranker>/recs_test.tsv and test_metrics.json")
    print(f"Wrote {out_dir}/rerank_test_comparison.csv")


if __name__ == "__main__":
    main()
