"""Task 3.1 / 3.2: apply every reranker, at the setting selected on validation, to the base models' test lists.

For each base model with a configs/rerankers/<base>.yaml (written by tune_rerankers.py), the reranked test
lists and their metrics are written next to the tuning results. The same lambda grid is also swept on the
test lists (Fairness at its selected inactive_weight): this gives the test side of the trade-off curves of
run_rerank_analysis.py and is never used to select anything.

    python run_rerankers.py                     # every configured base
    python run_rerankers.py --bases RecVAE      # some of them
    python run_rerankers.py --no-test-sweep     # only the selected settings

Writes outputs/Rerank_<reranker>/<base>/{recs_test.tsv, test_metrics.json, sweep_test.csv} and
outputs/evaluation/rerank_test_comparison.csv (one row per base model and reranker, `reranker = none` for the base).
"""

import argparse
import json
import os

import pandas as pd

from recbole.model.general_recommender.mixed_hybrid import save_recommendations
from recbole.model.general_recommender.rerankers import base_relevance_frame, build_rerankers, output_name, rerank
from recbole.utils.evaluation import Evaluator, load_item_genres
from recbole.utils.experiment import OUTPUT_DIR, load_recs, load_split
from tune_rerankers import FAIRNESS_PARAMS, SHOWN, configured_bases, load_rerank_config, sweep

SHOWN = SHOWN + ["TailHitShare"]


def share_changed(recs, k):
    """Share of the reranked top-k that was not in the base model's top-k."""
    return float((recs["base_rank"] > k).mean())


def test_settings(name, lambdas, chosen):
    """The lambda grid for the test sweep; Fairness keeps the inactive_weight selected on validation."""
    if name != "Fairness":
        return [{"lambda": lam} for lam in lambdas]
    return [{"lambda": lam, "inactive_weight": chosen["inactive_weight"]} for lam in lambdas]


def run_base(base, cfg, evaluator, rerankers, test_sweep=True):
    """Rerank `base`'s test list with every configured reranker; one metrics row per list (base first)."""
    k = cfg["k"]
    base_recs = base_relevance_frame(load_recs(base, "test"))
    rows = [{"base": base, "reranker": "none", "lambda": 0.0, "inactive_weight": float("nan"), "exposure_target": None,
             f"Changed@{k}": 0.0, **{m: float(v) for m, v in evaluator.evaluate(base_recs, k).items()}}]
    for name, chosen in cfg["rerankers"].items():
        reranker = rerankers[name]
        for attr in FAIRNESS_PARAMS:
            if attr in chosen:
                setattr(reranker, attr, chosen[attr])
        recs = rerank(base_recs, reranker, chosen["lambda"], k)
        metrics = {m: float(v) for m, v in evaluator.evaluate(recs, k).items()}
        out_dir = os.path.join(OUTPUT_DIR, output_name(name, base))
        save_recommendations(recs, os.path.join(out_dir, "recs_test.tsv"))
        with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"base_model": base, "k": k, **chosen, "test_metrics": metrics,
                       f"changed_share@{k}": share_changed(recs, k)}, f, indent=2)
        if test_sweep:
            table = sweep(base_recs, reranker, test_settings(name, cfg["lambdas"], chosen), evaluator, k)
            selected = table[table["lambda"] == chosen["lambda"]].iloc[0]
            if abs(selected[f"NDCG@{k}"] - metrics[f"NDCG@{k}"]) > 1e-9:  # the sweep must contain the list above
                raise AssertionError((base, name, selected[f"NDCG@{k}"], metrics[f"NDCG@{k}"]))
            table.to_csv(os.path.join(out_dir, "sweep_test.csv"), index=False)
        rows.append({"base": base, "reranker": name, "lambda": chosen["lambda"],
                     "inactive_weight": chosen.get("inactive_weight", float("nan")),
                     "exposure_target": chosen.get("exposure_target"), f"Changed@{k}": share_changed(recs, k), **metrics})
        params = ", ".join(f"{a} = {chosen[a]}" for a in ("lambda",) + FAIRNESS_PARAMS if a in chosen)
        print(f"reranked {base} with {name} ({params})", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bases", nargs="*", help="base models to rerank (default: all with configs/rerankers/<base>.yaml)")
    parser.add_argument("--no-test-sweep", action="store_true", help="skip the lambda sweep on test (no sweep_test.csv)")
    args = parser.parse_args()
    bases = args.bases or configured_bases()
    if not bases:
        raise SystemExit("No configs/rerankers/<base>.yaml found. Run tune_rerankers.py first.")

    train, _, _ = load_split()
    evaluator = Evaluator.from_outputs("test")
    rerankers = build_rerankers(train, load_item_genres(), evaluator.short_head, evaluator.active_users)

    rows, k = [], None
    for base in bases:
        cfg = load_rerank_config(base)
        if k is not None and cfg["k"] != k:
            raise SystemExit(f"{base} was tuned with k = {cfg['k']}, the others with k = {k}")
        k = cfg["k"]
        base_rows = run_base(base, cfg, evaluator, rerankers, test_sweep=not args.no_test_sweep)
        rows.extend(base_rows)
        table = pd.DataFrame(base_rows)
        table.index = [base if r == "none" else f"{base} + {r}" for r in table["reranker"]]
        with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
            print(f"\n=== {base}, test, k = {k}: base vs reranked (settings selected on validation) ===")
            print(table[["lambda", f"Changed@{k}"] + [f"{m}@{k}" for m in SHOWN]].T)  # metrics as rows: fits the terminal
            print()

    out_dir = os.path.join(OUTPUT_DIR, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "rerank_test_comparison.csv")
    table = pd.DataFrame(rows)
    if args.bases and os.path.exists(path):  # with --bases, keep the rows of the bases that were not rerun
        previous = pd.read_csv(path)
        table = pd.concat([previous[~previous["base"].isin(bases)], table], ignore_index=True)
    table.to_csv(path, index=False)
    print("Wrote outputs/Rerank_<reranker>/<base>/recs_test.tsv, test_metrics.json"
          + ("" if args.no_test_sweep else " and sweep_test.csv"))
    print(f"Wrote {out_dir}/rerank_test_comparison.csv")


if __name__ == "__main__":
    main()
