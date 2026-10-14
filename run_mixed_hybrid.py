"""Run the frozen Mixed Hybrid configuration once on the TEST set.

Loads configs/mixed_hybrid_best.yaml (written by tune_mixed_hybrid.py), mixes each
model's test lists, and compares the hybrid with the standalone models on the same
test ground truth. Never retunes on test.

For the report, it also scores (but does not select from) the configuration each other
selection method picked on validation.

Example:
    python run_mixed_hybrid.py
"""

import json
import os

import pandas as pd
import yaml

from recbole.utils.experiment import CONFIG_DIR, OUTPUT_DIR, load_split
from recbole.model.general_recommender.mixed_hybrid import (
    evaluate_recommendations,
    load_sources,
    mix_recommendations,
    quotas_to_label,
    save_recommendations,
)
from tune_mixed_hybrid import KINDS  # the candidate models of the tuning

CANDIDATES = list(KINDS)


def main():
    config_path = os.path.join(CONFIG_DIR, "mixed_hybrid_best.yaml")
    if not os.path.exists(config_path):
        raise SystemExit(f"Missing {config_path}. Run tune_mixed_hybrid.py first.")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    models, quotas = cfg["models"], cfg["quotas"]

    _, _, test = load_split()
    sources = load_sources(CANDIDATES, stage="test")

    rows = []
    for name in CANDIDATES:
        rows.append({"model": name, "role": "standalone", **evaluate_recommendations(sources[name], test)})

    mixed = mix_recommendations(sources, models, quotas)
    hybrid = evaluate_recommendations(mixed, test)
    rows.append({"model": f"Mixed Hybrid: {quotas_to_label(quotas)}", "role": f"FINAL ({cfg['selection_method']})", **hybrid})

    for method, alt in cfg.get("alternatives", {}).items():
        if method == cfg["selection_method"]:
            continue
        metrics = evaluate_recommendations(mix_recommendations(sources, alt["models"], alt["quotas"]), test)
        rows.append({"model": f"Mixed Hybrid: {quotas_to_label(alt['quotas'])}", "role": f"not selected ({method})", **metrics})

    table = pd.DataFrame(rows)
    print("--- Test metrics ---")
    print(table.to_string(index=False, float_format="%.4f"))
    print(f"\nFinal configuration selected on validation by {cfg['primary_metric']} ({cfg['selection_method']}).")

    out_dir = os.path.join(OUTPUT_DIR, "MixedHybrid")
    recs_path = os.path.join(out_dir, "recs_test.tsv")
    save_recommendations(mixed, recs_path)
    table.to_csv(os.path.join(out_dir, "test_comparison.csv"), index=False)
    metrics_path = os.path.join(out_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({**cfg, "mixed_hybrid_test": hybrid, "comparison_test": rows}, f, indent=2)

    print()
    print(f"Wrote {recs_path}")
    print(f"Wrote {metrics_path} and test_comparison.csv")


if __name__ == "__main__":
    main()
