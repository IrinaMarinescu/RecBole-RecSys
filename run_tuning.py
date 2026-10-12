"""Grid-search the hyperparameters of an experiment on the VALIDATION set.

The grid is read from configs/hyper/<experiment>.yaml (one list of values per parameter).
Results of every run go to outputs/tuning/<experiment>.csv, and the best parameters
(by the valid_metric in configs/base.yaml, NDCG@10) are written to configs/tuned/<experiment>.yaml,
which run_model.py picks up automatically.

Examples:
    python run_tuning.py UserKNN
    python run_tuning.py ItemKNN BPR
"""

import argparse
import itertools
import logging
import os

import pandas as pd
import yaml

from recbole.utils.experiment import CONFIG_DIR, HYPER_DIR, OUTPUT_DIR, train


def tune(experiment):
    with open(os.path.join(HYPER_DIR, f"{experiment}.yaml")) as f:
        grid = yaml.safe_load(f)
    names = list(grid)
    combos = list(itertools.product(*(grid[n] for n in names)))

    rows = []
    for i, values in enumerate(combos, 1):
        params = dict(zip(names, values))
        config, *_, result = train(experiment, params, use_tuned=False, verbose=False)
        metric = config["valid_metric"].lower()
        row = {**params, f"valid_{metric}": result["valid"][metric]}
        row.update({f"test_{k}": v for k, v in result["test"].items()})
        rows.append(row)
        print(f"[{experiment}] {i}/{len(combos)} {params} -> valid {metric} = {row[f'valid_{metric}']:.4f}", flush=True)

    table = pd.DataFrame(rows).sort_values(f"valid_{metric}", ascending=not config["valid_metric_bigger"])
    os.makedirs(os.path.join(OUTPUT_DIR, "tuning"), exist_ok=True)
    table.to_csv(os.path.join(OUTPUT_DIR, "tuning", f"{experiment}.csv"), index=False)

    best = {n: getattr(v, "item", lambda: v)() for n in names for v in [table[n].iloc[0]]}  # numpy -> python, so ints stay ints
    with open(os.path.join(CONFIG_DIR, "tuned", f"{experiment}.yaml"), "w") as f:
        f.write(f"# Best by valid {metric} = {table.iloc[0][f'valid_{metric}']:.4f} (written by run_tuning.py)\n")
        yaml.safe_dump(best, f, sort_keys=False)
    print(f"[{experiment}] best: {best}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="+")
    args = parser.parse_args()
    for experiment in args.experiments:
        tune(experiment)
        logging.getLogger().handlers.clear()


if __name__ == "__main__":
    main()
