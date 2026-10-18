"""Task 2.5: how each model performs for different groups of users and items.

The models are not retrained per group: each model's exported recommendations (outputs/<model>/recs_<stage>.tsv,
trained on all users) are evaluated separately per group. Group definitions are in recbole/utils/evaluation.py
(Evaluator.user_groupings / item_groupings):

    users: activity (tertiles of #training ratings), mainstreaminess (tertiles of the mean popularity of their
           training items), gender, age (<25, 25-34, 35-44, 45+), occupation
    items: popularity (head = top 20% / middle / tail = bottom 50% by training interactions), genre
"""

import argparse
import os

import pandas as pd

from recbole.utils.evaluation import GROUP_ORDER, Evaluator, load_user_demographics
from recbole.utils.experiment import OUTPUT_DIR, load_recs
from run_evaluation import available_models

MIN_PRINTED_GROUP = 20  # smaller user groups are kept in the CSV but not printed: too noisy


def show(title, table, value, models):
    """Print one model x group table, groups in their natural order (or largest first)."""
    grouping = table["grouping"].iloc[0]
    pivot = table.pivot_table(index="model", columns="group", values=value).reindex(models)
    size = table.drop_duplicates("group").set_index("group")["n_users" if "n_users" in table else "n_items"]
    order = GROUP_ORDER.get(grouping) or list(size.sort_values(ascending=False).index)
    print(f"\n{title}")
    print(pivot[[g for g in order if g in pivot.columns]])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="models to analyse (default: all with exported lists)")
    parser.add_argument("--stage", choices=["valid", "test"], default="test")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    models = args.models or available_models(args.stage)
    evaluator = Evaluator.from_outputs(args.stage)
    user_groupings = evaluator.user_groupings(load_user_demographics())
    item_groupings = evaluator.item_groupings()

    user_tables, item_tables = [], []
    for model in models:
        recs = load_recs(model, args.stage)
        user_tables.append(evaluator.user_group_table(recs, args.k, user_groupings).assign(model=model))
        item_tables.append(evaluator.item_group_table(recs, args.k, item_groupings).assign(model=model))
        print(f"analysed {model}", flush=True)

    def tidy(tables):
        table = pd.concat(tables)
        return table[["model"] + [c for c in table.columns if c != "model"]]

    users, items = tidy(user_tables), tidy(item_tables)
    out_dir = os.path.join(OUTPUT_DIR, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    users.to_csv(os.path.join(out_dir, f"{args.stage}_user_group_analysis.csv"), index=False)
    items.to_csv(os.path.join(out_dir, f"{args.stage}_item_group_analysis.csv"), index=False)

    k = args.k
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        for grouping in user_groupings:
            part = users[(users["grouping"] == grouping) & (users["n_users"] >= MIN_PRINTED_GROUP)]
            info = part.drop_duplicates("group").set_index("group")
            order = GROUP_ORDER.get(grouping) or list(info["n_users"].sort_values(ascending=False).index)
            sizes = ", ".join(f"{g} {info.loc[g, 'n_users']} users / {info.loc[g, 'avg_test_items']:.1f} test items"
                              for g in order if g in info.index)
            hidden = users[(users["grouping"] == grouping) & (users["n_users"] < MIN_PRINTED_GROUP)]["group"].unique()
            note = f"; not shown (< {MIN_PRINTED_GROUP} users): {', '.join(hidden)}" if len(hidden) else ""
            show(f"=== NDCG@{k} per {grouping} group  ({sizes}{note})", part, f"NDCG@{k}", models)
        popularity = items[items["grouping"] == "popularity"]
        show("=== Share of recommendation slots per popularity group (catalogue: head 20%, middle 30%, tail 50%)",
             popularity, "exposure_share", models)
        show(f"=== Recall@{k} on the items of each popularity group", popularity, f"recall@{k}", models)
    print(f"\nWrote {out_dir}/{args.stage}_user_group_analysis.csv and {args.stage}_item_group_analysis.csv")


if __name__ == "__main__":
    main()
