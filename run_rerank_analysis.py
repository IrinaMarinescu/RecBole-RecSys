"""Task 3.2: accuracy vs beyond-accuracy trade-off of the rerankers, across base models.

Reads what tune_rerankers.py (validation sweeps, selected settings) and run_rerankers.py (test lists, test
sweeps, rerank_test_comparison.csv) wrote; never reranks. Writes tables to outputs/evaluation/rerank_*.csv,
figures to outputs/evaluation/figures/ and a paste-ready summary to outputs/evaluation/rerank_summary.md.

    python run_rerank_analysis.py                              # every base with reranked test lists
    python run_rerank_analysis.py --main-base RecVAE --no-groups --no-figures
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

from recbole.model.general_recommender.rerankers import HYBRID_BASES, RERANKER_NAMES, budget_sensitivity, output_name
from recbole.utils.evaluation import GROUP_ORDER, Evaluator
from recbole.utils.experiment import OUTPUT_DIR, load_recs
from tune_mixed_hybrid import KINDS
from tune_rerankers import configured_bases, load_rerank_config, no_worse_for

EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluation")
FIG_DIR = os.path.join(EVAL_DIR, "figures")
ACCURACY = ["NDCG", "Recall", "Precision", "MRR"]
BEYOND = ["ILD", "Miscalibration", "Novelty", "APLT", "TailExposure", "Coverage", "Gini", "UserNDCGGap", "TailHitShare"]
LOWER_IS_BETTER = {"Miscalibration", "Gini", "UserNDCGGap"}
# Per-user counterpart of each reranker's target for the paired tests (TailExposure has no per-user form;
# over full-length lists it equals the mean APLT).
PER_USER_TARGET = {"MMR": "ILD", "Calibration": "Miscalibration", "Fairness": "APLT", "Popularity": "APLT"}
KEY = ["base", "reranker", "stage", "lambda", "inactive_weight"]
DEFAULT_BUDGETS = [0.01, 0.02, 0.05, 0.10]
LOW_ACCURACY = 0.05  # a relative budget on an NDCG this small is mostly noise: such bases are flagged
KIND_OF = {**KINDS, **{h: "hybrid" for h in HYBRID_BASES}}


def metric_name(column):
    return column.split("@")[0]


def higher_is_better(column):
    return metric_name(column) not in LOWER_IS_BETTER


def signed_pct(value, reference, column):
    """Change in % vs the reference, positive = better in the metric's own direction."""
    if not reference or not np.isfinite(reference):
        return float("nan")
    return (1.0 if higher_is_better(column) else -1.0) * 100.0 * (value - reference) / abs(reference)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def load_sweeps(bases, configs):
    """All lambda sweeps in one long frame: KEY columns, `selected` (the setting chosen on validation), `target`."""
    parts = []
    for base in bases:
        for name, chosen in configs[base]["rerankers"].items():
            out_dir = os.path.join(OUTPUT_DIR, output_name(name, base))
            for stage, file in (("valid", "tuning_valid.csv"), ("test", "sweep_test.csv")):
                path = os.path.join(out_dir, file)
                if not os.path.exists(path):
                    print(f"Warning: missing {path}")
                    continue
                table = pd.read_csv(path)
                if "inactive_weight" not in table.columns:
                    table["inactive_weight"] = np.nan
                selected = table["lambda"] == chosen["lambda"]
                if "inactive_weight" in chosen and chosen["lambda"] > 0:
                    selected &= table["inactive_weight"] == chosen["inactive_weight"]
                table.insert(0, "base", base)
                table.insert(1, "reranker", name)
                table.insert(2, "stage", stage)
                table["selected"] = selected
                table["target"] = chosen["target"]
                parts.append(table)
    sweeps = pd.concat(parts, ignore_index=True)
    rest = [c for c in sweeps.columns if c not in KEY + ["selected", "target"]]
    return sweeps[KEY + ["selected", "target"] + rest]


def relative_deltas(comparison, k):
    """Per (base, reranker): every metric's change in % vs the base list (`<metric>_pct`), Changed@k as is."""
    metrics = [c for c in comparison.columns if "@" in c and not c.startswith("Changed")]
    base = comparison[comparison["reranker"] == "none"].set_index("base")
    out = comparison[comparison["reranker"] != "none"].copy()
    for m in metrics:
        ref = out["base"].map(base[m])
        out[f"{m}_pct"] = (100.0 * (out[m] - ref) / ref.abs()).replace([np.inf, -np.inf], np.nan)
    keep = ["base", "reranker", "lambda", "inactive_weight", f"Changed@{k}"] + [f"{m}_pct" for m in metrics]
    return out[keep].reset_index(drop=True)


def side_effects(deltas, targets, k):
    """Per reranker and metric: mean / min / max change in % over the bases, and whether the metric is its target."""
    pct = [c for c in deltas.columns if c.endswith("_pct")]
    long = deltas.melt(["base", "reranker"], pct, "metric", "pct")
    long["metric"] = long["metric"].str[:-4]
    agg = (long.groupby(["reranker", "metric"], sort=False)["pct"]
           .agg(mean_pct="mean", min_pct="min", max_pct="max", n_bases="count").reset_index())
    agg["is_target"] = [targets.get(r) == m for r, m in zip(agg["reranker"], agg["metric"])]
    order = {f"{m}@{k}": i for i, m in enumerate(ACCURACY + BEYOND)}
    agg["_r"] = agg["reranker"].map({n: i for i, n in enumerate(RERANKER_NAMES)})
    agg["_m"] = agg["metric"].map(lambda m: order.get(m, len(order)))
    return agg.sort_values(["_r", "_m"], kind="stable").drop(columns=["_r", "_m"]).reset_index(drop=True)


def budget_table(sweeps, configs, budgets, k):
    """What each accuracy budget buys: the setting select_lambda picks on validation and its test metrics."""
    primary, rows = f"NDCG@{k}", []
    for (base, name), part in sweeps.groupby(["base", "reranker"], sort=False):
        target = configs[base]["rerankers"][name]["target"]
        valid = part[part["stage"] == "valid"].drop(columns=["base", "reranker", "stage", "selected", "target"])
        if valid["inactive_weight"].isna().all():
            valid = valid.drop(columns="inactive_weight")
        test = part[part["stage"] == "test"]
        test_ref = test[test["lambda"] == 0.0]
        sens = budget_sensitivity(valid, target, higher_is_better(target), budgets, primary, no_worse_for(name, k))
        for _, r in sens.iterrows():
            row = {"base": base, "reranker": name, "max_loss": r["max_loss"], "lambda": r["lambda"],
                   "inactive_weight": r["inactive_weight"], "admissible": r["admissible"],
                   "valid_NDCG": r[primary], "valid_target": r[target], "valid_target_gain_pct": r["target_gain_pct"],
                   "test_NDCG": np.nan, "test_NDCG_loss_pct": np.nan, "test_target": np.nan, "test_target_gain_pct": np.nan}
            match = test[test["lambda"] == r["lambda"]]
            if r["lambda"] > 0 and not np.isnan(r["inactive_weight"]):  # the test sweep has one inactive_weight only
                match = match[match["inactive_weight"] == r["inactive_weight"]]
            if len(match) and len(test_ref):
                t, t0 = match.iloc[0], test_ref.iloc[0]
                row.update(test_NDCG=t[primary], test_NDCG_loss_pct=100.0 * (t0[primary] - t[primary]) / t0[primary],
                           test_target=t[target], test_target_gain_pct=signed_pct(t[target], t0[target], target))
            rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(users_base, users_reranked, metrics):
    """Per-user paired comparison of two lists: Wilcoxon signed-rank and paired t-test, shares of users better / worse."""
    merged = users_base[["user_id"] + metrics].merge(users_reranked[["user_id"] + metrics], on="user_id",
                                                     suffixes=("_base", "_rerank"))
    rows = []
    for m in metrics:
        d = (merged[f"{m}_rerank"] - merged[f"{m}_base"]).to_numpy(dtype=float)
        better, worse = (d > 0, d < 0) if higher_is_better(m) else (d < 0, d > 0)
        if np.any(d != 0) and np.std(d) > 0:
            wilcoxon_p = float(stats.wilcoxon(d, zero_method="zsplit").pvalue)  # keeps the unchanged users
            ttest_p = float(stats.ttest_rel(merged[f"{m}_rerank"], merged[f"{m}_base"]).pvalue)
        else:
            wilcoxon_p = ttest_p = 1.0 if not np.any(d != 0) else 0.0
        rows.append({"metric": m, "n_users": len(d), "mean_base": merged[f"{m}_base"].mean(),
                     "mean_reranked": merged[f"{m}_rerank"].mean(), "mean_diff": d.mean(),
                     "share_improved": better.mean(), "share_worse": worse.mean(), "share_unchanged": (d == 0).mean(),
                     "wilcoxon_p": wilcoxon_p, "ttest_p": ttest_p})
    return pd.DataFrame(rows)


def holm(pvalues):
    """Holm step-down adjustment of a family of p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    adjusted, running = np.empty_like(p), 0.0
    for rank, idx in enumerate(np.argsort(p, kind="stable")):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def significance_table(per_user, k):
    """paired_tests of every reranked list vs its base, for NDCG@k and the reranker's per-user target;
    Holm-corrected within each (base, metric) family (the rerankers compared on the same base)."""
    rows = []
    for (base, name), users in per_user.items():
        if name == "none":
            continue
        metrics = [f"NDCG@{k}", f"{PER_USER_TARGET[name]}@{k}"]
        rows.append(paired_tests(per_user[(base, "none")], users, metrics).assign(base=base, reranker=name))
    table = pd.concat(rows, ignore_index=True)
    table["wilcoxon_p_holm"] = table.groupby(["base", "metric"])["wilcoxon_p"].transform(
        lambda s: pd.Series(holm(s.to_numpy()), index=s.index))
    front = ["base", "reranker", "metric"]
    return table[front + [c for c in table.columns if c not in front]]


def group_tables(evaluator, lists, k):
    """Task 2.5 group tables (user activity / mainstreaminess, item popularity) for every list."""
    user_groupings = {g: v for g, v in evaluator.user_groupings().items() if g in ("activity", "mainstreaminess")}
    item_groupings = {"popularity": evaluator.item_groupings()["popularity"]}
    users, items = [], []
    for (base, name), recs in lists.items():
        users.append(evaluator.user_group_table(recs, k, user_groupings).assign(base=base, reranker=name))
        items.append(evaluator.item_group_table(recs, k, item_groupings).assign(base=base, reranker=name))

    def tidy(tables):
        table = pd.concat(tables, ignore_index=True)
        return table[["base", "reranker"] + [c for c in table.columns if c not in ("base", "reranker")]]

    return tidy(users), tidy(items)


# --------------------------------------------------------------------------- #
# Summary (markdown)
# --------------------------------------------------------------------------- #
def _fmt_setting(lam, weight):
    return f"λ={lam:g}" + ("" if pd.isna(weight) else f", w={weight:g}")


def summary_markdown(main, k, configs, comparison, deltas, side, budget, significance, user_groups, item_groups,
                     figures, max_loss):
    primary = f"NDCG@{k}"
    targets = {n: c["target"] for n, c in configs[main]["rerankers"].items()}
    bases = list(configs)
    out = [f"# Rerankers: accuracy vs beyond-accuracy (test, k = {k})\n",
           f"Settings selected on validation (best target metric with {primary} >= {1 - max_loss:.2f} x the base list; "
           "Fairness: and no wider active/inactive NDCG gap). Test is evaluated once at those settings; the test "
           "lambda sweeps only draw the curves. Lower is better for Miscalibration, Gini and UserNDCGGap.\n"]

    # 1. selected settings
    out.append("## 1. Selected settings per base model\n")
    out.append("Cell: setting, change of the target metric (sign = better/worse), change of NDCG@k; all on test.\n")
    cells = {}
    for _, r in deltas.iterrows():
        target = targets[r["reranker"]]
        cells[(r["base"], r["reranker"])] = (f"{_fmt_setting(r['lambda'], r['inactive_weight'])}: "
                                            f"{signed_pct(1 + r[f'{target}_pct'] / 100, 1, target):+.0f}% {metric_name(target)}, "
                                            f"{r[f'{primary}_pct']:+.1f}% NDCG")
    table = pd.DataFrame({f"+{n}": [cells.get((b, n), "") for b in bases] for n in RERANKER_NAMES}, index=bases)
    out.append(table.to_markdown() + "\n")
    base_ndcg = comparison[comparison["reranker"] == "none"].set_index("base")[primary]
    off = deltas[deltas["lambda"] == 0.0]
    if len(off):
        out.append("Not applied (λ = 0 selected, nothing fits the budget): " +
                   ", ".join(f"{b} + {r}" for b, r in zip(off["base"], off["reranker"])) + ".\n")
    low = base_ndcg[base_ndcg < LOW_ACCURACY]
    if len(low):
        out.append(f"Bases with {primary} < {LOW_ACCURACY} (a relative budget on such a small value is mostly noise): " +
                   ", ".join(f"{b} ({v:.3f})" for b, v in low.items()) + ".\n")

    # 2. main base
    out.append(f"## 2. {main}: base list vs reranked lists\n")
    rows = [f"{m}@{k}" for m in ACCURACY + BEYOND if f"{m}@{k}" in comparison.columns]
    main_cmp = comparison[comparison["base"] == main].set_index("reranker")
    main_del = deltas[deltas["base"] == main].set_index("reranker")
    table = pd.DataFrame(index=rows + [f"Changed@{k}"])
    table[main] = [f"{main_cmp.loc['none', m]:.4f}" for m in rows] + ["-"]
    for n in RERANKER_NAMES:
        if n not in main_del.index:
            continue
        table[f"+{n} ({_fmt_setting(main_del.loc[n, 'lambda'], main_del.loc[n, 'inactive_weight'])})"] = [
            f"{main_cmp.loc[n, m]:.4f} ({main_del.loc[n, f'{m}_pct']:+.1f}%)" for m in rows
        ] + [f"{main_del.loc[n, f'Changed@{k}']:.2f}"]
    out.append(table.to_markdown() + "\n")

    # 3. across bases
    out.append(f"## 3. Change in % over all {len(bases)} bases: mean (min … max); **bold** = the reranker's target\n")
    table = pd.DataFrame(index=rows)
    for n in RERANKER_NAMES:
        s = side[side["reranker"] == n].set_index("metric")
        col = []
        for m in rows:
            if m not in s.index:
                col.append("")
                continue
            r = s.loc[m]
            text = f"{r['mean_pct']:+.1f} ({r['min_pct']:+.1f} … {r['max_pct']:+.1f})"
            col.append(f"**{text}**" if r["is_target"] else text)
        table[f"+{n}"] = col
    out.append(table.to_markdown() + "\n")

    # 4. budget
    out.append(f"## 4. Accuracy budget: what a larger accepted NDCG loss buys ({main}, test)\n")
    b = budget[budget["base"] == main]
    table = pd.DataFrame(index=[f"{x:.0%}" for x in sorted(b["max_loss"].unique())])
    for n in RERANKER_NAMES:
        part = b[b["reranker"] == n].set_index("max_loss").sort_index()
        if part.empty:
            continue
        table[f"+{n}"] = [f"{_fmt_setting(r['lambda'], r['inactive_weight'])}: {r['test_target_gain_pct']:+.0f}% "
                          f"{metric_name(targets[n])}, −{r['test_NDCG_loss_pct']:.1f}% NDCG" if np.isfinite(r["test_NDCG_loss_pct"])
                          else f"{_fmt_setting(r['lambda'], r['inactive_weight'])}: (not in the test sweep)"
                          for _, r in part.iterrows()]
    out.append(table.to_markdown() + "\n")
    default = budget[np.isclose(budget["max_loss"], max_loss)]
    out.append(f"At the default budget ({max_loss:.0%}) λ = 0 is selected for {int((default['lambda'] == 0).sum())} "
               f"of {len(default)} (base, reranker) pairs.\n")

    # 5. groups
    if item_groups is not None:
        out.append(f"## 5. Groups ({main})\n")
        ig = item_groups[(item_groups["base"] == main)]
        order = ["none"] + [n for n in RERANKER_NAMES if n in set(ig["reranker"])]
        pivot = ig.pivot_table(index="reranker", columns="group", values="exposure_share").reindex(order)
        pivot = pivot[[g for g in ("head", "middle", "tail") if g in pivot.columns]]
        catalog = ig.drop_duplicates("group").set_index("group")["catalog_share"]
        pivot.loc["(catalogue share)"] = catalog.reindex(pivot.columns)
        out.append("Share of the recommendation slots per item popularity group:\n")
        out.append(pivot.map(lambda v: f"{v:.1%}").to_markdown() + "\n")
        ug = user_groups[user_groups["base"] == main]
        for grouping in ("activity", "mainstreaminess"):
            part = ug[ug["grouping"] == grouping]
            pivot = part.pivot_table(index="reranker", columns="group", values=primary).reindex(order)
            pivot = pivot[[g for g in GROUP_ORDER.get(grouping, pivot.columns) if g in pivot.columns]]
            sizes = part.drop_duplicates("group").set_index("group")["n_users"]
            pivot.columns = [f"{g} (n={sizes[g]})" for g in pivot.columns]
            out.append(f"{primary} per user {grouping} group:\n")
            out.append(pivot.map(lambda v: f"{v:.4f}").to_markdown() + "\n")
    else:
        out.append("## 5. Groups\n\n(skipped with --no-groups)\n")

    # 6. significance
    out.append(f"## 6. Paired per-user tests vs the base list (Wilcoxon, Holm-corrected over the rerankers of a base)\n")
    s = significance[significance["base"] == main][["reranker", "metric", "mean_diff", "share_improved", "share_worse",
                                                    "wilcoxon_p_holm"]]
    out.append(f"{main}:\n")
    out.append(s.to_markdown(index=False, floatfmt=(".0f", ".0f", "+.4f", ".2f", ".2f", ".3g")) + "\n")
    ndcg = significance[significance["metric"] == primary]
    sig_drop = ndcg[(ndcg["wilcoxon_p_holm"] < 0.05) & (ndcg["mean_diff"] < 0)]
    tgt = significance[significance["metric"] != primary]
    sig_gain = tgt[(tgt["wilcoxon_p_holm"] < 0.05) & ((tgt["mean_diff"] > 0) == tgt["metric"].map(higher_is_better))]
    out.append(f"Over all bases: the NDCG drop is significant (p < 0.05) for {len(sig_drop)} of {len(ndcg)} reranked "
               f"lists, the target improvement for {len(sig_gain)} of {len(tgt)}.\n")

    # 7. figures and notes
    out.append("## 7. Figures and notes\n")
    out.extend(f"- `figures/{f}`" for f in figures)
    if not figures:
        out.append("- (figures not generated; run without --no-figures)")
    out.append("\nNotes for the report: ML-100K, ratings >= 3, seed 2020, k = 10; the 100 candidates per user are "
               "the base model's top-100; TailExposure equals the mean APLT over full lists; MixedHybrid lists are "
               "reranked with rank-based relevance (its scores come from different models); the per-user tests treat "
               "users as independent samples.\n")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Figures (matplotlib is imported here only)
# --------------------------------------------------------------------------- #
SURFACE, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#d9d8d3"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order
KIND_ORDER = ["neighbourhood", "latent factor", "content", "hybrid"]
KIND_COLOR = dict(zip(KIND_ORDER, SLOTS))
RERANKER_COLOR = dict(zip(RERANKER_NAMES, SLOTS))
POPULARITY_COLOR = {"head": "#104281", "middle": "#2a78d6", "tail": "#86b6ef"}  # one-hue ordinal ramp


def _base_style(base, main):
    kind = KIND_OF.get(base, "other")
    return {"color": KIND_COLOR.get(kind, INK2), "ls": "--" if kind == "hybrid" else "-",
            "lw": 2.2 if base == main else 1.3, "alpha": 1.0 if base == main else 0.85}


def _axes(fig, ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK2)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=150, facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return name


def _kind_legend(ax, bases, extra=()):
    from matplotlib.lines import Line2D
    kinds = [kd for kd in KIND_ORDER if any(KIND_OF.get(b) == kd for b in bases)]
    handles = [Line2D([], [], color=KIND_COLOR[kd], ls="--" if kd == "hybrid" else "-", lw=1.5, label=kd) for kd in kinds]
    handles += list(extra)
    ax.legend(handles=handles, fontsize=7, frameon=False, labelcolor=INK2)


def fig_tradeoff(plt, sweeps, name, target, k, max_loss, main, selected_w):
    primary = f"NDCG@{k}"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, facecolor=SURFACE)
    part = sweeps[sweeps["reranker"] == name]
    bases = list(dict.fromkeys(part["base"]))
    for ax, stage in zip(axes, ("valid", "test")):
        for base in bases:
            g = part[(part["base"] == base) & (part["stage"] == stage)]
            if name == "Fairness":  # the validation sweep is lambda x weight: draw the selected weight's curve
                g = g[(g["lambda"] == 0) | (g["inactive_weight"] == selected_w.get(base))]
            g = g.sort_values("lambda")
            if g.empty:
                continue
            ref = g[g["lambda"] == 0].iloc[0]
            x, y, st = g[target].to_numpy(), (g[primary] / ref[primary]).to_numpy(), _base_style(base, main)
            ax.plot(x, y, st["ls"], color=st["color"], lw=st["lw"], alpha=st["alpha"], marker="o", ms=2.5)
            ax.plot(x[0], y[0], "o", mfc=SURFACE, mec=st["color"], ms=7, mew=1.5)
            sel = g[g["selected"]]
            if len(sel):
                ax.plot(sel[target], sel[primary] / ref[primary], "*", color=st["color"], ms=13, mec=SURFACE, mew=0.8, zorder=4)
            ax.annotate(base, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.5, color=INK2, va="center")
        ax.axhline(1 - max_loss, ls=(0, (4, 3)), color=INK2, lw=1)
        ax.annotate(f"{max_loss:.0%} accuracy budget", (ax.get_xlim()[0], 1 - max_loss), xytext=(4, -10),
                    textcoords="offset points", fontsize=7, color=INK2)
        _axes(fig, ax, f"{stage}: {metric_name(target)} vs accuracy, λ = 0 … 1", f"{target}" +
              (" (lower is better)" if not higher_is_better(target) else ""),
              f"{primary} relative to the base list" if stage == "valid" else None)
    from matplotlib.lines import Line2D
    _kind_legend(axes[0], bases, [Line2D([], [], marker="o", mfc=SURFACE, mec=INK2, ls="", label="λ = 0 (base list)"),
                                  Line2D([], [], marker="*", color=INK2, ls="", ms=10, label="selected λ")])
    fig.suptitle(f"+{name}: every base model's trade-off curve", fontsize=11, color=INK, x=0.01, ha="left")
    return _save(fig, f"rerank_tradeoff_{name}.png")


def _metric_labels(metrics):
    return [metric_name(m) + (" ↓" if not higher_is_better(m) else "") for m in metrics]


def fig_relative_change(plt, side, deltas, targets, k):
    metrics = [f"{m}@{k}" for m in ["NDCG", "Recall"] + BEYOND]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), facecolor=SURFACE, sharex=True)
    for ax, name in zip(axes.flat, RERANKER_NAMES):
        s = side[side["reranker"] == name].set_index("metric").reindex(metrics)
        colors = [RERANKER_COLOR[name] if m == targets.get(name) else NEUTRAL for m in metrics]
        xs = np.arange(len(metrics))
        ax.bar(xs, s["mean_pct"], color=colors, width=0.62, edgecolor=SURFACE, linewidth=1, zorder=2)
        d = deltas[deltas["reranker"] == name]
        for i, m in enumerate(metrics):
            ax.plot(np.full(len(d), i), d[f"{m}_pct"], "o", ms=3.5, color=INK2, mec=SURFACE, mew=0.6, alpha=0.9, zorder=3)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.plot([-0.35, 0.35], [-100 * 0.05] * 2, ls=(0, (3, 2)), color=INK2, lw=1)
        ax.set_xticks(xs, _metric_labels(metrics), rotation=40, ha="right")
        _axes(fig, ax, f"+{name} (target: {metric_name(targets.get(name, ''))})", None, "change vs base list (%)")
    fig.suptitle("Relative change of every metric: bars = mean over the base models, dots = single base models; "
                 "coloured bar = the reranker's target, dashed = the 5 % NDCG budget", fontsize=9.5, color=INK, x=0.01, ha="left")
    return _save(fig, "rerank_relative_change.png")


def fig_relative_change_main(plt, deltas, main, k):
    metrics = [f"{m}@{k}" for m in ["NDCG", "Recall"] + BEYOND]
    d = deltas[deltas["base"] == main].set_index("reranker")
    names = [n for n in RERANKER_NAMES if n in d.index]
    fig, ax = plt.subplots(figsize=(11, 4.4), facecolor=SURFACE)
    xs, width = np.arange(len(metrics)), 0.8 / max(len(names), 1)
    for j, name in enumerate(names):
        ax.bar(xs + (j - (len(names) - 1) / 2) * width, [d.loc[name, f"{m}_pct"] for m in metrics], width * 0.92,
               color=RERANKER_COLOR[name], edgecolor=SURFACE, linewidth=1, label=f"+{name}", zorder=2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(xs, _metric_labels(metrics), rotation=40, ha="right")
    _axes(fig, ax, f"{main}: change of every metric per reranker (test, selected settings)", None, "change vs base list (%)")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    return _save(fig, f"rerank_relative_change_{main}.png")


def fig_budget(plt, budget, targets, main, max_loss):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), facecolor=SURFACE, sharex=True)
    bases = list(dict.fromkeys(budget["base"]))
    for ax, name in zip(axes, RERANKER_NAMES):
        for base in bases:
            g = budget[(budget["base"] == base) & (budget["reranker"] == name)].sort_values("max_loss")
            if g.empty:
                continue
            st = _base_style(base, main)
            ax.plot(100 * g["max_loss"], g["test_target_gain_pct"], st["ls"], color=st["color"], lw=st["lw"],
                    alpha=st["alpha"], marker="o", ms=3)
            ax.annotate(base, (100 * g["max_loss"].iloc[-1], g["test_target_gain_pct"].iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=6.5, color=INK2, va="center")
        ax.axvline(100 * max_loss, ls=(0, (2, 2)), color=INK2, lw=0.8)
        ax.set_xticks(sorted(100 * budget["max_loss"].unique()))
        _axes(fig, ax, f"+{name}: {metric_name(targets.get(name, ''))}", "accepted NDCG loss on validation (%)",
              "target gain on test (%)" if name == RERANKER_NAMES[0] else None)
    _kind_legend(axes[0], bases)
    return _save(fig, "rerank_budget_sensitivity.png")


def fig_item_exposure(plt, item_groups):
    bases = list(dict.fromkeys(item_groups["base"]))
    ncols = 4
    nrows = math.ceil(len(bases) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.9 * nrows), facecolor=SURFACE, sharey=True, squeeze=False)
    catalog = item_groups.drop_duplicates("group").set_index("group")["catalog_share"]
    for ax, base in zip(axes.flat, bases):
        g = item_groups[item_groups["base"] == base]
        order = ["none"] + [n for n in RERANKER_NAMES if n in set(g["reranker"])]
        pivot = g.pivot_table(index="reranker", columns="group", values="exposure_share").reindex(order)
        bottom = np.zeros(len(order))
        for group in ("head", "middle", "tail"):
            vals = pivot[group].fillna(0).to_numpy()
            ax.bar(range(len(order)), vals, bottom=bottom, color=POPULARITY_COLOR[group], width=0.68,
                   edgecolor=SURFACE, linewidth=1.5, label=group, zorder=2)
            bottom += vals
        ax.axhline(catalog["head"], ls=(0, (3, 2)), color=INK, lw=0.8)
        ax.axhline(catalog["head"] + catalog["middle"], ls=(0, (3, 2)), color=INK, lw=0.8)
        ax.set_xticks(range(len(order)), ["base" if o == "none" else f"+{o}" for o in order], rotation=40, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        _axes(fig, ax, base)
    for ax in axes.flat[len(bases):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, labelcolor=INK2, loc="upper right", ncol=3, title="item group",
               title_fontsize=8)
    fig.suptitle("Share of recommendation slots per item popularity group (dashed: catalogue shares, head 20 % / middle 30 % / tail 50 %)",
                 fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG_DIR, "rerank_item_exposure.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return "rerank_item_exposure.png"


def fig_user_groups(plt, user_groups, main, k):
    primary = f"NDCG@{k}"
    ug = user_groups[user_groups["base"] == main]
    order = ["none"] + [n for n in RERANKER_NAMES if n in set(ug["reranker"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE, sharey=True)
    for ax, grouping in zip(axes, ("activity", "mainstreaminess")):
        part = ug[ug["grouping"] == grouping]
        groups = [g for g in GROUP_ORDER.get(grouping, sorted(part["group"].unique())) if g in set(part["group"])]
        pivot = part.pivot_table(index="reranker", columns="group", values=primary).reindex(order)
        sizes = part.drop_duplicates("group").set_index("group")["n_users"]
        xs, width = np.arange(len(groups)), 0.8 / len(order)
        for j, name in enumerate(order):
            ax.bar(xs + (j - (len(order) - 1) / 2) * width, [pivot.loc[name, g] for g in groups], width * 0.92,
                   color=INK2 if name == "none" else RERANKER_COLOR[name], edgecolor=SURFACE, linewidth=1,
                   label="base list" if name == "none" else f"+{name}", zorder=2)
        ax.set_xticks(xs, [f"{g}\n(n = {sizes[g]})" for g in groups])
        _axes(fig, ax, f"{main}: {primary} per user {grouping} group", None, primary if grouping == "activity" else None)
    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK2)
    return _save(fig, f"rerank_user_groups_{main}.png")


def make_figures(sweeps, deltas, side, budget, user_groups, item_groups, configs, main, k, max_loss):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    targets = {n: c["target"] for n, c in configs[main]["rerankers"].items()}
    figures = []
    for name in RERANKER_NAMES:
        if name not in targets:
            continue
        selected_w = {b: c["rerankers"][name].get("inactive_weight") for b, c in configs.items() if name in c["rerankers"]}
        figures.append(fig_tradeoff(plt, sweeps, name, targets[name], k, max_loss, main, selected_w))
    figures.append(fig_relative_change(plt, side, deltas, targets, k))
    figures.append(fig_relative_change_main(plt, deltas, main, k))
    figures.append(fig_budget(plt, budget, targets, main, max_loss))
    if item_groups is not None:
        figures.append(fig_item_exposure(plt, item_groups))
        figures.append(fig_user_groups(plt, user_groups, main, k))
    return figures


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bases", nargs="*", help="base models (default: all with configs/rerankers/<base>.yaml)")
    parser.add_argument("--main-base", default="RecVAE", help="base model shown in detail")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--budgets", nargs="+", type=float, default=DEFAULT_BUDGETS, help="accepted NDCG losses to compare")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-groups", action="store_true", help="skip the user / item group tables")
    args = parser.parse_args()
    k, main_base = args.k, args.main_base

    comparison = pd.read_csv(os.path.join(EVAL_DIR, "rerank_test_comparison.csv"))
    bases = args.bases or [b for b in configured_bases() if b in set(comparison["base"])]
    if main_base not in bases:
        raise SystemExit(f"--main-base {main_base} is not among the evaluated bases: {bases}")
    comparison = comparison[comparison["base"].isin(bases)].reset_index(drop=True)
    configs = {b: load_rerank_config(b) for b in bases}
    max_loss = configs[main_base]["max_accuracy_loss"]
    targets = {n: c["target"] for n, c in configs[main_base]["rerankers"].items()}
    if any(c["k"] != k for c in configs.values()):
        raise SystemExit(f"the rerankers were tuned with another k than {k}")

    sweeps = load_sweeps(bases, configs)
    deltas = relative_deltas(comparison, k)
    side = side_effects(deltas, targets, k)
    budget = budget_table(sweeps, configs, args.budgets, k)
    sweeps.to_csv(os.path.join(EVAL_DIR, "rerank_sweeps.csv"), index=False)
    deltas.to_csv(os.path.join(EVAL_DIR, "rerank_test_deltas.csv"), index=False)
    side.to_csv(os.path.join(EVAL_DIR, "rerank_side_effects.csv"), index=False)
    budget.to_csv(os.path.join(EVAL_DIR, "rerank_budget_sensitivity.csv"), index=False)
    print("tables: sweeps, deltas, side effects, budget sensitivity", flush=True)

    evaluator = Evaluator.from_outputs("test")
    lists, per_user = {}, {}
    for base in bases:
        lists[(base, "none")] = load_recs(base, "test")
        for name in configs[base]["rerankers"]:
            lists[(base, name)] = load_recs(output_name(name, base), "test")
    for key, recs in lists.items():
        per_user[key] = evaluator.per_user(recs, k)[0]
    significance = significance_table(per_user, k)
    significance.to_csv(os.path.join(EVAL_DIR, "rerank_significance.csv"), index=False)
    print("tables: significance", flush=True)

    user_groups = item_groups = None
    if not args.no_groups:
        user_groups, item_groups = group_tables(evaluator, lists, k)
        user_groups.to_csv(os.path.join(EVAL_DIR, "rerank_user_group_analysis.csv"), index=False)
        item_groups.to_csv(os.path.join(EVAL_DIR, "rerank_item_group_analysis.csv"), index=False)
        print("tables: user and item groups", flush=True)

    figures = [] if args.no_figures else make_figures(sweeps, deltas, side, budget, user_groups, item_groups, configs,
                                                      main_base, k, max_loss)
    summary = summary_markdown(main_base, k, configs, comparison, deltas, side, budget, significance, user_groups,
                               item_groups, figures, max_loss)
    with open(os.path.join(EVAL_DIR, "rerank_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary)
    print(summary)
    written = ["sweeps", "test_deltas", "side_effects", "budget_sensitivity", "significance"]
    if not args.no_groups:
        written += ["user_group_analysis", "item_group_analysis"]
    print(f"Wrote {EVAL_DIR}/rerank_{{{','.join(written)}}}.csv and rerank_summary.md")
    if figures:
        print(f"Wrote {len(figures)} figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
