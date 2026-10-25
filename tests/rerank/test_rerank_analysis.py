"""Table functions of run_rerank_analysis.py on hand-made frames (file-system free)."""

import sys
import unittest

import numpy as np
import pandas as pd

import run_rerank_analysis as ana

K = 10
METRICS = ["NDCG@10", "ILD@10", "APLT@10", "Miscalibration@10"]


def _row(base, reranker, lam, changed, ndcg, ild, aplt, miscal):
    return {"base": base, "reranker": reranker, "lambda": lam, "inactive_weight": np.nan, "exposure_target": None,
            "Changed@10": changed, "NDCG@10": ndcg, "ILD@10": ild, "APLT@10": aplt, "Miscalibration@10": miscal}


COMPARISON = pd.DataFrame([
    _row("A", "none", 0.0, 0.0, 0.20, 0.50, 0.10, 1.0),
    _row("A", "MMR", 0.3, 0.2, 0.19, 0.60, 0.10, 0.9),
    _row("A", "Popularity", 0.5, 0.1, 0.18, 0.50, 0.20, 1.1),
    _row("B", "none", 0.0, 0.0, 0.10, 0.40, 0.05, 2.0),
    _row("B", "MMR", 0.3, 0.3, 0.10, 0.50, 0.05, 2.0),
    _row("B", "Popularity", 0.0, 0.0, 0.10, 0.40, 0.05, 2.0),  # not applied
])
TARGETS = {"MMR": "ILD@10", "Popularity": "APLT@10"}
CONFIGS = {
    base: {"base_model": base, "k": K, "max_accuracy_loss": 0.05, "primary_metric": "NDCG@10",
           "rerankers": {"MMR": {"lambda": 0.3, "target": "ILD@10"},
                         "Popularity": {"lambda": 0.5 if base == "A" else 0.0, "target": "APLT@10"}}}
    for base in ("A", "B")
}


def _sweep(base, reranker, stage, lambdas, ndcg, target, target_values, selected):
    return pd.DataFrame({"base": base, "reranker": reranker, "stage": stage, "lambda": lambdas, "inactive_weight": np.nan,
                         "selected": selected, "target": target, "NDCG@10": ndcg, target: target_values})


SWEEPS = pd.concat([
    _sweep("A", "Popularity", "valid", [0.0, 0.5, 1.0], [0.20, 0.19, 0.15], "APLT@10", [0.10, 0.20, 0.40], [False, True, False]),
    _sweep("A", "Popularity", "test", [0.0, 0.5, 1.0], [0.20, 0.185, 0.14], "APLT@10", [0.10, 0.21, 0.41], [False, True, False]),
], ignore_index=True)


class ImportTest(unittest.TestCase):
    def test_matplotlib_is_lazy(self):
        self.assertNotIn("matplotlib.pyplot", sys.modules)


class DeltaTest(unittest.TestCase):
    def test_relative_deltas(self):
        deltas = ana.relative_deltas(COMPARISON, K)
        self.assertEqual(len(deltas), 4)
        a_mmr = deltas[(deltas["base"] == "A") & (deltas["reranker"] == "MMR")].iloc[0]
        self.assertAlmostEqual(a_mmr["NDCG@10_pct"], -5.0)
        self.assertAlmostEqual(a_mmr["ILD@10_pct"], 20.0)
        self.assertAlmostEqual(a_mmr["Miscalibration@10_pct"], -10.0)
        self.assertEqual(a_mmr["Changed@10"], 0.2)
        b_pop = deltas[(deltas["base"] == "B") & (deltas["reranker"] == "Popularity")].iloc[0]
        self.assertTrue((b_pop[[f"{m}_pct" for m in METRICS]] == 0).all())

    def test_side_effects(self):
        side = ana.side_effects(ana.relative_deltas(COMPARISON, K), TARGETS, K)
        mmr_ild = side[(side["reranker"] == "MMR") & (side["metric"] == "ILD@10")].iloc[0]
        self.assertAlmostEqual(mmr_ild["mean_pct"], 22.5)  # +20 % on A, +25 % on B
        self.assertAlmostEqual(mmr_ild["min_pct"], 20.0)
        self.assertTrue(mmr_ild["is_target"])
        self.assertEqual(mmr_ild["n_bases"], 2)
        mmr_ndcg = side[(side["reranker"] == "MMR") & (side["metric"] == "NDCG@10")].iloc[0]
        self.assertFalse(mmr_ndcg["is_target"])
        self.assertEqual(side["reranker"].iloc[0], "MMR")  # RERANKER_NAMES order, metrics in ACCURACY + BEYOND order
        self.assertEqual(side["metric"].iloc[0], "NDCG@10")


class BudgetTest(unittest.TestCase):
    def test_budget_table(self):
        table = ana.budget_table(SWEEPS, CONFIGS, [0.0, 0.05, 0.3], K)
        self.assertEqual(len(table), 3)
        self.assertEqual(table["lambda"].tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(table["admissible"].tolist(), [1, 2, 3])
        mid = table.iloc[1]
        self.assertAlmostEqual(mid["valid_target_gain_pct"], 100.0)
        self.assertAlmostEqual(mid["test_NDCG"], 0.185)
        self.assertAlmostEqual(mid["test_NDCG_loss_pct"], 7.5)
        self.assertAlmostEqual(mid["test_target_gain_pct"], 110.0)
        self.assertAlmostEqual(table.iloc[0]["test_target_gain_pct"], 0.0)


class SignificanceTest(unittest.TestCase):
    def _users(self, shift_ndcg=0.0, shift_miscal=0.0):
        rng = np.random.default_rng(0)
        base = pd.DataFrame({"user_id": [f"u{i}" for i in range(30)], "NDCG@10": rng.random(30),
                             "Miscalibration@10": 1 + rng.random(30)})
        reranked = base.assign(**{"NDCG@10": base["NDCG@10"] + shift_ndcg * (1 + rng.random(30)),
                                  "Miscalibration@10": base["Miscalibration@10"] + shift_miscal})
        return base, reranked

    def test_identical_lists(self):
        base, same = self._users()
        t = ana.paired_tests(base, same, ["NDCG@10", "Miscalibration@10"]).set_index("metric")
        self.assertEqual(t.loc["NDCG@10", "mean_diff"], 0.0)
        self.assertEqual(t.loc["NDCG@10", "wilcoxon_p"], 1.0)
        self.assertEqual(t.loc["NDCG@10", "share_unchanged"], 1.0)
        self.assertEqual(t.loc["NDCG@10", "n_users"], 30)

    def test_shift_is_detected_in_the_metric_direction(self):
        base, shifted = self._users(shift_ndcg=0.05, shift_miscal=-0.1)
        t = ana.paired_tests(base, shifted, ["NDCG@10", "Miscalibration@10"]).set_index("metric")
        self.assertEqual(t.loc["NDCG@10", "share_improved"], 1.0)
        self.assertLess(t.loc["NDCG@10", "wilcoxon_p"], 0.05)
        self.assertEqual(t.loc["Miscalibration@10", "share_improved"], 1.0)  # lower is better
        self.assertEqual(t.loc["Miscalibration@10", "share_worse"], 0.0)

    def test_holm(self):
        np.testing.assert_allclose(ana.holm([0.01, 0.04, 0.03, 0.5]), [0.04, 0.09, 0.09, 0.5])
        self.assertTrue((ana.holm([0.2, 0.9, 0.7]) <= 1).all())

    def test_significance_table_families(self):
        base, shifted = self._users(shift_ndcg=-0.05)
        base["ILD@10"] = shifted["ILD@10"] = 0.5
        base["APLT@10"] = shifted["APLT@10"] = 0.1
        per_user = {("A", "none"): base, ("A", "MMR"): shifted, ("A", "Popularity"): shifted}
        table = ana.significance_table(per_user, K)
        self.assertEqual(set(table["metric"]), {"NDCG@10", "ILD@10", "APLT@10"})
        ndcg = table[table["metric"] == "NDCG@10"]
        self.assertEqual(len(ndcg), 2)
        self.assertTrue((ndcg["wilcoxon_p_holm"] >= ndcg["wilcoxon_p"]).all())


class SummaryTest(unittest.TestCase):
    def test_sections(self):
        deltas = ana.relative_deltas(COMPARISON, K)
        side = ana.side_effects(deltas, TARGETS, K)
        budget = ana.budget_table(SWEEPS, CONFIGS, [0.0, 0.05, 0.3], K)
        users = pd.DataFrame({"user_id": [f"u{i}" for i in range(5)], "NDCG@10": np.linspace(0, 1, 5),
                              "ILD@10": 0.5, "APLT@10": 0.1})
        per_user = {(b, r): users for b in ("A", "B") for r in ("none", "MMR", "Popularity")}
        significance = ana.significance_table(per_user, K)
        text = ana.summary_markdown("A", K, CONFIGS, COMPARISON, deltas, side, budget, significance, None, None, [], 0.05)
        for n in range(1, 9):
            self.assertIn(f"## {n}.", text)
        self.assertIn("Not applied", text)  # B + Popularity has lambda = 0
        self.assertIn("B + Popularity", text)
        self.assertIn("+MMR (λ=0.3)", text)
        self.assertIn("(skipped: needs MixedHybrid", text)  # no order table given
        # report tables: both bases have a usable NDCG, so nothing is dropped; the target is bold; absolutes are given
        self.assertIn("### 3b. Report table (Task 3.2): mean (min … max) change in % over 2 bases\n", text)
        self.assertIn("| Changed@10 ", text)
        self.assertIn("**+22.5 (+20.0 … +25.0)**", text.split("### 3b.")[1].split("## 4.")[0])  # MMR's ILD: A +20%, B +25%
        self.assertIn("Absolute APLT@10 over these bases: base lists 0.050–0.100", text)
        self.assertIn("### 4b. Report table (Task 3.2)", text)


class _CountingEvaluator:
    """Stands in for Evaluator: NDCG = share of the top-k starting with 'x', ILD = length of the top-k."""

    def evaluate(self, recs, k):
        top = recs[recs["rank"] <= k]
        return {f"NDCG@{k}": float(top["item_id"].str.startswith("x").mean()), f"ILD@{k}": float(len(top))}


def _list(user_items):
    return pd.DataFrame([{"user_id": u, "item_id": i, "rank": r, "score": float(10 - r)}
                         for u, items in user_items.items() for r, i in enumerate(items, start=1)])


class GroupReportTest(unittest.TestCase):
    def test_user_group_deltas(self):
        def row(base, reranker, group, ndcg, ild):
            return {"base": base, "reranker": reranker, "grouping": "activity", "group": group, "NDCG@10": ndcg, "ILD@10": ild}
        ug = pd.DataFrame([row("A", "none", "low", 0.20, 0.5), row("A", "MMR", "low", 0.18, 0.6),
                           row("B", "none", "low", 0.10, 0.4), row("B", "MMR", "low", 0.10, 0.5),
                           row("C", "none", "low", 0.10, 0.4), row("C", "MMR", "low", 0.05, 0.9)])  # C is left out
        d = ana.user_group_deltas(ug, K, bases=["A", "B"]).set_index(["grouping", "group", "reranker"])
        self.assertEqual(len(d), 1)
        self.assertAlmostEqual(d.loc[("activity", "low", "MMR"), "NDCG@10_pct"], -5.0)  # mean of -10% and 0%
        self.assertAlmostEqual(d.loc[("activity", "low", "MMR"), "ILD@10_d"], 0.1)
        self.assertNotIn("APLT@10_d", d.columns)  # only the metrics present

    def test_item_group_means(self):
        ig = pd.DataFrame([{"base": b, "reranker": r, "grouping": "popularity", "group": "tail", "exposure_share": e,
                            "hit_share": 0.0, "recall@10": 0.0, "catalog_share": 0.5}
                           for b, r, e in (("A", "none", 0.0), ("A", "Fairness", 0.2), ("B", "none", 0.1), ("B", "Fairness", 0.4))])
        m = ana.item_group_means(ig).set_index(["reranker", "group"])
        self.assertAlmostEqual(m.loc[("Fairness", "tail"), "exposure_share"], 0.3)
        self.assertEqual(m.columns.tolist(), ["exposure_share", "hit_share", "recall@10"])


class OrderTest(unittest.TestCase):
    HYBRID = {"none": _list({"u": ["x1", "x2"]}), "MMR": _list({"u": ["x1", "z"]}), "Popularity": _list({"u": ["x1", "x2"]})}
    SOURCES = {("MMR", "X"): _list({"u": ["x1", "x3"]}), ("MMR", "Y"): _list({"u": ["y1", "x1"]})}  # none for Popularity

    def test_order_table(self):
        table = ana.order_table(_CountingEvaluator(), self.HYBRID, self.SOURCES, ["X", "Y"], {"X": 1, "Y": 1}, 2)
        self.assertEqual(table["order"].tolist(), ["base", "rerank → mix", "mix → rerank"])  # Popularity is skipped
        self.assertEqual(table["reranker"].tolist(), ["none", "MMR", "MMR"])
        # rerank → mix = x1 (from X), y1 (from Y); mix → rerank = x1, z: one of the two items is shared
        self.assertAlmostEqual(table.loc[1, "NDCG@2"], 0.5)
        self.assertAlmostEqual(table.loc[2, "NDCG@2"], 0.5)
        self.assertAlmostEqual(table.loc[1, "Shared@2"], 0.5)
        self.assertAlmostEqual(table.loc[2, "Shared@2"], 0.5)
        self.assertTrue(np.isnan(table.loc[0, "Shared@2"]))

    def test_summary_section(self):
        deltas = ana.relative_deltas(COMPARISON, K)
        side = ana.side_effects(deltas, TARGETS, K)
        budget = ana.budget_table(SWEEPS, CONFIGS, [0.0, 0.05, 0.3], K)
        users = pd.DataFrame({"user_id": [f"u{i}" for i in range(5)], "NDCG@10": np.linspace(0, 1, 5),
                              "ILD@10": 0.5, "APLT@10": 0.1})
        per_user = {(b, r): users for b in ("A", "B") for r in ("none", "MMR", "Popularity")}
        significance = ana.significance_table(per_user, K)
        order = ana.order_table(_CountingEvaluator(), self.HYBRID, self.SOURCES, ["X", "Y"], {"X": 1, "Y": 1}, K)
        order.attrs["hybrid"] = "MixedHybrid: X=1, Y=1"
        text = ana.summary_markdown("A", K, CONFIGS, COMPARISON, deltas, side, budget, significance, None, None, [],
                                    0.05, order)
        self.assertIn("## 7. Order of composition", text)
        self.assertIn("MixedHybrid: X=1, Y=1", text)
        self.assertIn("| +MMR, rerank → mix", text)
        self.assertIn("| +MMR, mix → rerank", text)
        self.assertIn("**", text.split("## 7.")[1].split("## 8.")[0])  # the target (ILD) is bold
        self.assertIn("## 8. Figures", text)


if __name__ == "__main__":
    unittest.main()
