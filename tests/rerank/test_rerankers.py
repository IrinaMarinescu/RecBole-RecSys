import unittest

import numpy as np
import pandas as pd

from recbole.model.general_recommender.rerankers import (
    Calibration,
    Fairness,
    MMR,
    PopularityDiscount,
    Reranker,
    _relevance,
    base_relevance_frame,
    budget_sensitivity,
    build_rerankers,
    genre_similarity,
    output_name,
    rerank,
    select_lambda,
)
from recbole.utils.evaluation import intra_list_diversity, miscalibration

GENRES = {
    "a1": {"Action"}, "a2": {"Action"}, "a3": {"Action", "Thriller"},
    "c1": {"Comedy"}, "c2": {"Comedy", "Romance"},
    "d1": {"Drama"},
}


def _recs(lists):
    """{user: [(item, score), ...]} in rank order -> frame like outputs/<model>/recs_<stage>.tsv."""
    rows = [
        {"user_id": str(user), "item_id": item, "rank": rank, "score": float(score)}
        for user, items in lists.items()
        for rank, (item, score) in enumerate(items, start=1)
    ]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rank", "score"])


def _items(frame, user="u"):
    return frame[frame["user_id"] == user].sort_values("rank")["item_id"].tolist()


class ConstantObjective(Reranker):
    name = "Constant"

    def objective(self, user_id, item_id, chosen):
        return 0.5


class RelevanceTest(unittest.TestCase):
    def test_min_max(self):
        self.assertEqual(_relevance([3.0, 1.0, 2.0]).tolist(), [1.0, 0.0, 0.5])

    def test_constant_and_nan(self):
        self.assertEqual(_relevance([2.0, 2.0]).tolist(), [1.0, 1.0])
        self.assertEqual(_relevance([float("nan")] * 3).tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(_relevance([2.0, float("nan"), 0.0]).tolist(), [1.0, 0.0, 0.0])


class GreedyTest(unittest.TestCase):
    def test_lambda_zero_is_the_base_list(self):
        frame = _recs({"u": [("a1", 0.9), ("c1", 0.8), ("a2", 0.7), ("d1", 0.1)]})
        out = rerank(frame, MMR(GENRES), lam=0.0, top_k=3)
        self.assertEqual(_items(out), ["a1", "c1", "a2"])
        self.assertEqual(out["base_rank"].tolist(), [1, 2, 3])
        self.assertEqual(out["score"].tolist(), [0.9, 0.8, 0.7])

    def test_constant_objective_keeps_base_order_and_breaks_ties_by_rank(self):
        frame = _recs({"u": [("a1", 1.0), ("c1", 1.0), ("a2", 1.0)]})  # all scores equal
        out = rerank(frame, ConstantObjective(), lam=0.5, top_k=3)
        self.assertEqual(_items(out), ["a1", "c1", "a2"])

    def test_fewer_candidates_than_top_k(self):
        frame = _recs({"u": [("a1", 1.0), ("c1", 0.5)]})
        out = rerank(frame, MMR(GENRES), lam=0.7, top_k=10)
        self.assertEqual(len(out), 2)
        self.assertEqual(out["rank"].tolist(), [1, 2])

    def test_users_are_independent_and_columns_fixed(self):
        frame = _recs({"1": [("a1", 1.0), ("a2", 0.9), ("c1", 0.8)], "2": [("c1", 1.0), ("c2", 0.9), ("a1", 0.8)]})
        out = rerank(frame, MMR(GENRES), lam=1.0, top_k=2)
        self.assertEqual(list(out.columns), ["user_id", "item_id", "rank", "score", "base_rank"])
        self.assertEqual(_items(out, "1"), ["a1", "c1"])
        self.assertEqual(_items(out, "2"), ["c1", "a1"])

    def test_invalid_lambda(self):
        with self.assertRaises(ValueError):
            rerank(_recs({"u": [("a1", 1.0)]}), MMR(GENRES), lam=1.5)


class MMRTest(unittest.TestCase):
    def test_similarity_is_one_minus_evaluation_distance(self):
        self.assertAlmostEqual(genre_similarity({"Action", "Thriller"}, {"Action"}), 0.5)
        self.assertEqual(genre_similarity(set(), set()), 1.0)
        for a, b in (("a1", "a3"), ("a1", "c1"), ("c1", "c2")):
            self.assertAlmostEqual(1 - genre_similarity(GENRES[a], GENRES[b]), intra_list_diversity([a, b], GENRES))

    def test_pure_diversity_picks_different_genres(self):
        frame = _recs({"u": [("a1", 1.0), ("a2", 0.9), ("a3", 0.8), ("c1", 0.7), ("d1", 0.6)]})
        out = rerank(frame, MMR(GENRES), lam=1.0, top_k=3)
        # first pick by relevance (a1); then items that share no genre with the chosen ones, in rank order
        self.assertEqual(_items(out), ["a1", "c1", "d1"])
        self.assertGreater(intra_list_diversity(_items(out), GENRES), intra_list_diversity(["a1", "a2", "a3"], GENRES))

    def test_small_lambda_only_reorders_near_ties(self):
        frame = _recs({"u": [("a1", 1.0), ("a2", 0.99), ("c1", 0.98), ("d1", 0.0)]})
        self.assertEqual(_items(rerank(frame, MMR(GENRES), lam=0.3, top_k=3)), ["a1", "c1", "a2"])


class CalibrationTest(unittest.TestCase):
    HISTORY = {"u": ["c1", "c1", "c2", "a1"]}  # mostly comedy

    def test_objective_is_minus_evaluation_miscalibration(self):
        reranker = Calibration(GENRES, self.HISTORY)
        for chosen, item in (([], "a1"), (["a1"], "c1"), (["a1", "c2"], "d1")):
            self.assertAlmostEqual(
                reranker.objective("u", item, chosen), -miscalibration(self.HISTORY["u"], chosen + [item], GENRES)
            )

    def test_pure_calibration_matches_the_history(self):
        frame = _recs({"u": [("a1", 1.0), ("a2", 0.9), ("a3", 0.8), ("c1", 0.2), ("c2", 0.1)]})
        base = _items(rerank(frame, Calibration(GENRES, self.HISTORY), lam=0.0, top_k=3))
        calibrated = _items(rerank(frame, Calibration(GENRES, self.HISTORY), lam=1.0, top_k=3))
        # hand-computed: c2 (Comedy+Romance) has the lowest KL alone, then a1 restores the Action share, then c1
        self.assertEqual(calibrated, ["c2", "a1", "c1"])
        self.assertLess(miscalibration(self.HISTORY["u"], calibrated, GENRES), miscalibration(self.HISTORY["u"], base, GENRES))

    def test_user_without_history_is_untouched(self):
        frame = _recs({"new": [("a1", 1.0), ("c1", 0.5)]})
        self.assertEqual(_items(rerank(frame, Calibration(GENRES, self.HISTORY), lam=1.0), "new"), ["a1", "c1"])


class PopularityTest(unittest.TestCase):
    POP = {"a1": 99, "a2": 9, "c1": 0}

    def test_objective_range(self):
        reranker = PopularityDiscount(self.POP)
        self.assertAlmostEqual(reranker.objective("u", "a1", []), 0.0)
        self.assertAlmostEqual(reranker.objective("u", "a2", []), 0.5)
        self.assertAlmostEqual(reranker.objective("u", "c1", []), 1.0)
        self.assertAlmostEqual(reranker.objective("u", "unknown", []), 1.0)

    def test_pure_popularity_discount_puts_the_tail_first(self):
        frame = _recs({"u": [("a1", 1.0), ("a2", 0.9), ("c1", 0.8)]})
        self.assertEqual(_items(rerank(frame, PopularityDiscount(self.POP), lam=1.0)), ["c1", "a2", "a1"])


class FairnessTest(unittest.TestCase):
    HEAD = {"h1", "h2", "h3", "h4"}   # short head; t1, t2 are long-tail items
    ACTIVE = {"a"}

    def _fair(self, **kwargs):
        return Fairness(self.HEAD, self.ACTIVE, **kwargs)

    def test_user_side_lambda_scale(self):
        fair = self._fair(inactive_weight=0.25)
        self.assertEqual(fair.lambda_scale("a"), 1.0)
        self.assertEqual(fair.lambda_scale("b"), 0.25)

    def test_exposure_targets(self):
        items, relevance = ["h1", "t1", "h2"], np.array([1.0, 0.5, 0.0])
        self.assertEqual(self._fair(exposure_target="parity").tail_share(items, relevance), 0.5)
        self.assertAlmostEqual(self._fair(exposure_target="equity").tail_share(items, relevance), 0.5 / 1.5)
        with self.assertRaises(ValueError):
            self._fair(exposure_target="bogus").tail_share(items, relevance)

    def test_objective_is_the_unfilled_share_of_the_quota(self):
        fair = self._fair(exposure_target="parity")
        fair.begin_user("a", ["h1", "h2", "t1", "t2"], np.array([1.0, 0.9, 0.2, 0.1]), top_k=4)  # quota 2 + 2
        self.assertEqual(fair.objective("a", "t1", []), 1.0)
        self.assertEqual(fair.objective("a", "t2", ["t1"]), 0.5)
        self.assertEqual(fair.objective("a", "t2", ["t1", "h1", "t2"]), 0.0)
        self.assertEqual(fair.objective("a", "h2", ["h1"]), 0.5)

    def test_lambda_one_enforces_the_quota(self):
        frame = _recs({"a": [("h1", 1.0), ("h2", 0.9), ("h3", 0.8), ("h4", 0.7), ("t1", 0.2), ("t2", 0.1)]})
        out = _items(rerank(frame, self._fair(exposure_target="parity"), lam=1.0, top_k=4), "a")
        # relevance breaks the first tie (h1); then the group with the larger unfilled quota goes first
        self.assertEqual(out, ["h1", "t1", "h2", "t2"])
        self.assertEqual(sum(i in self.HEAD for i in out), 2)

    def test_small_lambda_keeps_relevance_in_charge(self):
        frame = _recs({"a": [("h1", 1.0), ("h2", 0.9), ("h3", 0.8), ("h4", 0.7), ("t1", 0.2), ("t2", 0.1)]})
        self.assertEqual(_items(rerank(frame, self._fair(exposure_target="parity"), lam=0.2, top_k=4), "a"),
                         ["h1", "h2", "h3", "h4"])

    def test_inactive_users_are_protected(self):
        lists = [("h1", 1.0), ("h2", 0.9), ("h3", 0.8), ("t1", 0.2)]
        frame = _recs({"a": lists, "b": lists})
        protected = rerank(frame, self._fair(exposure_target="parity", inactive_weight=0.0), lam=1.0, top_k=3)
        self.assertEqual(_items(protected, "a"), ["h1", "t1", "h2"])  # active user pays
        self.assertEqual(_items(protected, "b"), ["h1", "h2", "h3"])  # inactive user keeps the base list
        equal = rerank(frame, self._fair(exposure_target="parity", inactive_weight=1.0), lam=1.0, top_k=3)
        self.assertEqual(_items(equal, "b"), ["h1", "t1", "h2"])

    def test_equity_without_tail_candidates_is_the_base_list(self):
        frame = _recs({"a": [("h1", 1.0), ("h2", 0.5), ("h3", 0.0)]})
        self.assertEqual(_items(rerank(frame, self._fair(), lam=1.0, top_k=3), "a"), ["h1", "h2", "h3"])

    def test_equity_quota_follows_the_candidates(self):
        # tail relevance share = 1 / (1 + 1 + 0 + 1) -> quota 1 of 4 for the tail, so one tail item enters
        frame = _recs({"a": [("h1", 1.0), ("h2", 1.0), ("t1", 1.0), ("h3", 0.0)]})
        out = _items(rerank(frame, self._fair(exposure_target="equity"), lam=1.0, top_k=3), "a")
        self.assertEqual(sum(i not in self.HEAD for i in out), 1)


class BuildTest(unittest.TestCase):
    def test_side_data_from_training_split(self):
        train = pd.DataFrame({"user_id": ["u", "u", "v"], "item_id": ["a1", "a1", "c1"]})
        rerankers = build_rerankers(train, GENRES, short_head={"a1"}, active_users={"u"})
        self.assertEqual(set(rerankers), {"MMR", "Calibration", "Fairness", "Popularity"})
        self.assertEqual(rerankers["Fairness"].lambda_scale("u"), 1.0)
        self.assertAlmostEqual(rerankers["Popularity"].objective("u", "a1", []), 0.0)  # most popular (2 interactions)
        self.assertEqual(rerankers["Calibration"].profile["u"], {"Action": 1.0})


class SelectLambdaTest(unittest.TestCase):
    TABLE = pd.DataFrame({
        "lambda":  [0.0, 0.2, 0.4, 0.6, 0.8],
        "NDCG@10": [0.250, 0.245, 0.240, 0.230, 0.200],
        "ILD@10":  [0.50, 0.55, 0.60, 0.70, 0.90],
        "Miscalibration@10": [0.30, 0.25, 0.20, 0.10, 0.05],
    })

    def test_best_target_within_accuracy_tolerance(self):
        self.assertEqual(select_lambda(self.TABLE, "ILD@10", True, max_loss=0.05)["lambda"], 0.4)        # 0.2375 floor
        self.assertEqual(select_lambda(self.TABLE, "Miscalibration@10", False, max_loss=0.1)["lambda"], 0.6)
        self.assertEqual(select_lambda(self.TABLE, "ILD@10", True, max_loss=0.0)["lambda"], 0.0)

    def test_ties_go_to_the_smallest_lambda(self):
        table = self.TABLE.assign(**{"ILD@10": 0.5})
        self.assertEqual(select_lambda(table, "ILD@10", True)["lambda"], 0.0)

    def test_no_worse_constraint_and_inactive_weight_tie_break(self):
        table = pd.DataFrame({
            "lambda":          [0.0, 0.5, 0.5, 0.5, 1.0],
            "inactive_weight": [1.0, 0.0, 0.5, 1.0, 0.0],
            "NDCG@10":         [0.25, 0.245, 0.244, 0.243, 0.20],
            "TailExposure@10": [0.05, 0.08, 0.10, 0.10, 0.30],
            "UserNDCGGap@10":  [0.18, 0.15, 0.17, 0.19, 0.01],
        })
        best = select_lambda(table, "TailExposure@10", True, max_loss=0.05, no_worse=[("UserNDCGGap@10", False)])
        # lambda 1 fails the accuracy budget; (0.5, 1.0) widens the gap; (0.5, 0.5) has the best exposure left
        self.assertEqual((best["lambda"], best["inactive_weight"]), (0.5, 0.5))
        tie = table.assign(**{"UserNDCGGap@10": 0.1})
        best = select_lambda(tie, "TailExposure@10", True, max_loss=0.05, no_worse=[("UserNDCGGap@10", False)])
        self.assertEqual((best["lambda"], best["inactive_weight"]), (0.5, 1.0))  # equal target -> largest weight

    def test_needs_reference_row(self):
        with self.assertRaises(ValueError):
            select_lambda(self.TABLE[self.TABLE["lambda"] > 0], "ILD@10", True)

    def test_lambda_zero_is_always_admissible(self):
        # every lambda > 0 breaks the budget or widens the gap: the base list is selected, no error
        table = pd.DataFrame({
            "lambda":          [0.0, 0.5, 1.0],
            "inactive_weight": [1.0, 1.0, 0.0],
            "NDCG@10":         [0.25, 0.245, 0.10],
            "TailExposure@10": [0.05, 0.10, 0.30],
            "UserNDCGGap@10":  [0.18, 0.20, 0.01],
        })
        best = select_lambda(table, "TailExposure@10", True, max_loss=0.05, no_worse=[("UserNDCGGap@10", False)])
        self.assertEqual(best["lambda"], 0.0)

    def test_nan_reference_raises(self):
        table = self.TABLE.copy()
        table.loc[0, "NDCG@10"] = float("nan")
        with self.assertRaises(ValueError):
            select_lambda(table, "ILD@10", True)


class BudgetSensitivityTest(unittest.TestCase):
    def test_budgets_buy_more_target(self):
        sens = budget_sensitivity(SelectLambdaTest.TABLE, "ILD@10", True, [0.0, 0.02, 0.05, 0.1])
        self.assertEqual(sens["lambda"].tolist(), [0.0, 0.2, 0.4, 0.6])  # floors 0.25, 0.245, 0.2375, 0.225
        self.assertEqual(sens["admissible"].tolist(), [1, 2, 3, 4])
        self.assertEqual(sens["target_gain_pct"].iloc[0], 0.0)
        self.assertTrue((sens["ILD@10"].diff().dropna() >= 0).all())
        self.assertAlmostEqual(sens["target_gain_pct"].iloc[-1], 40.0)  # 0.70 vs 0.50
        self.assertTrue(sens["inactive_weight"].isna().all())

    def test_lower_is_better_gain_is_positive(self):
        sens = budget_sensitivity(SelectLambdaTest.TABLE, "Miscalibration@10", False, [0.1])
        self.assertEqual(sens["lambda"].iloc[0], 0.6)
        self.assertAlmostEqual(sens["target_gain_pct"].iloc[0], 100 * (0.30 - 0.10) / 0.30)


class OutputLayoutTest(unittest.TestCase):
    def test_output_name(self):
        self.assertEqual(output_name("MMR"), "Rerank_MMR")
        self.assertEqual(output_name("MMR", "RecVAE"), "Rerank_MMR/RecVAE")

    def test_base_relevance_frame(self):
        frame = _recs({"u": [("a1", 0.9), ("c1", 0.8)]})
        self.assertIs(base_relevance_frame(frame), frame)
        mixed = frame.assign(source=["BPR", "RecVAE"], score=[3.0, 0.9])  # scores from different models
        self.assertEqual(base_relevance_frame(mixed)["score"].tolist(), [-1.0, -2.0])
        self.assertEqual(_items(rerank(base_relevance_frame(mixed), ConstantObjective(), 0.5, 2)), ["a1", "c1"])


if __name__ == "__main__":
    unittest.main()
