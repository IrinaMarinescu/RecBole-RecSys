"""Tests for the score-matrix Hybrid Regression meta-learner and its selection methods."""

import unittest

import numpy as np
import pandas as pd

from recbole.model.general_recommender.hybrid_regression import (
    align_scores,
    combine_scores,
    evaluate_ranking,
    fit_ranking_weights,
    sample_pairs,
    top_k_items,
)
from recbole.model.general_recommender.mixed_hybrid import evaluate_recommendations
from tune_hybrid_regression import (
    best_single_index,
    exhaustive_index,
    greedy_index,
    one_per_kind_index,
    select_index,
)


class TestAlignment(unittest.TestCase):
    def test_extra_items_are_dropped_and_reordered(self):
        scores = np.arange(12, dtype=np.float32).reshape(3, 4)
        aligned = align_scores(scores, ["u2", "u0", "u1"], ["b", "a", "c", "extra"], ["u0", "u1"], ["a", "b"])
        # u0 row was index 1: items a=5, b=4. u1 row was index 2: items a=9, b=8.
        np.testing.assert_array_equal(aligned, np.array([[5, 4], [9, 8]], dtype=np.float32))

    def test_missing_reference_id_raises(self):
        with self.assertRaises(RuntimeError):
            align_scores(np.zeros((1, 1)), ["u"], ["a"], ["u"], ["b"])


class TestSampling(unittest.TestCase):
    def test_negatives_avoid_forbidden_and_are_deterministic(self):
        forbidden = np.zeros((4, 12), dtype=bool)
        forbidden[:, :3] = True
        pos_users = np.array([0, 1, 2, 3, 0], dtype=np.int32)
        pos_items = np.array([4, 5, 6, 7, 8], dtype=np.int32)
        first = sample_pairs(pos_users, pos_items, forbidden, 3, 12, np.random.default_rng(2020))
        second = sample_pairs(pos_users, pos_items, forbidden, 3, 12, np.random.default_rng(2020))
        for users, items, labels in (first, second):
            negatives = labels == 0
            self.assertEqual(int(labels.sum()), len(pos_users))
            self.assertFalse(forbidden[users[negatives], items[negatives]].any())
            self.assertEqual(len(labels), len(pos_users) * 4)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


class TestRanking(unittest.TestCase):
    def test_positive_weight_preserves_single_model_order(self):
        raw = np.arange(12 * 25, dtype=np.float64).reshape(12, 25)
        rng = np.random.default_rng(1)
        users = rng.integers(0, 12, 800)
        items = rng.integers(0, 25, 800)
        values = raw[users, items]
        labels = (values > np.median(values)).astype(np.int8)
        weight = fit_ranking_weights(values.reshape(-1, 1), labels, C=1.0, penalty="l2")
        self.assertGreater(weight[0], 0.0)
        seen = np.zeros(raw.shape, dtype=bool)
        combined = combine_scores(raw[None].astype(np.float32), np.array([weight[0]], dtype=np.float32))
        np.testing.assert_array_equal(top_k_items(raw, seen, 5)[0], top_k_items(combined, seen, 5)[0])

    def test_constant_shift_does_not_change_order(self):
        rng = np.random.default_rng(2)
        scores = rng.normal(size=(2, 8, 15)).astype(np.float32)
        weights = np.array([0.3, -0.2], dtype=np.float32)
        shifted = scores + np.array([5.0, -2.0], dtype=np.float32)[:, None, None]
        seen = np.zeros(scores.shape[1:], dtype=bool)
        left = top_k_items(combine_scores(scores, weights), seen, 4)[0]
        right = top_k_items(combine_scores(shifted, weights), seen, 4)[0]
        np.testing.assert_array_equal(left, right)

    def test_metrics_match_list_evaluation(self):
        rng = np.random.default_rng(3)
        n_users, n_items, k = 8, 30, 10
        combined = rng.normal(size=(n_users, n_items))
        seen = rng.random((n_users, n_items)) < 0.2
        relevant = rng.random((n_users, n_items)) < 0.15
        relevant[seen] = False
        for user in range(n_users):
            if not relevant[user].any():
                relevant[user, np.flatnonzero(~seen[user])[0]] = True
        metrics, _ = evaluate_ranking(combined, seen, relevant, k=k)
        chosen, chosen_scores = top_k_items(combined, seen, k)
        rows = []
        for user in range(n_users):
            for rank, (item, score) in enumerate(zip(chosen[user], chosen_scores[user]), start=1):
                rows.append({"user_id": str(user), "item_id": str(item), "rank": rank, "score": float(score)})
        truth_users, truth_items = np.nonzero(relevant)
        truth = pd.DataFrame({"user_id": truth_users.astype(str), "item_id": truth_items.astype(str)})
        expected = evaluate_recommendations(pd.DataFrame(rows), truth, k=k)
        for name, value in expected.items():
            self.assertAlmostEqual(metrics[name], value, places=10, msg=name)


class TestSelection(unittest.TestCase):
    def _toy(self):
        # masks: A, A (worse), B, C, A+B, A+C, B+C, A+B+C
        masks = np.array([1, 1, 2, 4, 3, 5, 6, 7], dtype=np.int16)
        scores = np.array([0.50, 0.10, 0.40, 0.30, 0.55, 0.45, 0.42, 0.54])
        names = ["A", "B", "C"]
        kinds = {"A": "latent factor", "B": "latent factor", "C": "content"}
        standalone = np.array([0.50, 0.40, 0.30])
        return scores, masks, names, kinds, standalone

    def test_best_single_and_exhaustive(self):
        scores, masks, _, _, _ = self._toy()
        self.assertEqual(best_single_index(scores, masks), 0)
        self.assertEqual(exhaustive_index(scores), 4)
        self.assertEqual(exhaustive_index(scores, masks), 4)

    def test_greedy_stops_when_the_next_model_hurts(self):
        scores, masks, _, _, _ = self._toy()
        chosen, path = greedy_index(scores, masks, 3)
        self.assertEqual(chosen, 4)
        self.assertEqual(path, [0, 4])

    def test_greedy_starts_from_the_best_single_model(self):
        scores, masks, _, _, _ = self._toy()
        keep = (masks & (masks - 1)) != 0
        chosen, path = greedy_index(
            scores[keep], masks[keep], 3, start_support=1, start_score=0.50
        )
        self.assertEqual(int(masks[keep][chosen]), 3)
        self.assertEqual([int(masks[keep][i]) for i in path], [3])

    def test_one_per_kind_uses_only_the_best_of_each_kind(self):
        scores, masks, names, kinds, standalone = self._toy()
        # Best latent is A, best content is C. A alone is not a hybrid, so A+C is chosen.
        self.assertEqual(one_per_kind_index(scores, masks, names, kinds, standalone), 5)
        scores = scores.copy()
        scores[5] = 0.70
        self.assertEqual(one_per_kind_index(scores, masks, names, kinds, standalone), 5)

    def test_one_per_kind_fold_uses_the_selection_half(self):
        scores, masks, names, kinds, standalone = self._toy()
        # Full validation prefers A. This selection half prefers B, so B+C is the mix.
        half = np.array([0.10, 0.90, 0.30])
        chosen = select_index(
            "OnePerKind", scores, masks, names, kinds, standalone, single_scores=half
        )
        self.assertEqual(chosen, 6)

    def test_tie_keeps_the_earlier_row(self):
        scores = np.array([0.2, 0.2])
        masks = np.array([1, 2])
        self.assertEqual(best_single_index(scores, masks), 0)


if __name__ == "__main__":
    unittest.main()
