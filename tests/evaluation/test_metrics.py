"""Hand-computed checks for recbole/utils/evaluation.py.

Run from the repository root:  python -m unittest discover -s tests/evaluation -v
"""

import math
import unittest

import numpy as np
import pandas as pd

from recbole.utils.evaluation import (
    Evaluator,
    auc,
    average_precision_at_k,
    f1_at_k,
    gini,
    hit_at_k,
    intra_list_diversity,
    miscalibration,
    mrr_at_k,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
)

RANKED = ["a", "b", "c", "d"]   # hits at ranks 2 and 4
RELEVANT = {"b", "d", "x"}       # "x" is relevant but not recommended


class AccuracyTest(unittest.TestCase):
    def test_precision_recall_hit(self):
        self.assertAlmostEqual(precision_at_k(RANKED, RELEVANT, 4), 2 / 4)
        self.assertAlmostEqual(recall_at_k(RANKED, RELEVANT, 4), 2 / 3)
        self.assertEqual(hit_at_k(RANKED, RELEVANT, 1), 0.0)
        self.assertEqual(hit_at_k(RANKED, RELEVANT, 2), 1.0)

    def test_mrr_uses_first_hit(self):
        self.assertAlmostEqual(mrr_at_k(RANKED, RELEVANT, 4), 1 / 2)
        self.assertEqual(mrr_at_k(RANKED, RELEVANT, 1), 0.0)

    def test_ndcg(self):
        dcg = 1 / math.log2(3) + 1 / math.log2(5)
        idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)   # 3 relevant items, k = 4
        self.assertAlmostEqual(ndcg_at_k(RANKED, RELEVANT, 4), dcg / idcg)
        self.assertAlmostEqual(ndcg_at_k(["b", "d", "x"], RELEVANT, 3), 1.0)

    def test_only_top_k_counts(self):
        self.assertAlmostEqual(precision_at_k(RANKED, RELEVANT, 2), 1 / 2)
        self.assertAlmostEqual(recall_at_k(RANKED, RELEVANT, 2), 1 / 3)


class LectureExampleTest(unittest.TestCase):
    """The worked example of the evaluation lectures (Part 1 slides 36-37, Part 2 slides 9-14), n = 4."""

    G = {
        "u1": {"i2", "i3"},
        "u2": {"i1", "i2", "i3", "i5", "i7"},
        "u3": {"i3", "i4", "i6"},
        "u4": {"i1", "i2", "i3", "i4", "i5", "i6"},
    }
    R = {
        "u1": ["i1", "i3", "i4", "i6"],
        "u2": ["i2", "i3", "i4", "i5"],
        "u3": ["i1", "i2", "i5", "i7"],
        "u4": ["i3", "i4", "i5", "i6"],
    }

    def mean(self, fn, **kwargs):
        return sum(fn(self.R[u], self.G[u], 4, **kwargs) for u in self.G) / len(self.G)

    def test_precision_and_recall(self):
        self.assertAlmostEqual(self.mean(precision_at_k), 0.50)                       # slide 36: 50%
        self.assertAlmostEqual(self.mean(recall_at_k), (2/4 + 3/5 + 0 + 4/6) / 4)    # slide 37: ~44%
        self.assertAlmostEqual(round(self.mean(recall_at_k), 2), 0.44)

    def test_ndcg(self):
        self.assertAlmostEqual(round(ndcg_at_k(self.R["u1"], self.G["u1"], 4), 2), 0.39)  # slide 9
        self.assertAlmostEqual(round(ndcg_at_k(self.R["u2"], self.G["u2"], 4), 1), 0.8)   # slide 10

    def test_mrr(self):
        self.assertAlmostEqual(mrr_at_k(self.R["u1"], self.G["u1"], 4), 1 / 2)  # slide 12

    def test_average_precision(self):
        # slide 14: AP(u2) = 1/3 * (1 + 1 + 3/4) = 0.92
        self.assertAlmostEqual(average_precision_at_k(self.R["u2"], self.G["u2"], 4), (1 + 1 + 3 / 4) / 3)
        # RecBole divides by min(|G|, k) = 4 instead
        self.assertAlmostEqual(average_precision_at_k(self.R["u2"], self.G["u2"], 4, normalize="min"), (1 + 1 + 3 / 4) / 4)
        self.assertEqual(average_precision_at_k(self.R["u3"], self.G["u3"], 4), 0.0)  # no hits

    def test_f1(self):
        p, r = 3 / 4, 3 / 5  # u2
        self.assertAlmostEqual(f1_at_k(self.R["u2"], self.G["u2"], 4), 2 * p * r / (p + r))
        self.assertEqual(f1_at_k(self.R["u3"], self.G["u3"], 4), 0.0)


class AucTest(unittest.TestCase):
    def test_perfect_random_and_ties(self):
        pos = np.array([True, True, False, False])
        everything = np.ones(4, dtype=bool)
        self.assertAlmostEqual(auc(np.array([4.0, 3.0, 2.0, 1.0]), pos, everything), 1.0)
        self.assertAlmostEqual(auc(np.array([1.0, 2.0, 3.0, 4.0]), pos, everything), 0.0)
        self.assertAlmostEqual(auc(np.array([1.0, 1.0, 1.0, 1.0]), pos, everything), 0.5)  # ties count 1/2
        # 4 (pos, neg) pairs: 4>2, 4>1, 1<2, 1=1 -> (1 + 1 + 0 + 0.5) / 4
        self.assertAlmostEqual(auc(np.array([4.0, 1.0, 2.0, 1.0]), pos, everything), 2.5 / 4)

    def test_non_candidates_are_ignored(self):
        pos = np.array([True, False, False])
        only_first_two = np.array([True, True, False])  # the third item was already seen
        self.assertAlmostEqual(auc(np.array([2.0, 1.0, 9.0]), pos, only_first_two), 1.0)


GENRES = {"a": {"Comedy"}, "b": {"Comedy"}, "c": {"Drama"}, "d": {"Comedy", "Drama"}}


class BeyondAccuracyTest(unittest.TestCase):
    def test_intra_list_diversity(self):
        self.assertAlmostEqual(intra_list_diversity(["a", "b"], GENRES), 0.0)
        self.assertAlmostEqual(intra_list_diversity(["a", "c"], GENRES), 1.0)
        # pairs: a-c 1, a-d 1/2, c-d 1/2 -> mean 2/3
        self.assertAlmostEqual(intra_list_diversity(["a", "c", "d"], GENRES), 2 / 3)

    def test_miscalibration_zero_when_genre_mix_matches(self):
        self.assertAlmostEqual(miscalibration(["a", "c"], ["b", "c"], GENRES), 0.0)

    def test_miscalibration_grows_when_list_misses_a_genre(self):
        # history 50% Comedy / 50% Drama, list 100% Comedy
        p, alpha = 0.5, 0.01
        expected = p * math.log2(p / ((1 - alpha) * 1 + alpha * p)) + p * math.log2(p / (alpha * p))
        self.assertAlmostEqual(miscalibration(["a", "c"], ["a", "b"], GENRES), expected)

    def test_novelty(self):
        popularity = {"a": 3, "b": 0}   # 3 users in total
        self.assertAlmostEqual(novelty(["a"], popularity, 3), -math.log2(4 / 4))
        self.assertAlmostEqual(novelty(["b"], popularity, 3), -math.log2(1 / 4))

    def test_gini(self):
        self.assertAlmostEqual(gini([5, 5, 5, 5]), 0.0)
        self.assertAlmostEqual(gini([0, 0, 0, 10]), 0.75)   # (n - 1) / n for full concentration


class EvaluatorTest(unittest.TestCase):
    def setUp(self):
        # 5 users; item "a" is the most popular, so the short head (top 20% of 5 items) = {"a"}.
        train = pd.DataFrame({
            "user_id": ["u1", "u1", "u1", "u2", "u3", "u4", "u5"],
            "item_id": ["a", "b", "c", "a", "a", "d", "b"],
        })
        test = pd.DataFrame({"user_id": ["u1", "u2"], "item_id": ["d", "e"]})
        genres = {**GENRES, "e": {"Drama"}}
        self.ev = Evaluator(train, test, genres, catalog=["a", "b", "c", "d", "e"])
        self.recs = pd.DataFrame({
            "user_id": ["u1", "u1", "u2", "u2"],
            "item_id": ["d", "e", "b", "c"],
            "rank": [1, 2, 1, 2],
        })

    def test_groups(self):
        self.assertEqual(self.ev.short_head, {"a"})
        self.assertEqual(self.ev.active_users, {"u1"})   # u1 has the most training interactions

    def test_evaluate(self):
        result = self.ev.evaluate(self.recs, k=2)
        self.assertAlmostEqual(result["Hit@2"], 0.5)          # u1 hits, u2 does not
        self.assertAlmostEqual(result["Coverage@2"], 4 / 5)   # b, c, d, e recommended
        self.assertAlmostEqual(result["APLT@2"], 1.0)         # nothing from the head
        self.assertAlmostEqual(result["UserNDCGGap@2"], 1.0)  # active u1: NDCG 1, inactive u2: 0
        self.assertAlmostEqual(result["TailHitShare@2"], 1.0)


if __name__ == "__main__":
    unittest.main()
