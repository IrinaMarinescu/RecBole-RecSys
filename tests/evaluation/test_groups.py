"""Hand-computed checks for the group analysis in recbole/utils/evaluation.py (Task 2.5).
"""

import math
import unittest

import pandas as pd

from recbole.utils.evaluation import Evaluator

# User u<n> rated the first n items of a..f, so user activity is 1..6 and item popularity is
# a: 6, b: 5, c: 4, d: 3, e: 2, f: 1, g-j: 0.
ITEMS = list("abcdefghij")
TRAIN = pd.DataFrame(
    [(f"u{n}", item) for n in range(1, 7) for item in ITEMS[:n]], columns=["user_id", "item_id"]
)
TEST = pd.DataFrame({"user_id": ["u1", "u1", "u2"], "item_id": ["c", "g", "h"]})
GENRES = {i: {"Comedy"} for i in ITEMS} | {"c": {"Comedy", "Drama"}}
DEMOGRAPHICS = pd.DataFrame({
    "user_id": [f"u{n}" for n in range(1, 7)],
    "age": [24, 25, 34, 35, 44, 45],
    "gender": ["F", "M", "F", "M", "F", "M"],
    "occupation": ["student"] * 6,
})
RECS = pd.DataFrame({"user_id": ["u1", "u1", "u2", "u2"], "item_id": ["c", "h", "h", "a"], "rank": [1, 2, 1, 2]})


class GroupingTest(unittest.TestCase):
    def setUp(self):
        self.ev = Evaluator(TRAIN, TEST, GENRES, ITEMS)

    def test_activity_and_mainstreaminess_tertiles(self):
        g = self.ev.user_groupings()
        self.assertEqual(g["activity"], {"u1": "low", "u2": "low", "u3": "medium", "u4": "medium", "u5": "high", "u6": "high"})
        # mean popularity of the history: u1 6.0 ... u6 3.5, so heavy users here are the niche ones
        self.assertEqual(g["mainstreaminess"]["u6"], "niche")
        self.assertEqual(g["mainstreaminess"]["u1"], "mainstream")

    def test_demographics(self):
        g = self.ev.user_groupings(DEMOGRAPHICS)
        self.assertEqual([g["age"][f"u{n}"] for n in range(1, 7)], ["<25", "25-34", "25-34", "35-44", "35-44", "45+"])
        self.assertEqual(g["gender"]["u2"], "M")

    def test_popularity_groups(self):
        pop = self.ev.item_groupings()["popularity"]
        self.assertEqual({i for i, g in pop.items() if g == ["head"]}, {"a", "b"})         # top 20% of 10
        self.assertEqual({i for i, g in pop.items() if g == ["middle"]}, {"c", "d", "e"})
        self.assertEqual({i for i, g in pop.items() if g == ["tail"]}, {"f", "g", "h", "i", "j"})  # bottom 50%

    def test_item_can_be_in_several_genres(self):
        self.assertEqual(self.ev.item_groupings()["genre"]["c"], ["Comedy", "Drama"])


class GroupTableTest(unittest.TestCase):
    def setUp(self):
        self.ev = Evaluator(TRAIN, TEST, GENRES, ITEMS)

    def test_item_group_table(self):
        # slots: c (middle), h (tail) for u1; h (tail), a (head) for u2. Hits: c (u1), h (u2).
        # relevant: c (middle), g (tail), h (tail)
        table = self.ev.item_group_table(RECS, k=2).set_index(["grouping", "group"]).loc["popularity"]
        self.assertAlmostEqual(table.loc["head", "exposure_share"], 1 / 4)
        self.assertAlmostEqual(table.loc["tail", "exposure_share"], 2 / 4)
        self.assertAlmostEqual(table.loc["tail", "relevant_share"], 2 / 3)
        self.assertAlmostEqual(table.loc["tail", "hit_share"], 1 / 2)
        self.assertAlmostEqual(table.loc["middle", "recall@2"], 1.0)
        self.assertAlmostEqual(table.loc["tail", "recall@2"], 1 / 2)
        self.assertTrue(math.isnan(table.loc["head", "recall@2"]))   # no relevant head items
        self.assertAlmostEqual(table.loc["tail", "coverage@2"], 1 / 5)
        

    def test_user_group_table(self):
        table = self.ev.user_group_table(RECS, k=2).set_index(["grouping", "group"])
        low = table.loc[("activity", "low")]                         # only u1 and u2 have test items
        self.assertEqual(low["n_users"], 2)
        self.assertAlmostEqual(low["avg_test_items"], (2 + 1) / 2)
        self.assertAlmostEqual(low["Hit@2"], 1.0)                    # both users get a hit
        self.assertAlmostEqual(low["avg_train_items"], (1 + 2) / 2)  # u1 rated 1 item, u2 rated 2

    def test_ndcg_standard_error(self):
        # u1: hit at rank 1 of 2 relevant -> 1 / (1 + 1/log2(3)); u2: its only relevant item at rank 1 -> 1
        ndcg_u1, ndcg_u2 = 1 / (1 + 1 / math.log2(3)), 1.0
        low = self.ev.user_group_table(RECS, k=2).set_index(["grouping", "group"]).loc[("activity", "low")]
        # standard error of the mean of two values = |a - b| / 2
        self.assertAlmostEqual(low["NDCG@2_se"], abs(ndcg_u1 - ndcg_u2) / 2)

    def test_unlabelled_users_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "1 evaluated users have no label for grouping 'custom'"):
            self.ev.user_group_table(RECS, k=2, groupings={"custom": {"u1": "x"}})   # u2 has no label

    def test_empty_recommendations(self):
        empty = RECS.iloc[0:0]
        table = self.ev.item_group_table(empty, k=2).set_index(["grouping", "group"]).loc["popularity"]
        self.assertEqual(table.loc["head", "exposure_share"], 0.0)
        self.assertEqual(table.loc["tail", "recall@2"], 0.0)


if __name__ == "__main__":
    unittest.main()
