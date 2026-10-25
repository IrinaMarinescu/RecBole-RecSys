"""Exporting longer hybrid lists (candidates for the rerankers) must not change their top-10.

mix_recommendations stops when top_k items are chosen, and top_k_items sorts a partition, so the first
k items of a longer list are the k items of the shorter one.
"""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from recbole.model.general_recommender.hybrid_regression import top_k_items
from recbole.model.general_recommender.mixed_hybrid import mix_recommendations


def _recs(lists):
    rows = [
        {"user_id": user, "item_id": item, "rank": rank, "score": float(len(items) - rank + 1)}
        for user, items in lists.items()
        for rank, item in enumerate(items, start=1)
    ]
    return pd.DataFrame(rows)


class MixedHybridDepthTest(unittest.TestCase):
    def test_longer_mix_keeps_the_top_k_prefix(self):
        rng = np.random.default_rng(0)
        items = [f"i{j}" for j in range(60)]
        sources = {
            name: _recs({f"u{u}": list(rng.permutation(items)[:40]) for u in range(5)})
            for name in ("A", "B", "C")
        }
        quotas = {"A": 1, "B": 2, "C": 1}
        short = mix_recommendations(sources, ["A", "B", "C"], quotas, top_k=10)
        long = mix_recommendations(sources, ["A", "B", "C"], quotas, top_k=30)
        self.assertEqual(long.groupby("user_id").size().max(), 30)
        assert_frame_equal(long[long["rank"] <= 10].reset_index(drop=True), short.reset_index(drop=True))


class TopKItemsDepthTest(unittest.TestCase):
    def test_longer_top_k_keeps_the_prefix(self):
        rng = np.random.default_rng(1)
        n_users, n_items = 7, 80
        scores = np.stack([rng.permutation(n_items).astype(float) for _ in range(n_users)])  # distinct per user
        seen = rng.random((n_users, n_items)) < 0.2
        short_items, short_scores = top_k_items(scores, seen, 10)
        long_items, long_scores = top_k_items(scores, seen, 50)
        np.testing.assert_array_equal(long_items[:, :10], short_items)
        np.testing.assert_array_equal(long_scores[:, :10], short_scores)
        self.assertFalse(seen[np.arange(n_users)[:, None], long_items].any())


if __name__ == "__main__":
    unittest.main()
