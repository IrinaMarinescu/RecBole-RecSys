"""Unit tests for Mixed Hybrid list mixing / deduplication."""

import unittest

import numpy as np
import pandas as pd

from recbole.model.general_recommender.mixed_hybrid import (
    evaluate_arrays,
    evaluate_recommendations,
    mix_arrays,
    mix_recommendations,
    to_arrays,
)


def _recs(model_rows):
    """Build a source frame from {user: [item, ...]} preserving list order as rank."""
    rows = []
    for user_id, items in model_rows.items():
        for rank, item_id in enumerate(items, start=1):
            rows.append(
                {
                    "user_id": str(user_id),
                    "item_id": str(item_id),
                    "rank": rank,
                    "score": float(len(items) - rank + 1),
                }
            )
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rank", "score"])


class TestMixing(unittest.TestCase):
    def test_duplicates_removed(self):
        sources = {
            "A": _recs({"u1": ["A", "B", "C"]}),
            "B": _recs({"u1": ["B", "D", "E"]}),
            "C": _recs({"u1": ["A", "F", "G"]}),
        }
        mixed = mix_recommendations(sources, ["A", "B", "C"], top_k=10)
        items = mixed["item_id"].tolist()
        # Round-robin order with per-user dedupe (skip already-seen, keep filling).
        self.assertEqual(items, ["A", "B", "F", "C", "D", "G", "E"])
        self.assertEqual(len(items), len(set(items)))

    def test_quota_affects_order(self):
        sources = {
            "ItemKNN": _recs({"u1": ["A", "B", "C", "D", "E"]}),
            "BPR": _recs({"u1": ["F", "G", "H", "I", "J"]}),
            "ContentBased": _recs({"u1": ["K", "L", "M", "N", "O"]}),
        }
        quotas = {"ItemKNN": 2, "BPR": 1, "ContentBased": 1}
        mixed = mix_recommendations(sources, list(quotas), quotas, top_k=8)
        self.assertEqual(
            mixed["item_id"].tolist(),
            ["A", "B", "F", "K", "C", "D", "G", "L"],
        )
        self.assertEqual(
            mixed["source"].tolist(),
            [
                "ItemKNN",
                "ItemKNN",
                "BPR",
                "ContentBased",
                "ItemKNN",
                "ItemKNN",
                "BPR",
                "ContentBased",
            ],
        )

    def test_user_isolation(self):
        sources = {
            "A": _recs({"u1": ["X", "Y"], "u2": ["X", "Z"]}),
            "B": _recs({"u1": ["X", "Z"], "u2": ["Y", "X"]}),
        }
        mixed = mix_recommendations(sources, ["A", "B"], top_k=3)
        u1 = mixed[mixed["user_id"] == "u1"]["item_id"].tolist()
        u2 = mixed[mixed["user_id"] == "u2"]["item_id"].tolist()
        # u1: A->X, B skips X then takes Z, A->Y
        self.assertEqual(u1, ["X", "Z", "Y"])
        # u2: A->X, B->Y, A->Z
        self.assertEqual(u2, ["X", "Y", "Z"])

    def test_missing_source_for_user(self):
        sources = {
            "A": _recs({"u1": ["A1", "A2", "A3"]}),
            "B": _recs({}),  # no recommendations at all
            "C": _recs({"u1": ["C1", "C2"]}),
        }
        mixed = mix_recommendations(sources, ["A", "B", "C"], top_k=4)
        self.assertEqual(mixed["item_id"].tolist(), ["A1", "C1", "A2", "C2"])

    def test_top_k_cap(self):
        sources = {
            "A": _recs({"u1": list("ABCDEFGHIJ")}),
            "B": _recs({"u1": list("KLMNOPQRST")}),
        }
        mixed = mix_recommendations(sources, ["A", "B"], top_k=5)
        self.assertEqual(len(mixed), 5)
        self.assertEqual(mixed["rank"].tolist(), [1, 2, 3, 4, 5])

    def test_determinism(self):
        sources = {
            "A": _recs({"u1": ["A", "B", "C"]}),
            "B": _recs({"u1": ["D", "E", "A"]}),
            "C": _recs({"u1": ["F", "G", "H"]}),
        }
        quotas = {"A": 2, "B": 1, "C": 1}
        first = mix_recommendations(sources, list(quotas), quotas, top_k=6)
        second = mix_recommendations(sources, list(quotas), quotas, top_k=6)
        pd.testing.assert_frame_equal(first, second)

    def test_exhausted_source_continues(self):
        sources = {
            "A": _recs({"u1": ["A1", "A2"]}),
            "B": _recs({"u1": ["B1", "B2", "B3", "B4", "B5", "B6"]}),
        }
        mixed = mix_recommendations(sources, ["A", "B"], {"A": 5, "B": 1}, top_k=6)
        # A only has 2 unique items; mixer should keep filling from B.
        self.assertEqual(len(mixed), 6)
        self.assertIn("A1", mixed["item_id"].tolist())
        self.assertIn("B1", mixed["item_id"].tolist())


class TestArrayVersion(unittest.TestCase):
    """mix_arrays / evaluate_arrays must give exactly the same results as the reference functions."""

    def test_matches_reference(self):
        rng = np.random.default_rng(0)
        models = ["A", "B", "C", "D"]
        users = [f"u{u}" for u in range(30)]
        # Small item pool so lists overlap a lot; some users are missing from some models
        # and some lists are shorter than top_k.
        sources = {
            name: _recs({
                user: rng.choice(25, size=rng.integers(3, 15), replace=False).tolist()
                for user in users if rng.random() > 0.1
            })
            for name in models
        }
        truth = pd.DataFrame(
            [(user, str(item)) for user in users for item in rng.choice(25, size=rng.integers(1, 6), replace=False)],
            columns=["user_id", "item_id"],
        )
        lists, item_ids = to_arrays(sources, models, users)
        index = {item: i for i, item in enumerate(item_ids)}
        relevant = np.zeros((len(users), len(item_ids) + 1), dtype=bool)  # last column: items no model recommends
        for user, item in zip(truth["user_id"], truth["item_id"]):
            relevant[users.index(user), index.get(item, len(item_ids))] = True

        for quotas in ([1, 1, 1, 1], [3, 0, 2, 1], [10, 0, 0, 0], [0, 4, 0, 6], [2, 5, 1, 7]):
            reference = mix_recommendations(sources, models, dict(zip(models, quotas)), top_k=10)
            chosen = mix_arrays(lists, quotas, top_k=10)
            for u, user in enumerate(users):
                expected = reference.loc[reference["user_id"] == user, "item_id"].tolist()
                got = [item_ids[i] for i in chosen[u] if i >= 0]
                self.assertEqual(got, expected, f"quotas={quotas} user={user}")

            expected_metrics = evaluate_recommendations(reference, truth, k=10)
            metrics, _ = evaluate_arrays(chosen, relevant)
            for name, value in expected_metrics.items():
                self.assertAlmostEqual(metrics[name], value, places=12, msg=f"{name} quotas={quotas}")


class TestEvaluation(unittest.TestCase):
    def test_evaluate_simple(self):
        recs = _recs({"1": ["a", "b", "c"]})
        truth = pd.DataFrame({"user_id": ["1", "1"], "item_id": ["b", "z"]})
        metrics = evaluate_recommendations(recs, truth, k=3)
        self.assertAlmostEqual(metrics["Hit@3"], 1.0)
        self.assertAlmostEqual(metrics["Recall@3"], 0.5)
        self.assertAlmostEqual(metrics["MRR@3"], 0.5)


if __name__ == "__main__":
    unittest.main()
