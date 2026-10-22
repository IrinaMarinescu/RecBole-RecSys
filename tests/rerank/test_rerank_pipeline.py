"""tune_rerankers.py / run_rerankers.py helpers on a tiny synthetic split (no files under outputs/ needed)."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

import tune_rerankers
from recbole.model.general_recommender.rerankers import PopularityDiscount, rerank
from recbole.utils.evaluation import Evaluator
from run_rerankers import share_changed, test_settings
from tune_rerankers import no_worse_for, settings_for, sweep

# User u<n> rated the first n items of a..f, so item popularity is a: 6 ... f: 1, g-j: 0.
ITEMS = list("abcdefghij")
TRAIN = pd.DataFrame([(f"u{n}", item) for n in range(1, 7) for item in ITEMS[:n]], columns=["user_id", "item_id"])
TEST = pd.DataFrame({"user_id": ["u1", "u1", "u2"], "item_id": ["c", "g", "h"]})
GENRES = {i: {"Comedy"} for i in ITEMS} | {"c": {"Comedy", "Drama"}}
BASE = pd.DataFrame([
    {"user_id": u, "item_id": item, "rank": rank, "score": 1.0 - 0.1 * rank}
    for u, items in {"u1": "bcdgha", "u2": "abhcdg"}.items()
    for rank, item in enumerate(items, start=1)
])


class SweepTest(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator(TRAIN, TEST, GENRES, ITEMS)
        self.reranker = PopularityDiscount(self.evaluator.item_popularity)

    def test_one_row_per_setting_and_base_row(self):
        table = sweep(BASE, self.reranker, settings_for("Popularity", [0.0, 0.5, 1.0], []), self.evaluator, k=3)
        self.assertEqual(table["lambda"].tolist(), [0.0, 0.5, 1.0])
        base = self.evaluator.evaluate(BASE, 3)
        for metric, value in base.items():
            if np.isnan(value):  # group metrics: the one active user (u6) has no test items
                self.assertTrue(np.isnan(table.loc[0, metric]), metric)
            else:
                self.assertAlmostEqual(table.loc[0, metric], value, msg=metric)
        self.assertGreater(table.loc[2, "APLT@3"], table.loc[0, "APLT@3"])  # pure popularity discount fills the tail

    def test_settings(self):
        lambdas = [round(0.1 * i, 1) for i in range(11)]
        self.assertEqual(len(settings_for("Fairness", lambdas, [0.0, 0.5, 1.0])), 1 + 10 * 3)
        self.assertEqual(settings_for("MMR", [0.0, 0.5], [0.0]), [{"lambda": 0.0}, {"lambda": 0.5}])
        chosen = {"lambda": 0.8, "inactive_weight": 0.5}
        self.assertEqual(test_settings("Fairness", [0.0, 0.5], chosen),
                         [{"lambda": 0.0, "inactive_weight": 0.5}, {"lambda": 0.5, "inactive_weight": 0.5}])
        self.assertEqual(test_settings("MMR", [0.0, 0.5], {"lambda": 0.5}), [{"lambda": 0.0}, {"lambda": 0.5}])
        self.assertEqual(no_worse_for("Fairness", 10), [("UserNDCGGap@10", False)])
        self.assertEqual(no_worse_for("MMR", 10), [])

    def test_share_changed(self):
        reranked = rerank(BASE, self.reranker, 1.0, top_k=3)
        # pure popularity discount: u1 picks g, h (ranks 4, 5) and d; u2 picks h, g (rank 6) and d (rank 5)
        self.assertAlmostEqual(share_changed(reranked, 3), 4 / 6)
        self.assertEqual(share_changed(rerank(BASE, self.reranker, 0.0, top_k=3), 3), 0.0)


class ConfigTest(unittest.TestCase):
    def test_round_trip(self):
        config = {"base_model": "Toy", "k": 10, "primary_metric": "NDCG@10", "max_accuracy_loss": 0.05,
                  "lambdas": [0.0, 0.5], "inactive_weights": [1.0], "base_valid_metrics": {"NDCG@10": 0.25},
                  "rerankers": {"MMR": {"lambda": 0.5, "target": "ILD@10", "valid_metrics": {"NDCG@10": 0.24}}}}
        original = tune_rerankers.RERANK_CONFIG_DIR
        with tempfile.TemporaryDirectory() as tmp:
            tune_rerankers.RERANK_CONFIG_DIR = tmp
            try:
                with open(tune_rerankers.config_path("Toy"), "w", encoding="utf-8") as f:
                    yaml.safe_dump(config, f, sort_keys=False)
                self.assertEqual(tune_rerankers.configured_bases(), ["Toy"])
                loaded = tune_rerankers.load_rerank_config("Toy")
            finally:
                tune_rerankers.RERANK_CONFIG_DIR = original
        self.assertEqual(loaded, config)
        self.assertIsInstance(loaded["rerankers"]["MMR"]["lambda"], float)
        self.assertEqual(os.path.basename(tune_rerankers.config_path("RecVAE")), "RecVAE.yaml")


if __name__ == "__main__":
    unittest.main()
