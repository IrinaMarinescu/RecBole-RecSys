"""Tests for tune_hybrid_feature_aug.py: configuration fitting and selection methods."""

import unittest

import numpy as np
import scipy.sparse as sp
import torch

from recbole.model.general_recommender.hybrid_feature_aug import HybridFeatureAugmentation
from tune_hybrid_feature_aug import (
    FixedScores,
    TargetConfig,
    TrainMatrix,
    build_search_space,
    content_boosted_index,
    exhaustive_index,
    fit_configuration,
    full_scores,
    greedy_index,
    no_augmentation_index,
    one_per_kind_index,
    param_grid,
)

USER_ID, ITEM_ID = "user_id", "item_id"
N_USERS, N_ITEMS = 12, 15  # without RecBole's padding row/column


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    seen = rng.random((N_USERS, N_ITEMS)) < 0.3
    seen[:, :2] = True  # every user has some history and every item some interactions
    matrix = sp.csr_matrix(np.pad(seen, ((1, 0), (1, 0))).astype(np.float32))
    sources = [rng.random((N_USERS, N_ITEMS)).astype(np.float32) for _ in range(3)]
    return matrix, sources


def make_hybrid(matrix, sources, target):
    config = TargetConfig(device=torch.device("cpu"), USER_ID_FIELD=USER_ID, ITEM_ID_FIELD=ITEM_ID, NEG_PREFIX="neg_")
    models = [FixedScores(s, USER_ID) for s in sources]
    return HybridFeatureAugmentation(config, TrainMatrix(matrix, USER_ID, ITEM_ID), models, target_model=target)


class TestFitConfiguration(unittest.TestCase):
    PARAMS = {
        "ItemKNN": {"k": 5, "shrink": 1.0},
        "EASE": {"reg_weight": 10.0},
        "SLIMElastic": {"alpha": 0.01, "l1_ratio": 0.1},
    }

    def test_cached_rankings_match_a_fresh_fit(self):
        matrix, sources = make_data()
        ranker = make_hybrid(matrix, sources, "ItemKNN")
        rankings = [ranker._rank_candidates(m, 6) for m in ranker.source_models]
        for target, params in self.PARAMS.items():
            with self.subTest(target=target):
                search = make_hybrid(matrix, [], target)
                got = fit_configuration(search, rankings, (0, 2), 3, 0.5, params)

                fresh = make_hybrid(matrix, [sources[0], sources[2]], target)
                np.random.seed(2020)
                fresh.fit_augmentation(n_pseudo=3, pseudo_weight=0.5, target_params=params)
                np.testing.assert_array_equal(got, full_scores(fresh))

    def test_no_sources_is_the_plain_target(self):
        matrix, sources = make_data()
        ranker = make_hybrid(matrix, sources, "EASE")
        rankings = [ranker._rank_candidates(m, 6) for m in ranker.source_models]
        search = make_hybrid(matrix, [], "EASE")
        got = fit_configuration(search, rankings, (), 0, 0.0, self.PARAMS["EASE"])

        plain = make_hybrid(matrix, [], "EASE")
        plain.fit_augmentation(n_pseudo=0, target_params=self.PARAMS["EASE"])
        np.testing.assert_array_equal(got, full_scores(plain))
        self.assertEqual((search.augmented_matrix != search.interaction_matrix).nnz, 0)

    def test_one_configuration_does_not_leak_into_the_next(self):
        matrix, sources = make_data()
        ranker = make_hybrid(matrix, sources, "ItemKNN")
        rankings = [ranker._rank_candidates(m, 6) for m in ranker.source_models]
        search = make_hybrid(matrix, [], "ItemKNN")
        first = fit_configuration(search, rankings, (1,), 2, 1.0, self.PARAMS["ItemKNN"])
        fit_configuration(search, rankings, (0, 1, 2), 6, 0.25, {"k": 3, "shrink": 0.0})
        again = fit_configuration(search, rankings, (1,), 2, 1.0, self.PARAMS["ItemKNN"])
        np.testing.assert_array_equal(first, again)


class TestSearchSpace(unittest.TestCase):
    def test_plain_targets_first_then_subsets_by_size(self):
        space, tasks, params = build_search_space(["ItemKNN", "EASE"], 3)
        n_plain = len(params["ItemKNN"]) + len(params["EASE"])
        self.assertTrue((space["mask"][:n_plain] == 0).all())
        self.assertTrue((space["mask"][n_plain:] > 0).all())
        sizes = [bin(m).count("1") for m in space["mask"]]
        self.assertEqual(sizes, sorted(sizes))

    def test_every_configuration_is_fitted_exactly_once(self):
        space, tasks, _ = build_search_space(["ItemKNN", "EASE"], 3)
        rows = np.concatenate([r for t in tasks.values() for _, r in t])
        np.testing.assert_array_equal(np.sort(rows), np.arange(len(space["mask"])))
        for target, target_tasks in tasks.items():
            for (task_target, sources, n_pseudo, weight), task_rows in target_tasks:
                self.assertEqual(task_target, target)
                mask = sum(1 << m for m in sources)
                self.assertTrue((space["mask"][task_rows] == mask).all())
                self.assertTrue((space["n_pseudo"][task_rows] == n_pseudo).all())

    def test_param_grid_is_the_product(self):
        self.assertEqual(len(param_grid("ItemKNN")), 6)
        self.assertIn({"k": 200, "shrink": 0.0}, param_grid("ItemKNN"))


class TestSelection(unittest.TestCase):
    # Three sources: 0 = neighbourhood, 1 = latent factor, 2 = content. Two targets.
    NAMES = ["A", "B", "C"]
    KINDS = {"A": "neighbourhood", "B": "latent factor", "C": "content"}

    def setUp(self):
        #                 plain t0  plain t1  {C}@t0  {C}@t1  {A}   {B}   {A,B}  {A,C}  {A,B,C}
        self.masks = np.array([0, 0, 4, 4, 1, 2, 3, 5, 7])
        self.targets = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0])
        self.params = np.zeros(9, dtype=int)
        self.scores = np.array([0.20, 0.22, 0.30, 0.21, 0.23, 0.25, 0.27, 0.24, 0.26])

    def test_no_augmentation_and_exhaustive(self):
        self.assertEqual(no_augmentation_index(self.scores, self.masks), 1)
        self.assertEqual(exhaustive_index(self.scores), 2)

    def test_content_boosted_stays_on_the_no_augmentation_target(self):
        # {C} on target 0 scores higher, but the plain-target winner is target 1.
        i = content_boosted_index(self.scores, self.masks, self.targets, self.params, content_bit=4)
        self.assertEqual(i, 3)

    def test_one_per_kind_uses_only_the_best_source_of_each_kind(self):
        names = ["A", "A2", "B"]
        kinds = {"A": "neighbourhood", "A2": "neighbourhood", "B": "latent factor"}
        masks = np.array([0, 1, 2, 3, 4, 5, 6])
        scores = np.array([0.1, 0.2, 0.9, 0.3, 0.25, 0.5, 0.4])
        # A2 (bit 2^1) is the weaker neighbourhood source, so masks containing it are not eligible.
        self.assertEqual(one_per_kind_index(scores, masks, names, kinds, source_scores=[0.3, 0.2, 0.25]), 5)

    def test_greedy_adds_sources_while_they_help(self):
        chosen, path = greedy_index(self.scores, self.masks, 3)
        # Plain best 0.22 -> {C} 0.30 (best single source) -> {A,C} 0.24 and {B,C} absent: stop.
        self.assertEqual(path, [2])
        self.assertEqual(chosen, 2)

    def test_greedy_returns_the_plain_target_when_nothing_helps(self):
        scores = np.array([0.30, 0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
        chosen, path = greedy_index(scores, self.masks, 3)
        self.assertEqual((chosen, path), (0, []))

    def test_tie_keeps_the_earlier_row(self):
        scores = np.array([0.2, 0.2, 0.3, 0.3, 0, 0, 0, 0, 0])
        self.assertEqual(no_augmentation_index(scores, self.masks), 0)
        self.assertEqual(exhaustive_index(scores), 2)


if __name__ == "__main__":
    unittest.main()
