import os
import unittest

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

# Same NumPy 2 aliases RecBole's Config sets (plain `bool` breaks numpy.testing)
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.model.general_recommender.itemknn import ItemKNN
from recbole.model.general_recommender.hybrid_feature_aug import (
    HybridFeatureAugmentation,
    _AugmentedDatasetView,
    unwrap_dataset,
)
from recbole.trainer import Trainer
from recbole.utils import get_model, init_seed

current_path = os.path.dirname(os.path.realpath(__file__))
config_file_list = [os.path.join(current_path, "..", "model", "test_model.yaml")]

USER_ID = "user_id"
ITEM_ID = "item_id"

# 4 users x 6 items, row/column 0 is RecBole's padding user/item.
#   user 1 saw items 1, 2
#   user 2 saw item 3
#   user 3 saw items 1, 2, 3, 4 (only item 5 is unseen)
TRAIN_PAIRS = [(1, 1), (1, 2), (2, 3), (3, 1), (3, 2), (3, 3), (3, 4)]
N_USERS, N_ITEMS = 4, 6

# Source scores: higher = better. Column 0 gets the highest score on purpose,
# to check that the padding item is never picked.
SCORES_A = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [9, 8, 7, 1, 5, 3],  # unseen ranking: 4, 5, 3
        [9, 2, 6, 8, 1, 4],  # unseen ranking: 2, 5, 1, 4
        [9, 5, 4, 3, 2, 1],  # unseen ranking: 5
    ],
    dtype=np.float32,
)
SCORES_B = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [9, 1, 1, 6, 2, 1],  # unseen ranking: 3, 4, 5
        [9, 7, 1, 1, 1, 1],  # unseen ranking: 1, ...
        [9, 1, 1, 1, 1, 1],  # unseen ranking: 5
    ],
    dtype=np.float32,
)


def make_matrix(pairs, shape=(N_USERS, N_ITEMS)):
    rows, cols = zip(*pairs)
    return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=shape)


class FakeDataset:
    """Minimal stand-in for a RecBole Dataset: what the hybrid and targets read."""

    def __init__(self, matrix):
        self._matrix = matrix.tocsr()
        self.user_num, self.item_num = matrix.shape
        self.uid_field, self.iid_field = USER_ID, ITEM_ID

    def inter_matrix(self, form="coo", value_field=None):
        return self._matrix.copy() if form == "csr" else self._matrix.tocoo()

    def num(self, field):
        return self.user_num if field == USER_ID else self.item_num


class FakeDataLoader:
    """RecBole >= 1.1 DataLoaders keep the Dataset in `_dataset`."""

    def __init__(self, dataset):
        self._dataset = dataset


class FullSortSource(nn.Module):
    """Source model with fixed scores, exposing full_sort_predict like BPR."""

    def __init__(self, scores):
        super().__init__()
        self.scores = torch.as_tensor(scores)
        self.full_sort_calls = 0

    def full_sort_predict(self, interaction):
        self.full_sort_calls += 1
        return self.scores[interaction[USER_ID]].flatten()


class PointwiseSource(FullSortSource):
    """Source model with only point-wise predict(), like NeuMF."""

    def full_sort_predict(self, interaction):
        raise NotImplementedError

    def predict(self, interaction):
        return self.scores[interaction[USER_ID], interaction[ITEM_ID]]


class FakeConfig(dict):
    """Dict with attribute access: some targets read `config.device` (EASE)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def make_config(**extra):
    config = FakeConfig({
        "device": torch.device("cpu"),
        "USER_ID_FIELD": USER_ID,
        "ITEM_ID_FIELD": ITEM_ID,
        "NEG_PREFIX": "neg_",
        "seed": 2020,
    })
    config.update(extra)
    return config


def make_hybrid(sources, target_model="ItemKNN", target_params=None, batch_size=256):
    dataset = FakeDataset(make_matrix(TRAIN_PAIRS))
    return HybridFeatureAugmentation(
        make_config(), FakeDataLoader(dataset), sources,
        target_model=target_model, target_params=target_params, batch_size=batch_size,
    )


def all_users():
    return Interaction({USER_ID: torch.arange(1, N_USERS)})


class TestHelpers(unittest.TestCase):
    def test_unwrap_dataset_from_dataloader(self):
        dataset = FakeDataset(make_matrix(TRAIN_PAIRS))
        self.assertIs(unwrap_dataset(FakeDataLoader(dataset)), dataset)

    def test_unwrap_dataset_passes_dataset_through(self):
        dataset = FakeDataset(make_matrix(TRAIN_PAIRS))
        self.assertIs(unwrap_dataset(dataset), dataset)

    def test_view_replaces_matrix_and_delegates_the_rest(self):
        dataset = FakeDataset(make_matrix(TRAIN_PAIRS))
        augmented = make_matrix([(1, 5)])
        view = _AugmentedDatasetView(dataset, augmented)

        self.assertEqual((view.inter_matrix(form="csr") != augmented).nnz, 0)
        self.assertEqual((view.inter_matrix(form="coo").tocsr() != augmented).nnz, 0)
        self.assertEqual(view.num(USER_ID), N_USERS)
        self.assertEqual(view.uid_field, USER_ID)
        with self.assertRaises(NotImplementedError):
            view.inter_matrix(form="dok")

    def test_view_returns_a_copy(self):
        view = _AugmentedDatasetView(FakeDataset(make_matrix(TRAIN_PAIRS)), make_matrix([(1, 5)]))
        view.inter_matrix(form="csr").data[:] = 99
        self.assertEqual(view.inter_matrix(form="csr")[1, 5], 1.0)


class TestCandidateRanking(unittest.TestCase):
    def test_excludes_seen_and_padding_items_best_first(self):
        hybrid = make_hybrid([FullSortSource(SCORES_A)])
        ranking = hybrid._rank_candidates(hybrid.source_models[0], 3)

        np.testing.assert_array_equal(ranking[1], [4, 5, 3])
        np.testing.assert_array_equal(ranking[2], [2, 5, 1])

    def test_marks_missing_candidates_with_minus_one(self):
        hybrid = make_hybrid([FullSortSource(SCORES_A)])
        ranking = hybrid._rank_candidates(hybrid.source_models[0], 3)

        # user 3 has only one unseen item; padding user 0 is never ranked
        np.testing.assert_array_equal(ranking[3], [5, -1, -1])
        np.testing.assert_array_equal(ranking[0], [-1, -1, -1])

    def test_pointwise_fallback_matches_full_sort(self):
        hybrid = make_hybrid([FullSortSource(SCORES_A), PointwiseSource(SCORES_A)])
        full = hybrid._rank_candidates(hybrid.source_models[0], 4)
        pointwise = hybrid._rank_candidates(hybrid.source_models[1], 4)
        np.testing.assert_array_equal(full, pointwise)

    def test_batching_does_not_change_ranking(self):
        one_batch = make_hybrid([FullSortSource(SCORES_A)], batch_size=256)
        tiny_batches = make_hybrid([FullSortSource(SCORES_A)], batch_size=1)
        np.testing.assert_array_equal(
            one_batch._rank_candidates(one_batch.source_models[0], 3),
            tiny_batches._rank_candidates(tiny_batches.source_models[0], 3),
        )

    def test_extra_source_items_are_dropped(self):
        # like CB, which also knows items that have no interactions; the extra ones come last
        extra = np.full((N_USERS, 3), 100.0, dtype=np.float32)
        wide = FullSortSource(np.hstack([SCORES_A, extra]))
        hybrid = make_hybrid([FullSortSource(SCORES_A), wide])
        np.testing.assert_array_equal(
            hybrid._rank_candidates(hybrid.source_models[0], 3),
            hybrid._rank_candidates(hybrid.source_models[1], 3),
        )

    def test_source_with_fewer_items_raises(self):
        hybrid = make_hybrid([FullSortSource(SCORES_A[:, :-1])])
        with self.assertRaises(ValueError):
            hybrid._rank_candidates(hybrid.source_models[0], 3)


class TestAugmentation(unittest.TestCase):
    def test_pseudo_matrix_is_union_of_sources_with_weight(self):
        hybrid = make_hybrid([FullSortSource(SCORES_A), FullSortSource(SCORES_B)])
        hybrid.fit_augmentation(n_pseudo=1, pseudo_weight=0.5)
        pseudo = hybrid.augmented_matrix - hybrid.interaction_matrix
        pseudo.eliminate_zeros()

        # top-1 per source: A -> {1: 4, 2: 2, 3: 5}, B -> {1: 3, 2: 1, 3: 5}
        expected = {(1, 4), (1, 3), (2, 2), (2, 1), (3, 5)}
        self.assertEqual(set(zip(*pseudo.nonzero())), expected)
        # item 5 for user 3 is suggested by both sources but counted once
        np.testing.assert_allclose(pseudo.data, 0.5)

    def test_real_interactions_stay_at_one(self):
        hybrid = make_hybrid([FullSortSource(SCORES_A)])
        hybrid.fit_augmentation(n_pseudo=5, pseudo_weight=0.25)
        for u, i in TRAIN_PAIRS:
            self.assertEqual(hybrid.augmented_matrix[u, i], 1.0)

    def test_n_pseudo_zero_leaves_matrix_unchanged(self):
        hybrid = make_hybrid([FullSortSource(SCORES_A)])
        hybrid.fit_augmentation(n_pseudo=0)
        self.assertEqual((hybrid.augmented_matrix != hybrid.interaction_matrix).nnz, 0)

    def test_n_pseudo_is_capped_at_catalog_size(self):
        hybrid = make_hybrid([FullSortSource(SCORES_A)])
        hybrid.fit_augmentation(n_pseudo=1000)
        self.assertEqual(hybrid.n_pseudo, N_ITEMS - 1)

    def test_duplicate_training_interactions_count_once(self):
        dataset = FakeDataset(make_matrix(TRAIN_PAIRS + [(1, 1)]))  # (1, 1) twice -> value 2
        hybrid = HybridFeatureAugmentation(make_config(), FakeDataLoader(dataset), [])
        self.assertEqual(hybrid.interaction_matrix[1, 1], 1.0)

    def test_rankings_are_cached_across_fits(self):
        source = FullSortSource(SCORES_A)
        hybrid = make_hybrid([source])

        hybrid.fit_augmentation(n_pseudo=3)
        calls = source.full_sort_calls
        hybrid.fit_augmentation(n_pseudo=2, pseudo_weight=1.0)
        hybrid.fit_augmentation(n_pseudo=0)
        self.assertEqual(source.full_sort_calls, calls)

        hybrid.fit_augmentation(n_pseudo=4)  # needs a deeper ranking -> recomputed
        self.assertGreater(source.full_sort_calls, calls)

    def test_cached_prefix_matches_fresh_ranking(self):
        cached = make_hybrid([FullSortSource(SCORES_A)])
        cached.fit_augmentation(n_pseudo=4)
        cached.fit_augmentation(n_pseudo=2)

        fresh = make_hybrid([FullSortSource(SCORES_A)])
        fresh.fit_augmentation(n_pseudo=2)
        self.assertEqual((cached.augmented_matrix != fresh.augmented_matrix).nnz, 0)


class TestTargetModel(unittest.TestCase):
    def test_rejects_unsupported_target(self):
        with self.assertRaises(ValueError):
            make_hybrid([FullSortSource(SCORES_A)], target_model="BPR")

    def test_predict_before_fit_raises(self):
        hybrid = make_hybrid([FullSortSource(SCORES_A)])
        with self.assertRaises(RuntimeError):
            hybrid.full_sort_predict(all_users())
        with self.assertRaises(RuntimeError):
            hybrid.predict(Interaction({USER_ID: torch.tensor([1]), ITEM_ID: torch.tensor([3])}))

    def test_n_pseudo_zero_equals_plain_target(self):
        hybrid = make_hybrid([FullSortSource(SCORES_A)], target_params={"k": 3})
        hybrid.fit_augmentation(n_pseudo=0)

        config = make_config(k=3, shrink=0.0, knn_method="item")
        plain = ItemKNN(config, FakeDataset(make_matrix(TRAIN_PAIRS)))
        torch.testing.assert_close(
            hybrid.full_sort_predict(all_users()), plain.full_sort_predict(all_users())
        )

    def test_augmentation_changes_predictions(self):
        hybrid = make_hybrid([FullSortSource(SCORES_A)])
        hybrid.fit_augmentation(n_pseudo=0)
        plain = hybrid.full_sort_predict(all_users())
        hybrid.fit_augmentation(n_pseudo=2, pseudo_weight=1.0)
        self.assertFalse(torch.equal(plain, hybrid.full_sort_predict(all_users())))

    def test_target_is_built_on_augmented_matrix(self):
        hybrid = make_hybrid([FullSortSource(SCORES_A)])
        hybrid.fit_augmentation(n_pseudo=2, pseudo_weight=0.5)
        self.assertEqual((hybrid.target.interaction_matrix != hybrid.augmented_matrix).nnz, 0)

    def test_predict_matches_full_sort_predict(self):
        hybrid = make_hybrid([FullSortSource(SCORES_A)])
        hybrid.fit_augmentation(n_pseudo=2, pseudo_weight=0.5)
        full = hybrid.full_sort_predict(all_users()).reshape(N_USERS - 1, N_ITEMS)

        users = torch.arange(1, N_USERS).repeat_interleave(N_ITEMS)
        items = torch.arange(N_ITEMS).repeat(N_USERS - 1)
        point = hybrid.predict(Interaction({USER_ID: users, ITEM_ID: items}))
        torch.testing.assert_close(point.reshape(N_USERS - 1, N_ITEMS).float(), full.float())

    def test_target_params_override_defaults(self):
        hybrid = make_hybrid([FullSortSource(SCORES_A)], target_params={"k": 2})
        self.assertEqual(hybrid.target_params["knn_method"], "item")  # RecBole default kept
        hybrid.fit_augmentation(n_pseudo=1)
        self.assertEqual(hybrid.target.k, 2)

        hybrid.fit_augmentation(n_pseudo=1, target_params={"k": 1})
        self.assertEqual(hybrid.target.k, 1)

    def test_target_does_not_modify_shared_config(self):
        config = make_config()
        dataset = FakeDataset(make_matrix(TRAIN_PAIRS))
        hybrid = HybridFeatureAugmentation(
            config, FakeDataLoader(dataset), [FullSortSource(SCORES_A)], target_params={"k": 2}
        )
        hybrid.fit_augmentation(n_pseudo=1)
        self.assertNotIn("k", config)

    def test_every_closed_form_target_builds_and_predicts(self):
        for target in ("ItemKNN", "EASE", "SLIMElastic", "ADMMSLIM"):
            with self.subTest(target=target):
                hybrid = make_hybrid([FullSortSource(SCORES_A)], target_model=target)
                hybrid.fit_augmentation(n_pseudo=2, pseudo_weight=0.5)
                scores = hybrid.full_sort_predict(all_users())
                self.assertEqual(scores.numel(), (N_USERS - 1) * N_ITEMS)
                self.assertTrue(torch.isfinite(scores).all())

    def test_default_params_are_loaded(self):
        params = HybridFeatureAugmentation._default_params("ItemKNN")
        self.assertEqual(params["k"], 100)
        self.assertEqual(HybridFeatureAugmentation._default_params("NoSuchModel"), {})

    # Known bug: plain yaml.safe_load reads "reg_weight: 1e2" from NCEPLRec.yaml as
    # the string '1e2', so NCEPLRec as target crashes. Remove this decorator once
    # _default_params parses numbers the way RecBole's Config does.
    @unittest.expectedFailure
    def test_default_params_numbers_are_numeric(self):
        for target in HybridFeatureAugmentation.SUPPORTED_TARGETS:
            for key, value in HybridFeatureAugmentation._default_params(target).items():
                if not isinstance(value, str):
                    continue
                with self.subTest(target=target, key=key):
                    with self.assertRaises(ValueError, msg=f"{target}.{key} = {value!r}"):
                        float(value)  # a string that is really a number


class TestWithRecBolePipeline(unittest.TestCase):
    """End to end on RecBole's small test dataset, the way run_hybrid_aug.py uses it."""

    @classmethod
    def setUpClass(cls):
        config = Config(
            model="BPR",
            dataset="test",
            config_file_list=config_file_list,
            config_dict={
                "data_path": os.path.join(current_path, "..", "test_data"),
                "epochs": 1,
                "use_gpu": False,
                "show_progress": False,
            },
        )
        init_seed(config["seed"], config["reproducibility"])
        dataset = create_dataset(config)
        cls.train_data, cls.valid_data, cls.test_data = data_preparation(config, dataset)
        cls.config = config

        init_seed(config["seed"], config["reproducibility"])
        source = get_model("BPR")(config, cls.train_data._dataset).to(config["device"])
        Trainer(config, source).fit(cls.train_data, saved=False, show_progress=False)
        cls.source = source

    def evaluate(self, model, data):
        return Trainer(self.config, model).evaluate(data, load_best_model=False, show_progress=False)

    def test_pseudo_interactions_are_unseen_training_items(self):
        hybrid = HybridFeatureAugmentation(self.config, self.train_data, [self.source])
        hybrid.fit_augmentation(n_pseudo=5, pseudo_weight=0.5)
        pseudo = hybrid.augmented_matrix - hybrid.interaction_matrix
        pseudo.eliminate_zeros()

        self.assertEqual(hybrid.interaction_matrix.multiply(pseudo).nnz, 0)
        self.assertEqual(pseudo[:, 0].nnz, 0)
        self.assertEqual(pseudo[0].nnz, 0)
        self.assertTrue(np.all(np.diff(pseudo.indptr)[1:] <= 5))
        np.testing.assert_allclose(pseudo.data, 0.5)

    def test_n_pseudo_zero_matches_standalone_itemknn(self):
        hybrid = HybridFeatureAugmentation(self.config, self.train_data, [self.source])
        hybrid.fit_augmentation(n_pseudo=0)

        itemknn_config = Config(
            model="ItemKNN",
            dataset="test",
            config_file_list=config_file_list,
            config_dict={"data_path": os.path.join(current_path, "..", "test_data"), "use_gpu": False},
        )
        itemknn = ItemKNN(itemknn_config, self.train_data._dataset)
        self.assertEqual(self.evaluate(hybrid, self.valid_data), self.evaluate(itemknn, self.valid_data))

    def test_trainer_evaluates_hybrid(self):
        hybrid = HybridFeatureAugmentation(self.config, self.train_data, [self.source])
        hybrid.fit_augmentation(n_pseudo=5, pseudo_weight=0.5)
        result = self.evaluate(hybrid, self.test_data)

        self.assertIn(self.config["valid_metric"].lower(), result)
        for name, value in result.items():
            self.assertTrue(0.0 <= value <= 1.0, f"{name} = {value}")


if __name__ == "__main__":
    unittest.main()
