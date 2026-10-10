import copy
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import yaml
import recbole
from recbole.utils import ModelType, get_model
from recbole.data.interaction import Interaction


def unwrap_dataset(data):
    """Returns the RecBole Dataset behind a DataLoader (or the Dataset itself)."""
    if hasattr(data, 'inter_matrix'):
        return data
    # RecBole >= 1.1 stores it as `_dataset` (`dataset` is torch's index list there)
    dataset = getattr(data, '_dataset', None)
    return dataset if dataset is not None else data.dataset


class _AugmentedDatasetView:
    """
    Read-only view of a RecBole Dataset whose interaction matrix is replaced by
    the augmented one. Everything else (field names, user/item counts, ...) is
    delegated to the real dataset, so any RecBole model that builds itself from
    dataset.inter_matrix() can be constructed on top of it unchanged.
    """

    def __init__(self, dataset, matrix):
        self._dataset = dataset
        self._matrix = matrix.tocsr()

    def inter_matrix(self, form='coo', value_field=None):
        if form == 'csr':
            return self._matrix.copy()
        if form == 'coo':
            return self._matrix.tocoo()
        raise NotImplementedError(f"Sparse matrix format [{form}] has not been implemented.")

    def __getattr__(self, name):
        if name in ('_dataset', '_matrix'):
            raise AttributeError(name)
        return getattr(self._dataset, name)


class HybridFeatureAugmentation(nn.Module):
    """
    Feature Augmentation Hybrid Recommender.

    Recommender 1 (the SOURCE models): any pre-trained RecBole models, e.g. BPR or
    NeuMF loaded from .pth files. For every user, each source model's top-n unseen
    items become derived features: down-weighted pseudo-interactions.

    Recommender n (the TARGET model): a RecBole model that is fitted in closed form
    from the interaction matrix (ItemKNN, EASE, SLIMElastic, ADMMSLIM, NCEPLRec).
    It is rebuilt on the ORIGINAL interactions PLUS the derived features:

        R_aug = R + pseudo_weight * TopN(source scores)

    With n_pseudo=0 the target is the plain target model trained on R (ablation).
    """
    type = ModelType.GENERAL

    # Targets that RecBole fits directly from dataset.inter_matrix() in __init__,
    # so they can be rebuilt on the augmented matrix without gradient training.
    SUPPORTED_TARGETS = ('ItemKNN', 'EASE', 'SLIMElastic', 'ADMMSLIM', 'NCEPLRec')

    def __init__(self, config, train_data, source_models, target_model='ItemKNN',
                 target_params=None, batch_size=256):
        super(HybridFeatureAugmentation, self).__init__()
        if target_model not in self.SUPPORTED_TARGETS:
            raise ValueError(f"target_model must be one of {self.SUPPORTED_TARGETS}, "
                             f"got '{target_model}'.")

        self.config = config
        self.device = config['device']
        self.dataset = unwrap_dataset(train_data)  # TRAINING split only
        self.source_models = source_models          # plain list, as in HybridRegression
        self.target_model_name = target_model
        self.batch_size = batch_size

        # Target hyperparameters: RecBole's defaults for that model, then overrides
        self.target_params = self._default_params(target_model)
        self.target_params.update(target_params or {})

        # A dummy parameter to prevent RecBole's Trainer from crashing
        self.dummy_param = nn.Parameter(torch.zeros(1))

        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']

        # Required by RecBole evaluators
        self.n_users = self.dataset.user_num
        self.n_items = self.dataset.item_num

        # Original input: training interactions (users x items), implicit 1s
        self.interaction_matrix = self.dataset.inter_matrix(form='csr').astype(np.float32)
        self.interaction_matrix.data[:] = 1.0  # guard against duplicates summing to > 1

        # Filled in by fit_augmentation()
        self._rankings = None  # per source model: (n_users, n_max) top items, -1 = none
        self.n_pseudo = None
        self.pseudo_weight = None
        self.augmented_matrix = None
        self.target = None

    @staticmethod
    def _default_params(model_name):
        path = os.path.join(os.path.dirname(recbole.__file__), 'properties', 'model',
                            f'{model_name}.yaml')
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    # ------------------------------------------------------------------ #
    # Step 1: Recommender 1 (source models) -> derived features
    # ------------------------------------------------------------------ #
    def _all_item_scores(self, model, users):
        """(len(users), n_items) scores of one source model."""
        model.eval()
        users_t = torch.as_tensor(users, dtype=torch.long)
        with torch.no_grad():
            try:
                inter = Interaction({self.USER_ID: users_t}).to(self.device)
                scores = model.full_sort_predict(inter)
            except NotImplementedError:
                # Some RecBole models (e.g. NeuMF) only implement point-wise predict()
                all_u = users_t.repeat_interleave(self.n_items)
                all_i = torch.arange(self.n_items).repeat(len(users))
                chunks = []
                for s in range(0, len(all_u), 2 ** 18):
                    pair = Interaction({self.USER_ID: all_u[s:s + 2 ** 18],
                                        self.ITEM_ID: all_i[s:s + 2 ** 18]}).to(self.device)
                    chunks.append(model.predict(pair))
                scores = torch.cat(chunks)
        scores = scores.reshape(len(users), -1)
        # A source can know more items than the hybrid's dataset: CB also loads ml-100k.item, which
        # lists movies that have no (>= 3 star) interactions. RecBole numbers items from the
        # interactions first, so the first n_items ids are shared and the extra ones come last.
        if scores.shape[1] < self.n_items:
            raise ValueError(
                f"{model.__class__.__name__} scores {scores.shape[1]} items, but the hybrid's dataset has "
                f"{self.n_items}. Pass the model with the most items (e.g. CB) after the others in "
                f"--model_files, so the hybrid takes its dataset from a model without the extra items."
            )
        return scores[:, :self.n_items].float().cpu().numpy()

    def _rank_candidates(self, model, n):
        """Top-n unseen items per user for one source model, best first (-1 = none)."""
        R = self.interaction_matrix
        ranking = np.full((self.n_users, n), -1, dtype=np.int64)

        # Index 0 is RecBole's padding user/item, so start at 1
        for start in range(1, self.n_users, self.batch_size):
            users = np.arange(start, min(start + self.batch_size, self.n_users))
            scores = self._all_item_scores(model, users)

            # Never pick the padding item or items the user already interacted with
            scores[:, 0] = -np.inf
            scores[R[users].nonzero()] = -np.inf

            top = np.argpartition(-scores, n - 1, axis=1)[:, :n]
            top_scores = np.take_along_axis(scores, top, axis=1)
            order = np.argsort(-top_scores, axis=1)
            top = np.take_along_axis(top, order, axis=1)
            top[~np.isfinite(np.take_along_axis(top_scores, order, axis=1))] = -1
            ranking[users] = top
        return ranking

    def _pseudo_matrix(self, n_pseudo, pseudo_weight):
        """Union of every source model's top-n_pseudo items, each with value pseudo_weight."""
        shape = (self.n_users, self.n_items)
        pseudo = sp.csr_matrix(shape, dtype=np.float32)
        for ranking in self._rankings:
            top = ranking[:, :n_pseudo]
            rows, cols = np.nonzero(top >= 0)
            data = np.full(len(rows), pseudo_weight, dtype=np.float32)
            pseudo = pseudo.maximum(sp.csr_matrix((data, (rows, top[rows, cols])), shape=shape))
        return pseudo

    # ------------------------------------------------------------------ #
    # Step 2: Recommender n (target model) on original input + derived features
    # ------------------------------------------------------------------ #
    def fit_augmentation(self, n_pseudo=20, pseudo_weight=0.5, target_params=None):
        """
        Builds the augmented matrix and (re)builds the target model on it.
        Source-model rankings are computed once and cached, so calling this
        repeatedly with different values (hyperparameter tuning) is cheap.
        """
        self.n_pseudo = min(n_pseudo, self.n_items - 1)
        self.pseudo_weight = pseudo_weight
        if target_params:
            self.target_params.update(target_params)

        if self.n_pseudo > 0 and self.source_models:
            if self._rankings is None or self._rankings[0].shape[1] < self.n_pseudo:
                self._rankings = [self._rank_candidates(m, self.n_pseudo) for m in self.source_models]
            pseudo = self._pseudo_matrix(self.n_pseudo, pseudo_weight)
        else:
            pseudo = sp.csr_matrix(self.interaction_matrix.shape, dtype=np.float32)

        self.augmented_matrix = (self.interaction_matrix + pseudo).tocsr()

        target_config = copy.deepcopy(self.config)
        for key, value in self.target_params.items():
            target_config[key] = value
        target_class = get_model(self.target_model_name)
        self.target = target_class(
            target_config, _AugmentedDatasetView(self.dataset, self.augmented_matrix)
        ).to(self.device)
        self.target.eval()
        return self

    # ------------------------------------------------------------------ #
    # RecBole interface
    # ------------------------------------------------------------------ #
    def calculate_loss(self, interaction):
        """Nothing to train by gradient descent; returns a zero loss for RecBole's Trainer."""
        return (self.dummy_param * 0.0).sum()

    def _check_fitted(self):
        if self.target is None:
            raise RuntimeError("Call fit_augmentation() before predicting.")

    def predict(self, interaction):
        """Point-wise prediction required by RecBole."""
        self._check_fitted()
        with torch.no_grad():
            return self.target.predict(interaction)

    def full_sort_predict(self, interaction):
        """Full-catalog ranking prediction required by RecBole Evaluator."""
        self._check_fitted()
        with torch.no_grad():
            return self.target.full_sort_predict(interaction)
        