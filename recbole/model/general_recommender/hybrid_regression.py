import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from recbole.utils import ModelType
from recbole.data.interaction import Interaction
from collections import defaultdict

class HybridRegression(nn.Module):
    """
    Weighted Hybrid Recommender learning coefficients via Logistic Regression.
    Acts as a wrapper around multiple pre-trained RecBole models.
    """
    type = ModelType.GENERAL

    def __init__(self, config, base_models, regressor_c=1.0, penalty='l2'):
        super(HybridRegression, self).__init__()
        self.config = config
        self.device = config['device']
        self.base_models = base_models

        # A dummy parameter to prevent RecBole's Trainer from crashing
        self.dummy_param = nn.Parameter(torch.zeros(1))
        
        # Required by RecBole evaluators
        self.n_items = self.base_models[0].n_items
        
        # Scikit-learn components for the regression meta-learner
        self.scaler = StandardScaler()
        # l1_ratio on current scikit-learn; penalty= on older versions. See _logistic.
        self.regressor = _logistic(regressor_c, penalty)

    def fit_weights(self, valid_data, num_negatives=5):
        """Extracts validation user-item pairs to train the regression weights."""
        dataset = valid_data.dataset
        user_field = self.config['USER_ID_FIELD']
        item_field = self.config['ITEM_ID_FIELD']

        users = dataset.inter_feat[user_field].numpy()
        pos_items = dataset.inter_feat[item_field].numpy()
        
        # Create a lookup of known positives for each user
        user_pos_dict = defaultdict(set)
        for u, i in zip(users, pos_items):
            user_pos_dict[u].add(i)

        samples_u, samples_i, labels = [], [], []
        
        # Generate supervised dataset with sampled negatives
        for u, pos_i in zip(users, pos_items):
            samples_u.append(u)
            samples_i.append(pos_i)
            labels.append(1)
            
            for _ in range(num_negatives):
                neg_i = np.random.randint(1, self.n_items)
                # Ensure the random item is NOT in the user's known positives
                while neg_i in user_pos_dict[u]:
                    neg_i = np.random.randint(1, self.n_items)
                    
                samples_u.append(u)
                samples_i.append(neg_i)
                labels.append(0)
        
        # Batch extraction to prevent memory issues
        X_features = []
        batch_size = 2048
        for i in range(0, len(samples_u), batch_size):
            batch_u = torch.tensor(samples_u[i:i+batch_size], device=self.device)
            batch_i = torch.tensor(samples_i[i:i+batch_size], device=self.device)
            
            interaction = Interaction({
                user_field: batch_u,
                item_field: batch_i
            })
            
            batch_scores = []
            for model in self.base_models:
                model.eval()
                with torch.no_grad():
                    score = model.predict(interaction).cpu().numpy()
                    batch_scores.append(score)
            
            X_features.append(np.column_stack(batch_scores))
            
        X = np.vstack(X_features)
        y = np.array(labels)
        
        # Fit scaler and train regressor
        X_scaled = self.scaler.fit_transform(X)
        self.regressor.fit(X_scaled, y)

        # Normalize the learned weights and intercept so the coefficients sum to 1.0
        weight_sum = np.sum(self.regressor.coef_)
        self.regressor.coef_ = self.regressor.coef_ / weight_sum
        self.regressor.intercept_ = self.regressor.intercept_ / weight_sum

    def predict(self, interaction):
        """Point-wise prediction required by RecBole."""
        scores = []
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                scores.append(model.predict(interaction).cpu().numpy())
                
        X = np.column_stack(scores)
        X_scaled = self.scaler.transform(X)
        final_scores = self.regressor.predict_proba(X_scaled)[:, 1]
        
        return torch.tensor(final_scores, device=self.device)

    def full_sort_predict(self, interaction):
        """Full-catalog ranking prediction required by RecBole Evaluator."""
        scores = []
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                # Extract and flatten scores for all items
                scores.append(model.full_sort_predict(interaction).cpu().numpy().flatten())
                
        X = np.column_stack(scores)
        X_scaled = self.scaler.transform(X)
        final_scores = self.regressor.predict_proba(X_scaled)[:, 1]
        
        # Reshape back to (batch_size, item_num) mapping
        batch_size = interaction[self.config['USER_ID_FIELD']].shape[0]
        final_scores = final_scores.reshape(batch_size, self.n_items)
        
        return torch.tensor(final_scores, device=self.device)


# --------------------------------------------------------------------------- #
# Score-matrix meta-learner used by tune_hybrid_regression.py
#
# Same model as HybridRegression.fit_weights: StandardScaler + L1/L2 logistic
# regression (liblinear) on base-model scores. Ranking uses the linear score
# coef / scale, which orders items the same way as predict_proba. The class
# also divides coefficients by their sum; that does not change the order when
# the sum is positive, and it reverses the order when the sum is negative, so
# the search does not apply it.
# --------------------------------------------------------------------------- #
import warnings

PRIMARY_METRIC = "NDCG@10"  # same as valid_metric in configs/base.yaml
TOP_K = 10
METRIC_NAMES = [f"{name}@{TOP_K}" for name in ("Recall", "MRR", "NDCG", "Hit", "Precision")]

# C grid matches run_hybrid.py. Penalty and the number of negatives are the other
# knobs of fit_weights; class_weight stays at the LogisticRegression default.
C_GRID = (0.01, 0.1, 1.0, 10.0)
PENALTIES = ("l2", "l1")
NUM_NEGATIVES_GRID = (1, 5, 20)


def _logistic(C, penalty):
    """LogisticRegression(liblinear) for either the old penalty= API or l1_ratio=."""
    if penalty not in PENALTIES:
        raise ValueError(f"penalty must be one of {PENALTIES}")
    common = {"C": float(C), "solver": "liblinear", "fit_intercept": True}
    l1_ratio = 1.0 if penalty == "l1" else 0.0
    try:
        return LogisticRegression(l1_ratio=l1_ratio, **common)
    except TypeError:
        return LogisticRegression(penalty=penalty, **common)


def align_scores(scores, users, items, ref_users, ref_items):
    """Reorder a score matrix onto the reference user and item ids.

    ContentBased keeps every item in ml-100k.item, including movies whose only ratings
    were below 3 and were dropped from the interaction models. Extra ids are ignored.
    A missing reference id is an error: that model cannot be combined with the others.
    """
    scores = np.asarray(scores)
    users = np.asarray(users).astype(str)
    items = np.asarray(items).astype(str)
    if np.array_equal(users, ref_users) and np.array_equal(items, ref_items):
        return scores
    user_index = {user: i for i, user in enumerate(users)}
    item_index = {item: i for i, item in enumerate(items)}
    try:
        user_pos = np.fromiter((user_index[user] for user in ref_users), dtype=np.int32, count=len(ref_users))
        item_pos = np.fromiter((item_index[item] for item in ref_items), dtype=np.int32, count=len(ref_items))
    except KeyError as exc:
        raise RuntimeError(f"score matrix is missing id {exc.args[0]} from the reference catalog") from exc
    return scores[np.ix_(user_pos, item_pos)]


def load_score_stack(model_names):
    """Stacked score matrices [n_models, n_users, n_items] from outputs/<model>/scores.npz.

    Columns follow the first model. Later models are reindexed onto those user and item ids.
    """
    from recbole.utils.experiment import load_scores

    matrices, user_ids, item_ids = [], None, None
    for name in model_names:
        try:
            scores, users, items = load_scores(name)
        except FileNotFoundError as exc:
            joined = " ".join(model_names)
            raise FileNotFoundError(
                f"Missing outputs/{name}/scores.npz. Run: python run_model.py {joined}"
            ) from exc
        users = np.asarray(users).astype(str)
        items = np.asarray(items).astype(str)
        scores = np.asarray(scores, dtype=np.float32)
        if user_ids is None:
            user_ids, item_ids = users, items
            matrices.append(scores)
            continue
        if len(items) != len(item_ids) or len(users) != len(user_ids):
            warnings.warn(
                f"{name} scores cover {len(users)} users x {len(items)} items; "
                f"aligning onto {model_names[0]} ({len(user_ids)} x {len(item_ids)})",
                stacklevel=2,
            )
        matrices.append(align_scores(scores, users, items, user_ids, item_ids))
    return np.stack(matrices, axis=0), user_ids, item_ids


def sample_pairs(pos_users, pos_items, forbidden, num_negatives, n_items, rng):
    """One positive row per validation interaction, plus num_negatives items outside `forbidden`.

    forbidden is a bool mask [n_users, n_items]. Positives are returned first.
    """
    pos_users = np.asarray(pos_users, dtype=np.int32)
    pos_items = np.asarray(pos_items, dtype=np.int32)
    n_pos = len(pos_users)
    neg_users = np.repeat(pos_users, num_negatives)
    neg_items = rng.integers(0, n_items, size=neg_users.shape[0], dtype=np.int32)
    for _ in range(40):
        bad = forbidden[neg_users, neg_items]
        if not bad.any():
            break
        neg_items[bad] = rng.integers(0, n_items, size=int(bad.sum()), dtype=np.int32)
    else:
        bad = np.flatnonzero(forbidden[neg_users, neg_items])
        allowed = ~forbidden
        for row in bad:
            choices = np.flatnonzero(allowed[neg_users[row]])
            if len(choices) == 0:
                raise RuntimeError(f"user {int(neg_users[row])} has no item that can be a negative")
            neg_items[row] = rng.choice(choices)
    users = np.concatenate([pos_users, neg_users])
    items = np.concatenate([pos_items, neg_items])
    labels = np.concatenate([np.ones(n_pos, dtype=np.int8), np.zeros(len(neg_users), dtype=np.int8)])
    return users, items, labels


def design_matrix(scores, user_idx, item_idx):
    """scores[:, user, item] as a contiguous [n_pairs, n_models] array."""
    return np.ascontiguousarray(np.transpose(scores[:, user_idx, item_idx]))


def fit_ranking_weights(features, labels, C, penalty):
    """Ranking weights (coef / scaler.scale) for one feature matrix.

    A positive rescaling of these weights does not change item order. The intercept
    and the scaler mean are constant across items, so they are not returned.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(np.asarray(features, dtype=np.float64))
    model = _logistic(C, penalty)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(scaled, labels)
    scale = np.asarray(scaler.scale_, dtype=np.float64).copy()
    scale[scale == 0] = 1.0
    return model.coef_.ravel() / scale


def combine_scores(scores, weights):
    """Weighted sum of score matrices. scores is [n_models, n_users, n_items]."""
    weights = np.asarray(weights, dtype=scores.dtype)
    return np.einsum("m,mui->ui", weights, scores)


def top_k_items(combined, seen, k=TOP_K):
    """Item indices and scores of the top-k unseen items per user. Unseen scores may be -inf."""
    if k <= 0 or k >= combined.shape[1]:
        raise ValueError(f"k must be between 1 and n_items - 1, got {k}")
    masked = np.array(combined, copy=True)
    masked[seen] = -np.inf
    chosen = np.argpartition(-masked, k, axis=1)[:, :k]
    part = np.take_along_axis(masked, chosen, axis=1)
    order = np.argsort(-part, axis=1, kind="stable")
    chosen = np.take_along_axis(chosen, order, axis=1)
    return chosen, np.take_along_axis(masked, chosen, axis=1)


def evaluate_ranking(combined, seen, relevant, k=TOP_K):
    """Recall, MRR, NDCG, Hit, Precision @k, averaged over users who have a relevant item.

    Returns (metrics, per-user NDCG). Per-user NDCG is NaN for users with no relevant items.
    """
    chosen, _ = top_k_items(combined, seen, k)
    n_users = combined.shape[0]
    hits = relevant[np.arange(n_users)[:, None], chosen]
    n_relevant = relevant.sum(axis=1)
    active = n_relevant > 0
    discounts = 1.0 / np.log2(np.arange(k) + 2)
    ndcg = np.full(n_users, np.nan, dtype=np.float64)
    ideal = np.cumsum(discounts)[np.minimum(n_relevant[active], k) - 1]
    ndcg[active] = (hits[active] * discounts).sum(axis=1) / ideal
    n_hits = hits.sum(axis=1)
    any_hit = n_hits > 0
    mrr = np.where(any_hit, 1.0 / (hits.argmax(axis=1) + 1), 0.0)
    metrics = {
        f"Recall@{k}": float((n_hits[active] / n_relevant[active]).mean()) if active.any() else 0.0,
        f"MRR@{k}": float(mrr[active].mean()) if active.any() else 0.0,
        f"NDCG@{k}": float(np.nanmean(ndcg)) if active.any() else 0.0,
        f"Hit@{k}": float(any_hit[active].mean()) if active.any() else 0.0,
        f"Precision@{k}": float((n_hits[active] / k).mean()) if active.any() else 0.0,
    }
    return metrics, ndcg


def format_metrics(metrics, digits=4):
    return "  ".join(f"{name}={value:.{digits}f}" for name, value in metrics.items())