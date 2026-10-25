import math
import os
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from recbole.model.general_recommender.mixed_hybrid import TOP_K, _ranked_item_lists, _score_lookup
from recbole.utils.evaluation import CALIBRATION_ALPHA, genre_distribution

PRIMARY_METRIC = "NDCG@10"   # same as valid_metric in configs/base.yaml
MAX_ACCURACY_LOSS = 0.05     # tuning: accept at most this relative drop in the primary metric vs lambda = 0
RERANKER_NAMES = ("MMR", "Calibration", "Fairness", "Popularity")
EXPOSURE_TARGETS = ("equity", "parity")
HYBRID_BASES = ("MixedHybrid", "HybridRegression", "HybridFeatureAug")  # base models that are not run by run_model.py

ItemGenres = Mapping[str, Set[str]]


def output_name(reranker_name: str, base: str = None) -> str:
    """Directory under outputs/ for a reranker's lists: Rerank_MMR, or Rerank_MMR/RecVAE for a given base model."""
    name = f"Rerank_{reranker_name}"
    return name if base is None else os.path.join(name, base)


def base_relevance_frame(recs: pd.DataFrame) -> pd.DataFrame:
    """The base list with a score the greedy loop can normalise.

    A list-mixing hybrid (frame with a `source` column) carries each source model's own score, which are
    on different scales, so its order is the only usable relevance: score = -rank. Other frames are returned as is.
    """
    if "source" in recs.columns:
        return recs.assign(score=-recs["rank"].astype(float))
    return recs


# --------------------------------------------------------------------------- #
# Rerankers
# --------------------------------------------------------------------------- #
class Reranker:
    """`objective(user_id, item_id, chosen)` scores a candidate given the items chosen so far.

    `target` is the metric (as named by Evaluator.evaluate, without the @k) the reranker is meant
    to improve, and `higher_is_better` its direction; tune_rerankers.py selects lambda with them.
    `begin_user` is called once per user before the greedy loop (for per-user state) and
    `lambda_scale` lets a reranker apply a different lambda to different users.
    """

    name = ""
    target = ""
    higher_is_better = True

    def begin_user(self, user_id: str, items: Sequence[str], relevance: np.ndarray, top_k: int) -> None:
        pass

    def lambda_scale(self, user_id: str) -> float:
        return 1.0

    def objective(self, user_id: str, item_id: str, chosen: Sequence[str]) -> float:
        raise NotImplementedError


def genre_similarity(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity of two genre sets (1 - the distance used by intra_list_diversity)."""
    union = a | b
    return len(a & b) / len(union) if union else 1.0


class MMR(Reranker):
    """Maximal Marginal Relevance (Carbonell & Goldstein, 1998) with genre Jaccard similarity."""

    name, target, higher_is_better = "MMR", "ILD", True

    def __init__(self, item_genres: ItemGenres):
        self.item_genres = item_genres

    def objective(self, user_id, item_id, chosen):
        if not chosen:
            return 1.0  # the same for every candidate, so the first pick is by relevance alone
        genres = self.item_genres.get(item_id, set())
        return 1.0 - max(genre_similarity(genres, self.item_genres.get(j, set())) for j in chosen)


class Calibration(Reranker):
    """Calibrated recommendations (Steck, 2018): greedy on (1 - lambda) * score - lambda * KL(p || q).

    p is the genre distribution of the user's training history, q that of the list being built;
    both as in recbole.utils.evaluation (each item spreads weight 1 evenly over its genres, and q is
    smoothed with alpha * p so the divergence stays finite).
    """

    name, target, higher_is_better = "Calibration", "Miscalibration", False

    def __init__(self, item_genres: ItemGenres, history: Mapping[str, Sequence[str]], alpha: float = CALIBRATION_ALPHA):
        self.alpha = alpha
        self.profile = {user: genre_distribution(items, item_genres) for user, items in history.items() if items}
        self.weights = {item: {g: 1.0 / len(genres) for g in genres} for item, genres in item_genres.items() if genres}

    def objective(self, user_id, item_id, chosen):
        p = self.profile.get(user_id)
        if not p:
            return 0.0
        q = Counter()
        for item in (*chosen, item_id):
            q.update(self.weights.get(item, {}))
        total = sum(q.values())
        if total == 0:
            return 0.0
        kl = 0.0
        for g, pg in p.items():
            q_smoothed = (1 - self.alpha) * q.get(g, 0.0) / total + self.alpha * pg
            kl += pg * math.log2(pg / q_smoothed)
        return -kl


class Fairness(Reranker):
    """Two-sided fairness: exposure for long-tail items, paid for by the advantaged users.

    Item side (Mehrotra et al., 2018; Singh & Joachims, 2018). Items are split into the short head
    (the 20% most popular, as in the Evaluator) and the long tail, and each user's top-k gets an
    exposure quota per group. `exposure_target` = "equity" makes the quotas proportional to the
    groups' relevance in the user's candidate list (sum of normalised scores: equity of attention),
    "parity" gives both groups half of the list (demographic parity). A candidate's objective is the
    unfilled share of its group's quota, so lambda = 1 enforces the quota as a hard constraint and
    lambda < 1 trades it off against relevance (Mehrotra's weighted greedy).

    User side (Li et al., 2021). Inactive users (the 80% with the fewest training interactions)
    already receive worse recommendations, so their lists are reranked with lambda * inactive_weight:
    with inactive_weight = 0 they keep the base list and only the active users pay for the extra tail
    exposure, which also narrows the active-inactive accuracy gap.
    """

    name, target, higher_is_better = "Fairness", "TailExposure", True

    def __init__(self, short_head: Iterable[str], active_users: Iterable[str],
                 exposure_target: str = "equity", inactive_weight: float = 1.0):
        self.short_head = set(short_head)
        self.active_users = set(active_users)
        self.exposure_target = exposure_target
        self.inactive_weight = inactive_weight
        self._quota: Dict[str, float] = {}

    def lambda_scale(self, user_id):
        return 1.0 if user_id in self.active_users else self.inactive_weight

    def tail_share(self, items: Sequence[str], relevance: np.ndarray) -> float:
        """Target share of the list for long-tail items."""
        if self.exposure_target == "parity":
            return 0.5
        if self.exposure_target != "equity":
            raise ValueError(f"exposure_target must be one of {EXPOSURE_TARGETS}")
        is_tail = np.array([item not in self.short_head for item in items], dtype=bool)
        total = float(relevance.sum())
        return float(relevance[is_tail].sum() / total) if total > 0 else float(is_tail.mean())

    def begin_user(self, user_id, items, relevance, top_k):
        tail = self.tail_share(items, relevance)
        self._quota = {"tail": tail * top_k, "head": (1.0 - tail) * top_k}

    def objective(self, user_id, item_id, chosen):
        in_head = item_id in self.short_head
        quota = self._quota["head" if in_head else "tail"]
        if quota <= 0:
            return 0.0
        filled = sum((j in self.short_head) == in_head for j in chosen)
        return max(0.0, 1.0 - filled / quota)


class PopularityDiscount(Reranker):
    """Favour long-tail items: objective is 1 for an item nobody interacted with, 0 for the most popular one."""

    name, target, higher_is_better = "Popularity", "APLT", True

    def __init__(self, item_popularity: Mapping[str, int]):
        self.item_popularity = item_popularity
        self.log_max = math.log(max(item_popularity.values(), default=0) + 1) or 1.0

    def objective(self, user_id, item_id, chosen):
        return 1.0 - math.log(self.item_popularity.get(item_id, 0) + 1) / self.log_max


def build_rerankers(train: pd.DataFrame, item_genres: ItemGenres, short_head: Iterable[str],
                    active_users: Iterable[str]) -> Dict[str, Reranker]:
    """All rerankers. Histories and popularity come from the training split, the head/tail and
    active/inactive groups from the Evaluator (`evaluator.short_head`, `evaluator.active_users`)."""
    history = train.groupby("user_id")["item_id"].apply(list).to_dict()
    popularity = train["item_id"].value_counts().to_dict()
    rerankers = [MMR(item_genres), Calibration(item_genres, history),
                 Fairness(short_head, active_users), PopularityDiscount(popularity)]
    return {r.name: r for r in rerankers}


# --------------------------------------------------------------------------- #
# Greedy reranking
# --------------------------------------------------------------------------- #
def _relevance(scores: Sequence[float]) -> np.ndarray:
    """Scores min-max normalised to [0, 1]; NaN counts as the lowest score, equal scores all give 1."""
    s = np.asarray(scores, dtype=float)
    finite = np.isfinite(s)
    if not finite.any():
        return np.ones(len(s))
    s = np.where(finite, s, s[finite].min())
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else np.ones(len(s))


def rerank_user(items: Sequence[str], scores: Sequence[float], reranker: Reranker, lam: float, user_id: str,
                top_k: int = TOP_K) -> List[int]:
    """Greedy top_k over one user's candidates (given in the model's order). Returns indices into `items`."""
    lam = lam * reranker.lambda_scale(user_id)
    if lam == 0.0:
        return list(range(min(top_k, len(items))))
    relevance = _relevance(scores)
    reranker.begin_user(user_id, items, relevance, top_k)
    remaining = list(range(len(items)))
    chosen_idx: List[int] = []
    chosen_items: List[str] = []
    while remaining and len(chosen_idx) < top_k:
        best, best_value = None, -math.inf
        for idx in remaining:  # ascending model rank, so a tie keeps the model's order
            value = (1 - lam) * relevance[idx] + lam * reranker.objective(user_id, items[idx], chosen_items)
            if value > best_value:
                best, best_value = idx, value
        remaining.remove(best)
        chosen_idx.append(best)
        chosen_items.append(items[best])
    return chosen_idx


def rerank(frame: pd.DataFrame, reranker: Reranker, lam: float, top_k: int = TOP_K) -> pd.DataFrame:
    """Rerank every user's candidate list.

    Returns user_id, item_id, rank, score, base_rank; score is the base model's own score and
    base_rank the item's position in the base model's list.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must be in [0, 1]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    lists = _ranked_item_lists(frame)
    scores = _score_lookup(frame)
    rows = []
    for user_id, items in lists.items():
        user_scores = scores.get(user_id, {})
        item_scores = [user_scores.get(item, float("nan")) for item in items]
        for rank, idx in enumerate(rerank_user(items, item_scores, reranker, lam, user_id, top_k), start=1):
            rows.append({"user_id": user_id, "item_id": items[idx], "rank": rank,
                         "score": item_scores[idx], "base_rank": idx + 1})
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rank", "score", "base_rank"])


# --------------------------------------------------------------------------- #
# Selection of lambda (tuning)
# --------------------------------------------------------------------------- #
def select_lambda(table: pd.DataFrame, target: str, higher_is_better: bool, primary: str = PRIMARY_METRIC,
                  max_loss: float = MAX_ACCURACY_LOSS, no_worse: Sequence[Tuple[str, bool]] = ()) -> pd.Series:
    """The row with the best `target` among the settings that keep `primary` within max_loss (relative)
    of the lambda = 0 row and leave every `no_worse` metric (name, higher_is_better) no worse than there.

    Selecting on accuracy alone would always return lambda = 0, so the accuracy loss is a constraint
    and the beyond-accuracy target the objective. Ties go to the smallest lambda, then (if the table
    has an `inactive_weight` column) to the largest weight, i.e. the most equal treatment of users.
    The lambda = 0 row always satisfies the constraints, so lambda = 0 is returned (the reranker is not
    applied) when no other setting fits the budget.
    """
    ref, allowed = _admissible(table, primary, max_loss, no_worse)
    if allowed.empty:
        raise ValueError(f"no admissible setting for {target}: is {primary} NaN in the lambda = 0 row?")
    keys, ascending = [target, "lambda"], [not higher_is_better, True]
    if "inactive_weight" in table.columns:
        keys.append("inactive_weight")
        ascending.append(False)
    return allowed.sort_values(keys, ascending=ascending, kind="mergesort").iloc[0]


def _admissible(table: pd.DataFrame, primary: str, max_loss: float,
                no_worse: Sequence[Tuple[str, bool]]) -> Tuple[pd.Series, pd.DataFrame]:
    """The lambda = 0 row and the rows within `max_loss` of it on `primary` and no worse on `no_worse`."""
    reference = table[table["lambda"] == 0.0]
    if reference.empty:
        raise ValueError("the table needs a lambda = 0 row as the accuracy reference")
    ref = reference.iloc[0]
    allowed = table[table[primary] >= (1 - max_loss) * float(ref[primary])]
    for metric, better in no_worse:
        allowed = allowed[allowed[metric] >= ref[metric]] if better else allowed[allowed[metric] <= ref[metric]]
    return ref, allowed


def budget_sensitivity(table: pd.DataFrame, target: str, higher_is_better: bool, budgets: Sequence[float],
                       primary: str = PRIMARY_METRIC, no_worse: Sequence[Tuple[str, bool]] = ()) -> pd.DataFrame:
    """`select_lambda` for several accuracy budgets: what each budget buys on the target.

    One row per budget: max_loss, lambda, inactive_weight (NaN without that column), the primary and target
    values of the selected setting, target_gain_pct (relative change of the target vs lambda = 0, signed so
    that positive = better) and admissible (number of settings within the budget).
    """
    rows = []
    for budget in budgets:
        ref, allowed = _admissible(table, primary, budget, no_worse)
        best = select_lambda(table, target, higher_is_better, primary, budget, no_worse)
        sign = 1.0 if higher_is_better else -1.0
        gain = sign * 100.0 * (best[target] - ref[target]) / abs(ref[target]) if ref[target] else float("nan")
        rows.append({
            "max_loss": budget, "lambda": float(best["lambda"]),
            "inactive_weight": float(best["inactive_weight"]) if "inactive_weight" in table.columns else float("nan"),
            primary: float(best[primary]), target: float(best[target]),
            "target_gain_pct": float(gain), "admissible": int(len(allowed)),
        })
    return pd.DataFrame(rows)
