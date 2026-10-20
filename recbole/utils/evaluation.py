import math
import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from recbole.utils.experiment import REPO_DIR, load_split

HEAD_SHARE = 0.2            # share of items in the short head / users in the active group
CALIBRATION_ALPHA = 0.01    # smoothing of the recommended genre distribution (Steck, 2018)
TAIL_SHARE = 0.5            # popularity groups: head = top 20%, tail = bottom 50%, middle = the rest
AGE_BINS = [0, 25, 35, 45, 200]
AGE_LABELS = ["<25", "25-34", "35-44", "45+"]
GROUP_ORDER = {  # natural order of the group labels, for tables and plots
    "activity": ["low", "medium", "high"],
    "mainstreaminess": ["niche", "average", "mainstream"],
    "age": AGE_LABELS,
    "popularity": ["head", "middle", "tail"],
}


# --------------------------------------------------------------------------- #
# Accuracy (per user, then averaged)
# --------------------------------------------------------------------------- #
def precision_at_k(ranked, relevant, k):
    """Share of the top-k that is relevant."""
    return sum(i in relevant for i in ranked[:k]) / k


def recall_at_k(ranked, relevant, k):
    """Share of the relevant items that appear in the top-k."""
    return sum(i in relevant for i in ranked[:k]) / len(relevant)


def hit_at_k(ranked, relevant, k):
    """1 if at least one relevant item is in the top-k."""
    return float(any(i in relevant for i in ranked[:k]))


def mrr_at_k(ranked, relevant, k):
    """1 / rank of the first relevant item in the top-k (0 if none)."""
    for rank, item in enumerate(ranked[:k], 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    """DCG of the top-k divided by the best possible DCG (all relevant items ranked first)."""
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(ranked[:k], 1) if item in relevant)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / idcg


def f1_at_k(ranked, relevant, k):
    """Harmonic mean of precision@k and recall@k (0 if both are 0)."""
    p, r = precision_at_k(ranked, relevant, k), recall_at_k(ranked, relevant, k)
    return 2 * p * r / (p + r) if p + r else 0.0


def average_precision_at_k(ranked, relevant, k, normalize="hits"):
    """Sum of precision@j over the ranks j of the relevant items in the top-k, divided by:

    - "hits" (default, as in the lecture example): the number of relevant items found in the top-k;
    - "min": min(|relevant|, k), RecBole's MAP, which also penalises relevant items that are missed.
    """
    hits, total = 0, 0.0
    for rank, item in enumerate(ranked[:k], 1):
        if item in relevant:
            hits += 1
            total += hits / rank
    denominator = hits if normalize == "hits" else min(len(relevant), k)
    return total / denominator if denominator else 0.0


def auc(scores, positive, candidate):
    """Probability that a random relevant item is scored above a random irrelevant one (ties count 1/2).

    Args:
        scores (array): the model's score for every item.
        positive, candidate (bool arrays): relevant items / items that may be recommended.
    Returns nan when the user has no relevant or no irrelevant candidate.
    """
    s, pos = scores[candidate], positive[candidate]
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)  # ascending, average rank for ties
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


ACCURACY_METRICS = {
    "Precision": precision_at_k,
    "Recall": recall_at_k,
    "F1": f1_at_k,
    "Hit": hit_at_k,
    "MRR": mrr_at_k,
    "MAP": average_precision_at_k,
    "NDCG": ndcg_at_k,
}


# --------------------------------------------------------------------------- #
# Beyond accuracy
# --------------------------------------------------------------------------- #
def intra_list_diversity(ranked, item_genres):
    """Average pairwise genre distance (1 - Jaccard) between the items of one list.

    0 = every item has the same genres, 1 = no two items share a genre.
    """
    if len(ranked) < 2:
        return 0.0
    total, pairs = 0.0, 0
    for a in range(len(ranked)):
        ga = item_genres[ranked[a]]
        for b in range(a + 1, len(ranked)):
            gb = item_genres[ranked[b]]
            total += 1.0 - len(ga & gb) / len(ga | gb)
            pairs += 1
    return total / pairs


def genre_distribution(items, item_genres):
    """p(g | items): each item spreads weight 1 evenly over its genres (Steck, 2018)."""
    dist = Counter()
    for item in items:
        genres = item_genres[item]
        for g in genres:
            dist[g] += 1.0 / len(genres)
    total = sum(dist.values())
    return {g: w / total for g, w in dist.items()}


def miscalibration(history, ranked, item_genres, alpha=CALIBRATION_ALPHA):
    """KL(p || q~) between the genre mix of the user's history (p) and of the list (q).

    q~ = (1 - alpha) * q + alpha * p keeps the divergence finite when the list misses a genre.
    0 = perfectly calibrated; larger = the list's genre mix drifts further from the user's taste.
    """
    p = genre_distribution(history, item_genres)
    q = genre_distribution(ranked, item_genres)
    kl = 0.0
    for g, pg in p.items():
        q_smoothed = (1 - alpha) * q.get(g, 0.0) + alpha * pg
        kl += pg * math.log2(pg / q_smoothed)
    return kl


def novelty(ranked, item_popularity, n_users):
    """Mean self-information -log2(p(i)) of the list; p(i) = share of users who interacted with i.

    Uses add-one smoothing, so items without training interactions get a finite (maximal) value.
    nan for an empty list.
    """
    if not ranked:
        return float("nan")
    return float(np.mean([-math.log2((item_popularity.get(i, 0) + 1) / (n_users + 1)) for i in ranked]))


def gini(values):
    """Gini coefficient of non-negative values: 0 = perfectly equal, ~1 = all on one element."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
def load_item_genres(path=None):
    """{item_id: set of genres} from the item file's `class` column."""
    path = path or os.path.join(REPO_DIR, "dataset", "ml-100k", "ml-100k.item")
    items = pd.read_csv(path, sep="\t", dtype=str, quoting=3)  # quoting=3: quotes in descriptions are text
    items.columns = [c.split(":")[0] for c in items.columns]   # "class:token_seq" -> "class"
    return {item: set(genres.split()) for item, genres in zip(items["item_id"], items["class"])}


def load_user_demographics(path=None):
    """DataFrame user_id, age, gender, occupation from the ML-100K user file (ids as strings)."""
    path = path or os.path.join(REPO_DIR, "dataset", "ml-100k", "ml-100k.user")
    users = pd.read_csv(path, sep="\t", dtype=str)
    users.columns = [c.split(":")[0] for c in users.columns]
    users["age"] = users["age"].astype(int)
    return users[["user_id", "age", "gender", "occupation"]]


def _tertiles(values, labels):
    """{key: label} splitting a Series into three equal-size groups (ties broken by order)."""
    return pd.qcut(values.rank(method="first"), 3, labels=labels).astype(str).to_dict()


class Evaluator:
    """Computes all metrics for top-N lists against one ground-truth split.

    Args:
        train (DataFrame): training interactions (user_id, item_id); defines popularity, user activity
            and each user's history for calibration.
        truth (DataFrame): ground-truth interactions of the evaluated split (validation or test).
        item_genres (dict): item_id -> set of genres.
        catalog (iterable): all candidate items (for coverage, Gini and the short head).
        seen (DataFrame, optional): interactions hidden from the candidates when computing AUC
            (training for validation, training + validation for test). Defaults to `train`.
    """

    def __init__(self, train, truth, item_genres, catalog, seen=None):
        self.item_genres = item_genres
        self.catalog = sorted(set(catalog))
        self.n_users = train["user_id"].nunique()
        self.history = train.groupby("user_id")["item_id"].apply(list).to_dict()
        self.relevant = truth.groupby("user_id")["item_id"].apply(set).to_dict()
        seen = train if seen is None else seen
        self.seen = seen.groupby("user_id")["item_id"].apply(set).to_dict()

        popularity = train["item_id"].value_counts()
        self.item_popularity = popularity.to_dict()
        ranked_items = popularity.reindex(self.catalog, fill_value=0).sort_values(ascending=False, kind="stable")
        self.short_head = set(ranked_items.index[: int(round(HEAD_SHARE * len(self.catalog)))])

        activity = train["user_id"].value_counts()
        self.active_users = set(activity.index[: int(round(HEAD_SHARE * len(activity)))])

    @classmethod
    def from_outputs(cls, stage="test"):
        """Build from outputs/split (written by run_model.py) and the ML-100K item file."""
        train, valid, test = load_split()
        truth = {"valid": valid, "test": test}[stage]
        seen = {"valid": train, "test": pd.concat([train, valid])}[stage]
        catalog = pd.concat([train, valid, test])["item_id"].unique()
        return cls(train, truth, load_item_genres(), catalog, seen)

    def user_group(self, user):
        return "active" if user in self.active_users else "inactive"

    def item_group(self, item):
        return "head" if item in self.short_head else "tail"

    def per_user(self, recs, k=10):
        """One row per evaluated user with every per-user metric (useful for group analyses)."""
        lists = (
            recs.sort_values(["user_id", "rank"], kind="stable")
            .groupby("user_id")["item_id"].apply(lambda s: list(s)[:k]).to_dict()
        )
        rows = []
        for user, relevant in self.relevant.items():
            ranked = lists.get(user, [])
            history = self.history.get(user, [])
            row = {"user_id": user, "user_group": self.user_group(user)}
            row.update({f"{name}@{k}": fn(ranked, relevant, k) for name, fn in ACCURACY_METRICS.items()})
            profile_pop = np.mean([self.item_popularity.get(i, 0) for i in history]) if history else 0.0
            # A user without recommendations gets nan for the list averages instead of a numpy warning.
            arp = np.mean([self.item_popularity.get(i, 0) for i in ranked]) if ranked else float("nan")
            row.update({
                f"ILD@{k}": intra_list_diversity(ranked, self.item_genres),
                f"Novelty@{k}": novelty(ranked, self.item_popularity, self.n_users),
                f"ARP@{k}": arp,
                f"APLT@{k}": np.mean([i not in self.short_head for i in ranked]) if ranked else float("nan"),
                f"PopLift@{k}": (arp - profile_pop) / profile_pop if profile_pop else 0.0,
                f"Miscalibration@{k}": miscalibration(history, ranked, self.item_genres) if history else 0.0,
            })
            rows.append(row)
        return pd.DataFrame(rows), lists

    def per_user_auc(self, scores, user_ids, item_ids):
        """{user: AUC} from a full score matrix (rows user_ids, columns item_ids, as in scores.npz).

        Candidates are all items the user has not seen yet, as in RecBole's full-sort evaluation.
        """
        column = {item: j for j, item in enumerate(item_ids)}
        result = {}
        for row, user in enumerate(user_ids):
            relevant = self.relevant.get(user)
            if not relevant:
                continue
            positive = np.zeros(len(item_ids), dtype=bool)
            positive[[column[i] for i in relevant if i in column]] = True
            candidate = np.ones(len(item_ids), dtype=bool)
            candidate[[column[i] for i in self.seen.get(user, ()) if i in column]] = False
            result[user] = auc(scores[row], positive, candidate)
        return result

    def evaluate(self, recs, k=10, scores=None):
        """All metrics for one model: per-user metrics averaged, plus list-level and fairness metrics.

        AUC needs a score for every item, so it is only computed when `scores` = (matrix, user_ids, item_ids)
        is given (e.g. from load_scores); a top-k list alone is not enough.
        """
        users, lists = self.per_user(recs, k)
        result = users.drop(columns=["user_id", "user_group"]).mean().to_dict()
        if scores is not None:
            result["AUC"] = float(np.nanmean(list(self.per_user_auc(*scores).values())))

        # Aggregate (system-level) beyond-accuracy metrics, over the lists of all evaluated users.
        exposure = Counter(i for u in self.relevant for i in lists.get(u, []))
        result[f"Coverage@{k}"] = len(exposure) / len(self.catalog)
        result[f"Gini@{k}"] = gini([exposure.get(i, 0) for i in self.catalog])
        result[f"TailExposure@{k}"] = sum(c for i, c in exposure.items() if i not in self.short_head) / sum(exposure.values())

        # User-side fairness: accuracy gap between active and inactive users (Li et al., 2021).
        by_group = users.groupby("user_group")[f"NDCG@{k}"].mean()
        result[f"NDCG@{k}_active"] = by_group.get("active", float("nan"))
        result[f"NDCG@{k}_inactive"] = by_group.get("inactive", float("nan"))
        result[f"UserNDCGGap@{k}"] = result[f"NDCG@{k}_active"] - result[f"NDCG@{k}_inactive"]

        # Item-side fairness: share of the hits that go to long-tail items.
        hits = Counter(self.item_group(i) for u, rel in self.relevant.items() for i in lists.get(u, []) if i in rel)
        result[f"TailHitShare@{k}"] = hits["tail"] / max(sum(hits.values()), 1)
        return result

    def group_breakdown(self, recs, k=10):
        """Per user group (active/inactive): every per-user metric averaged within the group (Task 2.5)."""
        users, _ = self.per_user(recs, k)
        table = users.drop(columns=["user_id"]).groupby("user_group").mean()
        table.insert(0, "n_users", users["user_group"].value_counts())
        return table

    # ----------------------------------------------------------------------- #
    # Group analysis (Task 2.5)
    # ----------------------------------------------------------------------- #
    def user_groupings(self, demographics=None):
        """{grouping name: {user_id: group label}}.

        activity        tertiles of the number of training interactions (low / medium / high)
        mainstreaminess tertiles of the mean popularity of the user's training items (niche / average / mainstream)
        gender, age, occupation   from the ML-100K user file, if `demographics` is given
        """
        activity = pd.Series({u: len(h) for u, h in self.history.items()})
        taste = pd.Series({u: np.mean([self.item_popularity.get(i, 0) for i in h]) for u, h in self.history.items()})
        groupings = {
            "activity": _tertiles(activity, ["low", "medium", "high"]),
            "mainstreaminess": _tertiles(taste, ["niche", "average", "mainstream"]),
        }
        if demographics is not None:
            d = demographics.set_index("user_id")
            groupings["gender"] = d["gender"].to_dict()
            groupings["age"] = pd.cut(d["age"], AGE_BINS, right=False, labels=AGE_LABELS).astype(str).to_dict()
            groupings["occupation"] = d["occupation"].to_dict()
        return groupings

    def item_groupings(self):
        """{grouping name: {item_id: list of group labels}} (an item can be in several genres).

        popularity  head = top 20% by training interactions (the short head), tail = bottom 50%, middle = rest
        genre       every genre of the movie
        """
        ranked = (pd.Series(self.item_popularity).reindex(self.catalog, fill_value=0)
                  .sort_values(ascending=False, kind="stable").index)
        n_head = int(round(HEAD_SHARE * len(ranked)))
        n_tail = int(round(TAIL_SHARE * len(ranked)))
        popularity = {i: ["head"] for i in ranked[:n_head]}
        popularity.update({i: ["middle"] for i in ranked[n_head:len(ranked) - n_tail]})
        popularity.update({i: ["tail"] for i in ranked[len(ranked) - n_tail:]})
        genre = {i: sorted(self.item_genres.get(i, {"unknown"})) for i in self.catalog}
        return {"popularity": popularity, "genre": genre}

    def user_group_table(self, recs, k=10, groupings=None):
        """Every per-user metric averaged within each user group; one row per (grouping, group).

        avg_test_items and avg_train_items are reported because accuracy@k rises with the number of relevant
        items a user has, so groups that differ in them (e.g. niche users are mostly heavy users) are not
        directly comparable. NDCG@k_se is the standard error of the group's mean NDCG, to judge whether a gap
        between groups is larger than noise (it is nan for a group of one user).
        Raises ValueError if an evaluated user has no label in a grouping, instead of silently dropping them.
        """
        users, _ = self.per_user(recs, k)
        metrics = users.drop(columns=["user_id", "user_group"])
        metrics.insert(0, "avg_test_items", users["user_id"].map(lambda u: len(self.relevant[u])))
        metrics.insert(1, "avg_train_items", users["user_id"].map(lambda u: len(self.history.get(u, []))))
        ndcg = f"NDCG@{k}"
        rows = []
        for name, labels in (groupings or self.user_groupings()).items():
            group = users["user_id"].map(labels)
            if group.isna().any():
                raise ValueError(f"{int(group.isna().sum())} evaluated users have no label for grouping {name!r}")
            for label, part in metrics.groupby(group):
                values = part.mean().to_dict()
                values[f"{ndcg}_se"] = part[ndcg].std(ddof=1) / np.sqrt(len(part))
                rows.append({"grouping": name, "group": label, "n_users": len(part), **values})
        return pd.DataFrame(rows)

    def item_group_table(self, recs, k=10, groupings=None):
        """How each item group is treated; one row per (grouping, group).

        catalog_share   share of catalogue items in the group
        relevant_share  share of all relevant (test) interactions that involve the group
        exposure_share  share of all recommendation slots filled with items of the group
        hit_share       share of all hits that are items of the group
        recall          hits on the group / relevant interactions with the group (summed over users)
        coverage        share of the group's items recommended to at least one user
        Genres overlap, so their shares sum to more than 1.
        """
        _, lists = self.per_user(recs, k)
        lists = {u: lists.get(u, []) for u in self.relevant}
        n_slots = sum(len(lst) for lst in lists.values())
        n_relevant = sum(len(rel) for rel in self.relevant.values())
        n_hits = sum(i in self.relevant[u] for u, lst in lists.items() for i in lst)
        rows = []
        for name, labels in (groupings or self.item_groupings()).items():
            members = {}
            for item, groups in labels.items():
                for g in groups:
                    members.setdefault(g, set()).add(item)
            for label, items in sorted(members.items()):
                relevant = sum(len(rel & items) for rel in self.relevant.values())
                exposed = [i for lst in lists.values() for i in lst if i in items]
                hits = sum(i in self.relevant[u] for u, lst in lists.items() for i in lst if i in items)
                rows.append({
                    "grouping": name, "group": label, "n_items": len(items),
                    "catalog_share": len(items) / len(self.catalog) if self.catalog else float("nan"),
                    "relevant_share": relevant / n_relevant if n_relevant else float("nan"),
                    "exposure_share": len(exposed) / n_slots if n_slots else 0.0,
                    "hit_share": hits / n_hits if n_hits else 0.0,
                    f"recall@{k}": hits / relevant if relevant else float("nan"),
                    f"coverage@{k}": len(set(exposed)) / len(items) if items else float("nan"),
                })
        return pd.DataFrame(rows)
