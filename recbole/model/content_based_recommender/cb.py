from recbole.model.abstract_recommender import ContentBasedRecommender
from recbole.utils import InputType, FeatureType
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np


class CB(ContentBasedRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(CB, self).__init__(config, dataset)

        # dummy parameter so the trainer's optimizer has something to hold (the model itself is not trained)
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

        # Extract content based on the fields selected in config
        self.content_fields = self.content
        self.item_texts = self._extract_item_content(dataset)

        # build item embeddings
        self.register_buffer("item_embeddings", self._build_item_embeddings())

        # build user embeddings
        self.register_buffer("user_embeddings", self._build_user_embeddings(dataset))

        # items each user interacted with in training, used when filter_interacted is True
        history_item, _, _ = dataset.history_item_matrix()
        self.register_buffer("history_item", history_item.long())

        # range used to map similarity scores into the rating scale
        if self.score_mapping:
            self._init_score_mapping(dataset)

    def _extract_item_content(self, dataset):
        """prepares the content according to the config option"""
        texts = [""]  # Index 0 is the padding item
        item_feat = dataset.item_feat
        fields = [f for f in self.content_fields if item_feat is not None and f in item_feat]

        for item_idx in range(1, self.n_items):
            item_text_parts = []
            for field in fields:
                val = item_feat[field][item_idx]
                field_type = dataset.field2type[field]

                if field_type == FeatureType.TOKEN:
                    tokens = [dataset.id2token(field, int(val))] if int(val) != 0 else []
                elif field_type == FeatureType.TOKEN_SEQ:
                    ids = val[val != 0].cpu().numpy()
                    tokens = list(dataset.id2token(field, ids)) if len(ids) > 0 else []
                elif field_type == FeatureType.FLOAT:
                    tokens = [str(val.item())]
                else:  # FLOAT_SEQ
                    tokens = [str(v) for v in val.tolist()]

                text_val = " ".join(str(t) for t in tokens if str(t) != "[PAD]")
                if text_val:
                    item_text_parts.append(text_val)

            texts.append(" ".join(item_text_parts))

        return texts

    def _build_item_embeddings(self):
        """creates fixed item embedding vectors using BERT, according to the configured pooling strategy"""
        tokenizer = AutoTokenizer.from_pretrained(self.bert_model)
        model = AutoModel.from_pretrained(self.bert_model).to(self.device)
        model.eval()

        embeddings = []
        pooling_strategy = str(self.pooling_strategy or "cls").lower()

        with torch.no_grad():
            for i in range(0, len(self.item_texts), self.batch_size):
                batch_texts = self.item_texts[i: i + self.batch_size]
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                ).to(self.device)

                outputs = model(**inputs)
                last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
                attention_mask = inputs["attention_mask"]  # [batch_size, seq_len]

                if pooling_strategy == "cls":
                    batch_embeds = last_hidden_state[:, 0, :]
                elif pooling_strategy == "mean":
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
                    batch_embeds = sum_embeddings / sum_mask
                elif pooling_strategy == "max":
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
                    last_hidden_state_masked = last_hidden_state.masked_fill(input_mask_expanded == 0, -1e9)
                    batch_embeds = torch.max(last_hidden_state_masked, dim=1)[0]
                else:
                    raise ValueError(
                        f"Unsupported pooling strategy: '{pooling_strategy}'. Expected one of ['cls', 'mean', 'max']."
                    )

                embeddings.append(batch_embeds.cpu())

        # Free transformer
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        item_embeddings = torch.cat(embeddings, dim=0).to(self.device)
        item_embeddings[0] = 0 #padding
        return item_embeddings

    def _build_user_embeddings(self, dataset):
        """aggregates item embeddings into user_embeddings, according to the config"""
        inter_feat = dataset.inter_feat
        user_ids = self._to_numpy(inter_feat[self.USER_ID])
        item_ids = self._to_numpy(inter_feat[self.ITEM_ID])
        ratings = self._to_numpy(inter_feat[self.RATING]) if self.RATING in inter_feat else None

        embed_dim = self.item_embeddings.shape[1]
        cold_start_embedding = self._cold_start_embedding(item_ids)

        user_embeddings = torch.zeros((self.n_users, embed_dim), device=self.device)

        # group interactions by user once instead of scanning all interactions per user
        order = np.argsort(user_ids, kind="stable")
        uniq_users, starts = np.unique(user_ids[order], return_index=True)
        user_groups = dict(zip(uniq_users.tolist(), np.split(order, starts[1:])))

        for user_idx in range(self.n_users):
            idx = user_groups.get(user_idx)

            if idx is None or len(idx) == 0:
                user_embeddings[user_idx] = cold_start_embedding
                continue

            hist_item_embeds = self.item_embeddings[torch.as_tensor(item_ids[idx], device=self.device)]
            hist_ratings = ratings[idx] if ratings is not None else None

            if self.aggregation_method == "avg":
                user_embeddings[user_idx] = hist_item_embeds.mean(dim=0)
            elif self.aggregation_method == "weighted_avg":
                if hist_ratings is not None:
                    weights = torch.tensor(hist_ratings, dtype=torch.float, device=self.device)
                else:
                    weights = torch.ones(len(idx), device=self.device)
                weights = weights / torch.clamp(weights.sum(), min=1e-9)
                user_embeddings[user_idx] = (hist_item_embeds * weights.unsqueeze(-1)).sum(dim=0)
            elif self.aggregation_method == "avg_pos":
                if hist_ratings is not None:
                    pos_mask = torch.as_tensor(hist_ratings >= self.pos_rating_threshold, device=self.device)
                    pos_item_embeds = hist_item_embeds[pos_mask]
                else:
                    pos_item_embeds = hist_item_embeds

                # no positively rated items -> fall back to the plain average
                if len(pos_item_embeds) == 0:
                    pos_item_embeds = hist_item_embeds
                user_embeddings[user_idx] = pos_item_embeds.mean(dim=0)
            else:
                raise ValueError(
                    f"Unsupported aggregation method: '{self.aggregation_method}'. "
                    f"Expected one of ['avg', 'weighted_avg', 'avg_pos']."
                )

        return user_embeddings

    def _cold_start_embedding(self, item_ids):
        """returns the fallback embedding, according to the config"""
        item_embeds = self.item_embeddings[1:]  # skip padding item
        if self.cold_start_strategy == "zero":
            return torch.zeros(self.item_embeddings.shape[1], device=self.device)
        elif self.cold_start_strategy == "global_mean":
            return item_embeds.mean(dim=0)
        elif self.cold_start_strategy == "popularity":
            counts = np.bincount(item_ids, minlength=self.n_items)[1:]
            weights = torch.tensor(counts, dtype=torch.float, device=self.device)
            weights = weights / torch.clamp(weights.sum(), min=1e-9)
            return (item_embeds * weights.unsqueeze(-1)).sum(dim=0)
        else:
            raise ValueError(
                f"Unsupported cold start strategy: '{self.cold_start_strategy}'. "
                f"Expected one of ['zero', 'global_mean', 'popularity']."
            )

    def _init_score_mapping(self, dataset):
        """stores the min/max similarity over all user-item pairs and the rating range, for min-max mapping"""
        min_val, max_val = float("inf"), float("-inf")
        # skip the padding user/item (index 0), their zero embeddings would distort the range
        all_items = torch.arange(1, self.n_items, device=self.device)
        for start in range(1, self.n_users, self.batch_size):
            users = torch.arange(start, min(start + self.batch_size, self.n_users), device=self.device)
            scores = self._similarity(self.user_embeddings[users], self.item_embeddings[all_items], pairwise=False)
            min_val = min(min_val, scores.min().item())
            max_val = max(max_val, scores.max().item())
        self.sim_min, self.sim_max = min_val, max_val

        if self.RATING in dataset.inter_feat:
            ratings = dataset.inter_feat[self.RATING]
            self.rating_min, self.rating_max = ratings.min().item(), ratings.max().item()
        else:
            self.rating_min, self.rating_max = 1.0, 5.0

    def _similarity(self, user_e, item_e, pairwise=True):
        """pairwise=True: user_e[i] vs item_e[i] -> [B]; pairwise=False: all users vs all items -> [B, n_items]"""
        if self.similarity_metric == "cosine":
            user_e, item_e = F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)

        if self.similarity_metric in ("dot", "cosine"):
            return (user_e * item_e).sum(dim=-1) if pairwise else user_e @ item_e.T
        elif self.similarity_metric in ("euclidean", "euclidian"):
            # negative distance so that higher means more similar
            return -(user_e - item_e).norm(dim=-1) if pairwise else -torch.cdist(user_e, item_e)
        else:
            raise ValueError(
                f"Unsupported similarity metric: '{self.similarity_metric}'. "
                f"Expected one of ['dot', 'cosine', 'euclidean']."
            )

    def _map_scores(self, scores):
        """maps similarity scores into the rating scale, i.e. [1, 5] for ml-100k"""
        if not self.score_mapping:
            return scores
        if self.sim_max > self.sim_min:
            scale = (self.rating_max - self.rating_min) / (self.sim_max - self.sim_min)
            return self.rating_min + (scores - self.sim_min) * scale
        # every prediction identical -> fall back to the midpoint of the rating scale
        return torch.full_like(scores, (self.rating_min + self.rating_max) / 2)

    @staticmethod
    def _to_numpy(values):
        """converts tensor-like or array-like values into a numpy array"""
        if isinstance(values, torch.Tensor):
            return values.cpu().numpy()
        return np.array(values)

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        # embeddings are fixed, so basically nothing to learn
        return torch.nn.Parameter(torch.zeros(1)).to(self.device)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        scores = self._similarity(self.user_embeddings[user], self.item_embeddings[item])
        return self._map_scores(scores)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        scores = self._similarity(self.user_embeddings[user], self.item_embeddings, pairwise=False)
        scores = self._map_scores(scores)
        if self.filter_interacted:
            scores = scores.scatter(1, self.history_item[user], float("-inf"))
        return scores.view(-1)
