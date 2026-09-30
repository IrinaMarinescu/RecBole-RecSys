from recbole.model.abstract_recommender import ContentBasedRecommender
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np

class CB(ContentBasedRecommender):

    def __init__(self, config, dataset):
        super(CB, self).__init__(config, dataset)

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model)
        self.model = AutoModel.from_pretrained(self.bert_model).to(self.device)

        # Extract content based on the fields selected in config
        item_content = self._extract_item_content(dataset)

        if isinstance(item_content, pd.Series):
            item_content = item_content.fillna("").astype(str).tolist()
        elif isinstance(item_content, np.ndarray):
            item_content = item_content.astype(str).tolist()
        self.content = item_content

        # build item embeddings
        self.item_embeddings = self._build_item_embeddings()

        # build user embeddings
        self.user_embeddings = self._build_user_embeddings(dataset)

    def _extract_item_content(self, dataset):
        """prepares the content according to the config option"""
        texts = []
        item_feat = dataset.item_feat
        content_fields = self.content

        for item_idx in range(self.n_items):
            if item_idx == 0:
                # Index 0 has no item
                texts.append("")
                continue

            item_text_parts = []
            for field in content_fields:
                if field in item_feat:
                    val = item_feat[field][item_idx]

                    if isinstance(val, (list, np.ndarray, torch.Tensor)):
                        tokens = dataset.id2token(field, val) if hasattr(dataset, "id2token") else val
                        text_val = " ".join([str(t) for t in tokens if str(t) != "[PAD]"])
                    else:
                        text_val = str(val)

                    if text_val and text_val != "[PAD]":
                        item_text_parts.append(text_val)

            combined_text = " ".join(item_text_parts)
            texts.append(combined_text)

        return texts

    def _build_item_embeddings(self):
        """creates fixed item embedding vectors using BERT, according to the configured pooling strategy"""
        self.model.eval()
        embeddings = []
        pooling_strategy = self.pooling_strategy
        if pooling_strategy is None:
            pooling_strategy = "cls"
        pooling_strategy = str(pooling_strategy).lower()

        with torch.no_grad():
            for i in range(0, len(self.content), self.batch_size):
                batch_texts = self.content[i: i + self.batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)
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
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                    last_hidden_state_masked = last_hidden_state.clone()
                    last_hidden_state_masked[input_mask_expanded == 0] = -1e9
                    batch_embeds = torch.max(last_hidden_state_masked, dim=1)[0]
                else:
                    raise ValueError(
                        f"Unsupported pooling strategy: '{pooling_strategy}'. Expected one of ['cls', 'mean', 'max']."
                    )

                embeddings.append(batch_embeds.cpu())

        # Returns tensor of shape [n_items, hidden_size]
        item_embeddings = torch.cat(embeddings, dim=0).to(self.device)

        # Free transformer
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return item_embeddings

    def _build_user_embeddings(self, dataset):
        """aggregates item embeddings into user_embeddings, according to the config"""
        inter_feat = dataset.inter_feat
        user_ids = self._to_numpy(inter_feat[self.USER_ID])
        item_ids = self._to_numpy(inter_feat[self.ITEM_ID])
        ratings = self._to_numpy(inter_feat[self.RATING]) if self.RATING in inter_feat else None

        embed_dim = self.item_embeddings.shape[1]
        global_mean_embedding = self.item_embeddings.mean(dim=0)

        user_embeddings = torch.zeros((self.n_users, embed_dim), device=self.device)

        for user_idx in range(self.n_users):
            mask = user_ids == user_idx
            hist_item_ids = item_ids[mask]

            if len(hist_item_ids) == 0:
                user_embeddings[user_idx] = self._cold_start_embedding(global_mean_embedding)
                continue

            hist_item_embeds = self.item_embeddings[hist_item_ids]

            if self.aggregation_method == "avg":
                user_embeddings[user_idx] = hist_item_embeds.mean(dim=0)
            elif self.aggregation_method == "weighted_avg":
                if ratings is not None:
                    hist_ratings = torch.tensor(ratings[mask], dtype=torch.float, device=self.device)
                else:
                    hist_ratings = torch.ones(len(hist_item_ids), device=self.device)
                weight_sum = torch.clamp(hist_ratings.sum(), min=1e-9)
                weights = hist_ratings / weight_sum
                user_embeddings[user_idx] = (hist_item_embeds * weights.unsqueeze(-1)).sum(dim=0)
            elif self.aggregation_method == "avg_pos":
                if ratings is not None:
                    pos_mask = ratings[mask] >= self.pos_rating_threshold
                    pos_item_embeds = hist_item_embeds[pos_mask]
                else:
                    pos_item_embeds = hist_item_embeds

                if len(pos_item_embeds) == 0:
                    user_embeddings[user_idx] = self._cold_start_embedding(global_mean_embedding)
                else:
                    user_embeddings[user_idx] = pos_item_embeds.mean(dim=0)
            else:
                raise ValueError(
                    f"Unsupported aggregation method: '{self.aggregation_method}'. "
                    f"Expected one of ['avg', 'weighted_avg', 'avg_pos']."
                )

        return user_embeddings

    def _cold_start_embedding(self, global_mean_embedding):
        """returns the fallback embedding, according to the config"""
        if self.cold_start_strategy == "zero":
            return torch.zeros_like(global_mean_embedding)
        elif self.cold_start_strategy in ("global_mean", "popularity"):
            return global_mean_embedding
        else:
            return global_mean_embedding

    @staticmethod
    def _to_numpy(values):
        """converts tensor-like or array-like values into a numpy array"""
        if isinstance(values, torch.Tensor):
            return values.cpu().numpy()
        return np.array(values)

    def calculate_loss(self, interaction):
        ...

    def predict(self, interaction):
        ...

    def full_sort_predict(self, interaction):
        ...