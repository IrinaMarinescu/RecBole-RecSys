import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)  # so recbole is importable without `pip install -e .`

import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed

config = Config(
    model="CB",
    dataset="ml-100k",
    config_file_list=[os.path.join(root, "configs", "cb_ranking.yaml")],
    config_dict={"show_progress": False, "save_dataset": False},
)  # extra --key=value arguments on the command line override these
init_seed(config["seed"], config["reproducibility"])

dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)
model = get_model("CB")(config, train_data.dataset).to(config["device"])
model.eval()

print("\n=== config ===")
for key in ["content", "pooling_strategy", "aggregation_method", "cold_start_strategy",
            "similarity_metric", "score_mapping", "filter_interacted"]:
    print(f"{key}: {config[key]}")

print("\n=== embeddings ===")
print("item embeddings:", tuple(model.item_embeddings.shape))
print("user embeddings:", tuple(model.user_embeddings.shape))

# full-sort batches are (user interaction, history index, positive_u, positive_i)
interaction, _, positive_u, positive_i = next(iter(test_data))
interaction = interaction.to(config["device"])
users = interaction[model.USER_ID]
n_users, n_items = len(users), model.n_items

with torch.no_grad():
    scores = model.full_sort_predict(interaction).view(n_users, n_items)

print("\n=== full_sort_predict ===")
print("scores:", tuple(scores.shape))
assert scores.shape == (n_users, n_items), "full_sort_predict must return one score per (user, item)"
assert not torch.isnan(scores).any(), "scores contain NaN"
if not model.filter_interacted:
    assert torch.isfinite(scores).all(), "scores contain inf without filter_interacted"

# a positive item counts as a hit if it shows up in that user's top-k
k = config["topk"][0]
topk_items = scores[:, 1:].topk(k, dim=1).indices + 1  # skip the padding item
positives = torch.zeros_like(scores, dtype=torch.bool)
positives[positive_u.to(scores.device), positive_i.to(scores.device)] = True
hits = positives.gather(1, topk_items).sum(dim=1)
n_pos = positives.sum(dim=1).clamp(min=1)
print(f"mean Recall@{k} on this batch: {(hits / n_pos).float().mean().item():.4f}")

print(f"\n=== top-{k} for the first 3 users ===")
for row in range(min(3, n_users)):
    token = dataset.id2token(model.USER_ID, users[row].item())
    print(f"user {token} (hits {hits[row].item()}/{int(n_pos[row].item())}):")
    for item in topk_items[row].tolist():
        mark = "*" if positives[row, item] else " "
        print(f"  {mark} {scores[row, item].item():8.4f}  {model.item_texts[item]}")

print("\n=== filter_interacted ===")
original = model.filter_interacted
model.filter_interacted = True
with torch.no_grad():
    filtered = model.full_sort_predict(interaction).view(n_users, n_items)
model.filter_interacted = original
history = model.history_item[users]
seen = filtered.gather(1, history)
assert torch.isinf(seen).all() and (seen < 0).all(), "train items must be scored -inf when filtered"
print(f"all {int((history != 0).sum().item())} train interactions of this batch scored -inf")

print("\n=== full test evaluation ===")
trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
for metric, value in result.items():
    print(f"{metric}: {value:.4f}")

print("\nranking smoke test passed")
