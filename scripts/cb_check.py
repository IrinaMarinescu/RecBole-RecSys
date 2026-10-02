import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)  # so recbole is importable without `pip install -e .`

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, init_seed

config = Config(
    model="CB",
    dataset="ml-100k",
    config_file_list=[os.path.join(root, "configs", "cb_rating.yaml")],
    config_dict={"show_progress": False, "save_dataset": False},
)  # extra --key=value arguments on the command line override these
init_seed(config["seed"], config["reproducibility"])

dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)
model = get_model("CB")(config, train_data.dataset).to(config["device"])

print("\n=== config ===")
for key in ["content", "pooling_strategy", "aggregation_method", "cold_start_strategy",
            "similarity_metric", "score_mapping"]:
    print(f"{key}: {config[key]}")

print("\n=== item texts fed to BERT ===")
for item_idx in [1, 2, 50, 100]:
    token = dataset.id2token(model.ITEM_ID, item_idx)
    print(f"item {token}: {model.item_texts[item_idx]!r}")

print("\n=== embeddings ===")
print("item embeddings:", tuple(model.item_embeddings.shape))
print("user embeddings:", tuple(model.user_embeddings.shape))
zero_users = (model.user_embeddings[1:].abs().sum(dim=1) == 0).sum().item()
print("users with an all-zero (cold-start) profile:", zero_users)

if model.score_mapping:
    print("\n=== score mapping ===")
    print(f"similarity range [{model.sim_min:.4f}, {model.sim_max:.4f}]"
          f" -> rating range [{model.rating_min}, {model.rating_max}]")

print("\n=== sample test predictions ===")
batch = next(iter(test_data))[0].to(config["device"])
preds = model.predict(batch)
for user, item, rating, pred in list(zip(batch[model.USER_ID], batch[model.ITEM_ID],
                                         batch[model.RATING], preds))[:10]:
    print(f"user {dataset.id2token(model.USER_ID, user.item()):>4}"
          f"  item {dataset.id2token(model.ITEM_ID, item.item()):>4}"
          f"  true {rating.item():.1f}  pred {pred.item():.2f}")
