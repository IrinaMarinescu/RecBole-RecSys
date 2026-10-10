import argparse
from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer
from recbole.model.general_recommender.hybrid_feature_aug import HybridFeatureAugmentation

import numpy as np
np.float = float
np.int = int
np.object = object
np.bool = bool


def evaluate(config, model, data, show_progress=False):
    trainer = Trainer(config, model)
    return trainer.evaluate(data, load_best_model=False, show_progress=show_progress)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_files', nargs='+', required=True,
                        help='Pre-trained .pth files in saved/ of the SOURCE models '
                             '(Recommender 1), e.g. BPR and NeuMF')
    parser.add_argument('--target_model', default='ItemKNN',
                        choices=HybridFeatureAugmentation.SUPPORTED_TARGETS,
                        help='RecBole model rebuilt on the augmented data (Recommender n)')
    args = parser.parse_args()

    source_models = []
    source_evals = []  # (config, test_data) of each source's own checkpoint
    config, dataset, train_data, valid_data, test_data = None, None, None, None, None

    # Load pre-trained source models and identically seeded dataset splits
    for file_path in args.model_files:
        print(f"Loading {file_path}...")
        cfg, model, ds, tr_data, val_data, te_data = load_data_and_model(file_path)
        source_models.append(model)
        source_evals.append((cfg, te_data))

        if config is None:
            config = cfg
            dataset = ds
            train_data = tr_data
            valid_data = val_data
            test_data = te_data

    print("\n--- Source Model Standalone Test Metrics ---")
    # Each source on its own test data: CB's dataset also holds the items without interactions,
    # so its scores have more columns than the first checkpoint's test data expects
    for model, (cfg, te_data) in zip(source_models, source_evals):
        test_result = evaluate(cfg, model, te_data)
        print(f"[{model.__class__.__name__}] Test Metrics: {test_result}")

    # Hyperparameters of the target model. Set them to the values in your target
    # model's .yaml so the n_pseudo=0 run equals your standalone target model.
    # Anything left out falls back to RecBole's default for that model.
    target_params = {}  # e.g. {'k': 100, 'shrink': 0.0} for ItemKNN

    # Hyperparameter tuning grid for the hybrid model
    n_pseudo_values = [0, 5, 10, 20, 50]   # 0 = plain target model (ablation)
    pseudo_weights = [0.25, 0.5, 1.0]

    hybrid_model = HybridFeatureAugmentation(
        config, train_data, source_models,
        target_model=args.target_model, target_params=target_params,
    )
    metric_key = config['valid_metric'].lower()
    best_score, best_params = -np.inf, None

    print(f"\n--- Tuning Feature Augmentation (target: {args.target_model}) ---")
    # Largest n_pseudo first, so source rankings are computed only once
    for n_pseudo in sorted(n_pseudo_values, reverse=True):
        for weight in (pseudo_weights if n_pseudo > 0 else pseudo_weights[:1]):
            hybrid_model.fit_augmentation(n_pseudo=n_pseudo, pseudo_weight=weight)
            val_result = evaluate(config, hybrid_model, valid_data)
            if metric_key not in val_result:
                metric_key = list(val_result.keys())[0]
            score = val_result[metric_key]

            label = 'plain target' if n_pseudo == 0 else f'pseudo_weight={weight}'
            print(f"n_pseudo={n_pseudo:<3} {label:<20} | Validation {metric_key}: {score:.4f}")

            if score > best_score:
                best_score, best_params = score, (n_pseudo, weight)

    print("\n--- Final Offline Evaluation on Test Set ---")
    n_pseudo, weight = best_params
    print(f"Best: n_pseudo={n_pseudo}, pseudo_weight={weight}")

    hybrid_model.fit_augmentation(n_pseudo=0)
    baseline = evaluate(config, hybrid_model, test_data)
    print(f"[{args.target_model} without augmentation] Test Metrics: {baseline}")

    hybrid_model.fit_augmentation(n_pseudo=n_pseudo, pseudo_weight=weight)
    test_result = evaluate(config, hybrid_model, test_data, show_progress=True)
    print(f"[Feature augmentation hybrid] Test Set Metrics: {test_result}")


if __name__ == '__main__':
    main()