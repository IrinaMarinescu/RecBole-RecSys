import argparse
from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer
from recbole.model.general_recommender.hybrid_regression import HybridRegression

import numpy as np
np.float = float    
np.int = int        
np.object = object  
np.bool = bool      

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_files', nargs='+', required=True, 
                        help='Paths to pre-trained .pth files in saved/ directory')
    args = parser.parse_args()
    
    base_models = []
    config, dataset, train_data, valid_data, test_data = None, None, None, None, None

    # Load pre-trained base models and identically seeded dataset splits
    for file_path in args.model_files:
        print(f"Loading {file_path}...")
        cfg, model, ds, tr_data, val_data, te_data = load_data_and_model(file_path)
        base_models.append(model)
        
        if config is None:
            config = cfg
            dataset = ds
            train_data = tr_data
            valid_data = val_data
            test_data = te_data

    print("\n--- Base Model Standalone Test Metrics ---")
    model_names = ["BPR", "NeuMF", "ItemKNN"] # MUST HAVE THE SAME ORDER as the --model_files arguments
    for name, model in zip(model_names, base_models):
        # Wrap the base model in a Trainer just to evaluate it
        temp_trainer = Trainer(config, model)
        test_result = temp_trainer.evaluate(test_data, load_best_model=False, show_progress=False)
        print(f"[{name}] Test Metrics: {test_result}")

    # Task 1.5: Hyperparameter tuning grid for the hybrid model
    c_values = [0.01, 0.1, 1.0, 10.0]
    best_score = 0.0
    best_hybrid = None

    print("\n--- Tuning Hybrid Regression Weights ---")
    for c in c_values:
        hybrid_model = HybridRegression(config, base_models, regressor_c=c)
        hybrid_model.fit_weights(valid_data)
        
        # Evaluate hybrid using RecBole Trainer
        trainer = Trainer(config, hybrid_model)
        val_result = trainer.evaluate(valid_data, load_best_model=False, show_progress=False)
        
        # Identify validation metric
        metric_key = list(val_result.keys())[0] 
        score = val_result[metric_key]
        
        print(f"C={c} | Validation {metric_key}: {score:.4f} | Coefficients: {hybrid_model.regressor.coef_[0]}")
        
        if score > best_score:
            best_score = score
            best_hybrid = hybrid_model

    print("\n--- Final Offline Evaluation on Test Set ---")
    trainer = Trainer(config, best_hybrid)
    test_result = trainer.evaluate(test_data, load_best_model=False)
    print(f"Test Set Metrics: {test_result}")


if __name__ == '__main__':
    main()