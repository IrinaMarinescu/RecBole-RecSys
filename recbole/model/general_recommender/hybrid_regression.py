import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from recbole.utils import ModelType
from recbole.data.interaction import Interaction

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
        self.regressor = LogisticRegression(
            C=regressor_c, 
            penalty=penalty, 
            solver='liblinear',
            fit_intercept=True
        )

    def fit_weights(self, valid_data, num_negatives=5):
        """Extracts validation user-item pairs to train the regression weights."""
        dataset = valid_data.dataset
        user_field = self.config['USER_ID_FIELD']
        item_field = self.config['ITEM_ID_FIELD']
        
        # Extract ground truth positive interactions from validation split
        users = dataset.inter_feat[user_field].numpy()
        pos_items = dataset.inter_feat[item_field].numpy()
        
        samples_u, samples_i, labels = [], [], []
        
        # Generate supervised dataset with sampled negatives
        for u, pos_i in zip(users, pos_items):
            samples_u.append(u)
            samples_i.append(pos_i)
            labels.append(1)
            
            for _ in range(num_negatives):
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