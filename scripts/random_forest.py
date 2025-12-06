from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from ensemble_model import base_model

class random_forest_model(base_model):

    def train(self):
        param_grid = {
            "max_depth": [3, 5, None],
            'max_features': ['sqrt', 'log2'],
            'max_samples': [0.7, 1.0],
            "min_samples_split": [2, 3, 4],
            "min_samples_leaf": [1, 2, 3],
            'n_estimators': [300, 500]
        }
        scoring = {
            'accuracy':'accuracy', 'precision':'precision',
            'recall':'recall', 'f1':'f1', 'roc_auc':'roc_auc'
        }

        forest = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring=scoring,
            refit='roc_auc',
            n_jobs=-1
        )

        forest.fit(self.X_train, self.y_train)
        self.cv_model = forest
        self.best_params = forest.best_params_  

        print("🌳 RF Best Params:", self.best_params)
        print("🔥 RF Best AUC:", forest.best_score_)

        return self
