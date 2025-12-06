from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from ensemble_model import base_model

class xgboost_model(base_model):

    def train(self):

        param_grid = {
            'n_estimators': [200, 300],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.7, 1.0],
            'colsample_bytree': [0.7, 1.0],
            'gamma': [0, 1]
        }
        scoring = {
            'accuracy':'accuracy', 'precision':'precision',
            'recall':'recall', 'f1':'f1', 'roc_auc':'roc_auc'
        }
        xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, n_jobs=-1)
        xgb_fit = GridSearchCV(xgb, param_grid=param_grid, scoring=scoring, cv=5, refit="roc_auc", n_jobs=-1)
        xgb_fit.fit(self.X_train, self.y_train)
        self.cv_model = xgb_fit
        self.best_params = xgb_fit.best_params_  

        print("⚡ XGB Best Params:", self.best_params)
        print("🔥 XGB Best AUC:", xgb_fit.best_score_)

        return self

