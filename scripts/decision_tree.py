import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from results_logger import log_result


class decision_tree_model:

    def __init__(self, eda):

        self.df = eda.df
        self.model = None
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None


    def preprocess(self, target_col="left"):

        X = self.df.drop(target_col, axis=1)
        y = self.df[target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        return self

    def train(self, model_name, metric="auc"):
    
        param_grid = {
            "max_depth": [4, 6, 8, None],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf": [1, 2, 4]
        }
        scoring = {'accuracy':'accuracy', 'precision':'precision', 'recall':'recall', 'f1':'f1', 'roc_auc':'roc_auc'}
        
        tree = GridSearchCV (DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring=scoring, refit='roc_auc', n_jobs=-1)
        tree.fit(self.X_train, self.y_train)
        self.best_params = tree.best_params_
        self.model = tree.best_estimator_

        print("✅ Best params found:", self.best_params)
        print("✅ Best Score: ", tree.best_score_)
    

        metric_dict = {'auc': 'mean_test_roc_auc',
                    'precision': 'mean_test_precision',
                    'recall': 'mean_test_recall',
                    'f1': 'mean_test_f1',
                    'accuracy': 'mean_test_accuracy'
                    }
        cv_results = pd.DataFrame(tree.cv_results_)
        best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

        auc = best_estimator_results.mean_test_roc_auc
        f1 = best_estimator_results.mean_test_f1
        recall = best_estimator_results.mean_test_recall
        precision = best_estimator_results.mean_test_precision
        accuracy = best_estimator_results.mean_test_accuracy

        metrics_df = pd.DataFrame([{
        "model": model_name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "AUC": auc
        }])

        print("\n=== 🌳 Decision Tree Result 🌳 ===")
        print(metrics_df)
        log_result(metrics_df, model_name, "train")