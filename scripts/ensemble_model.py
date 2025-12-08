import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from results_logger import log_result
import joblib
import os

class base_model:

    def __init__(self, eda):
        self.df = eda.df
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.best_params = None

    def preprocess(self, target_col="left"):
        X = self.df.drop(target_col, axis=1)
        y = self.df[target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return self
    
    def get_best_cv_results(self, model_name, metric="roc_auc"):
        
        grid = self.cv_model
        metric_map = {
            'roc_auc': 'mean_test_roc_auc',
            'accuracy': 'mean_test_accuracy',
            'precision': 'mean_test_precision',
            'recall': 'mean_test_recall',
            'f1': 'mean_test_f1'
        }

        cv_results = pd.DataFrame(grid.cv_results_)
        best_idx = cv_results[metric_map[metric]].idxmax()
        best_result = cv_results.iloc[best_idx]

        self.best_params = grid.best_params_
        self.model = grid.best_estimator_

        auc = best_result.mean_test_roc_auc
        f1 = best_result.mean_test_f1
        recall = best_result.mean_test_recall
        precision = best_result.mean_test_precision
        accuracy = best_result.mean_test_accuracy

        metrics_train = pd.DataFrame([{
        "model": model_name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "AUC": auc
        }])

        print(f"\n=== 🌳 {model_name} - train 🌳 ===")
        print(metrics_train)
        log_result(metrics_train, model_name, "train")
        return self

    def get_scores(self, model_name):

        preds = self.model.predict(self.X_test)

        auc = roc_auc_score(self.y_test, preds)
        accuracy = accuracy_score(self.y_test, preds)
        precision = precision_score(self.y_test, preds)
        recall = recall_score(self.y_test, preds)
        f1 = f1_score(self.y_test, preds)

        metrics_test = pd.DataFrame([{
            'model': model_name,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Accuracy': accuracy,
            'AUC': auc
        }])
        print(f"\n=== 🌳 {model_name} - test 🌳 ===")
        print(metrics_test) 
        log_result(metrics_test, model_name, "test")

        return self

    def feature_importance(self, model_name, top_n=10):

        if not hasattr(self.model, "feature_importances_"):
            print("❌ Model does not support feature importance.")
            return self

        feat = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n✅ Top {top_n} Features for {model_name}:")
        print(feat.head(top_n))

        fig = plt.figure(figsize=(10, 6))
        plt.barh(feat['feature'].head(top_n)[::-1], feat['importance'].head(top_n)[::-1])
        plt.title(f"Feature Importance - {model_name}")
        save_path = f"D:/DA_Google_Advanced/Course7_ProjectCapstone/Project_SalifortMotors/report/feature_importance_{model_name}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Feature importance saved to: {save_path}")

        return self
    
    def save_model(self, model_name="model"):
        save_path = f"D:\DA_Google_Advanced\Course7_ProjectCapstone\Project_SalifortMotors\deploy_model\{model_name}.joblib"
        joblib.dump(self.model, save_path)
        print(f"✅ Model saved to: {save_path}")

        return self
