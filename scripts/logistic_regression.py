import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from results_logger import log_result

class logistic_model:
    def __init__(self, eda, target_col="left"):
        self.df = eda.df
        self.target_col = target_col
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess(self):
        
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return self
    
    def train(self):
        self.model = LogisticRegression(max_iter=500).fit(self.X_train, self.y_train)

        return self
    
    def evaluate(self, save_path= "D:/DA_Google_Advanced/Course7_ProjectCapstone/Project_SalifortMotors/report/confusion_matrix.png"):
        y_pred = self.model.predict(self.X_test)

        cm = confusion_matrix(self.y_test, y_pred, labels=self.model.classes_)
        
        target_names = ['Not leave', 'Leave']
        report=classification_report(self.y_test, y_pred, target_names=target_names, output_dict=True)
        fig, ax = plt.subplots(figsize=(5,4))
        log_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        log_disp.plot(values_format='', ax=ax)

        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        print(f"✅ Confusion matrix saved to: {save_path}")
        print("\n📊 Confusion Matrix:")
        print(cm)
        print("\n📝 Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=target_names))
        log_result(report, "Logistic", "train")
        
        return self