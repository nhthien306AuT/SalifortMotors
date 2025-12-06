import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

class data_eda:
    def __init__(self, checker):
        self.df = checker.df
        self.report = {}

    def overview(self):
        print("\n🟢 Describe (numeric)")
        print(self.df.describe())
        print("\n🔴 Describe (object)")
        print(self.df.describe(include='object'))
        return self


    def categorical_summary(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include='object').columns.tolist()
        print("\n✨ Categorical Summary")
        for col in columns:
            print(f"\nColumn '{col}':")
            print(self.df[col].value_counts())
        return self
    
    def transform_data(self, col):
        self.df["workload_status"] = self.df[col].apply(
        lambda x: "overworked" if x > 175 else "normal")
        self.df = self.df.drop(columns=[col])
        print(f"\n✅ Transform & Drop {col} column successfully")
        
        return self


    def encode(self):
        remaining = self.df.select_dtypes(include='object').columns
        for col in remaining:
            if self.df[col].nunique() <= 5:  
                self.df[col] = pd.factorize(self.df[col])[0]
            else:  
                self.df = pd.get_dummies(self.df, columns=[col], drop_first=False, dtype=int)
        print("\n✅ Encode successfully")
        return self
    
    def plot_correlation(self, method='pearson', figsize=(20,14), cmap="coolwarm", save_path="D:/DA_Google_Advanced/Course7_ProjectCapstone/Project_SalifortMotors/report/correlation_heatmap.png"):
        corr = self.df.corr(method=method)
        self.report['correlation'] = corr  

        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, cbar=True)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved heatmap to: {save_path}")
        return self

    def left_summary(self, col='left'):

        if col not in self.df.columns:
            print(f"Column '{col}' not found in DataFrame.")
            return self

        summary_df = self.df[col].value_counts().to_frame('Count')
        summary_df['Percentage (%)'] = (summary_df['Count'] / len(self.df) * 100).round(2)

        print("\n📊 Left vs Stayed Summary")
        print(summary_df)
        self.report['left_summary'] = summary_df
        return self
    
    def plot_boxplot(self, x=None, y=None, save_dir="D:/DA_Google_Advanced/Course7_ProjectCapstone/Project_SalifortMotors/report/"):
        plt.figure(figsize=(20, 14))

        if x is not None and y is not None:
            sns.boxplot(data=self.df, x=x, y=y, hue='left')
            plt.title(f"Boxplot of {y} by {x} grouped by 'left'")
        elif x is not None:
            sns.boxplot(data=self.df, y=x, hue='left')
            plt.title(f"Boxplot of {x} grouped by 'left'")
        else:
            print("⚠️ Please provide at least one variable for boxplot.")
            return self

        plt.tight_layout()
        file_name = f"boxplot_{x}_vs_{y}.png"
        save_path = save_dir + file_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved boxplot to: {save_path}")
        return self


    def plot_histogram(self, col=None, bins=30, save_dir="D:/DA_Google_Advanced/Course7_ProjectCapstone/Project_SalifortMotors/report/"):
        if col:
            cols = [col]
        else:
            cols = self.df.select_dtypes(include=['number']).columns.tolist()

        for c in cols:
            plt.figure(figsize=(20, 14))
            sns.histplot(data=self.df, x=c, hue='left', multiple='dodge', bins=bins, kde=True)
            plt.title(f"Histogram of {c} grouped by 'left'")
            plt.tight_layout()
            file_name = f"histogram_{col}.png"
            save_path = save_dir + file_name
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Saved boxplot to: {save_path}")

        return self


    def plot_scatter(self, x=None, y=None, save_dir="D:/DA_Google_Advanced/Course7_ProjectCapstone/Project_SalifortMotors/report/"):
        if x and y:
            plt.figure(figsize=(20, 14))
            sns.scatterplot(data=self.df, x=x, y=y, hue='left')
            plt.title(f"Scatterplot: {x} vs {y} grouped by 'left'")
            plt.tight_layout()
            file_name = f"scatter_{x}_vs_{y}.png"
            save_path = save_dir + file_name
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Saved boxplot to: {save_path}")
            return self

        

