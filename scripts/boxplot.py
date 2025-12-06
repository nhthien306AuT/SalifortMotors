import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 

class check_outliers:

    def __init__(self, cleaner):
        self.df = cleaner.df
        self.columns_to_check = ['number_project', 'average_monthly_hours', 'time_spend_company']
        self.report = None

    def create_boxplot(self):
        report = []
        n_cols = len(self.columns_to_check)
        fig, axes = plt.subplots(n_cols, 1, figsize=(6, 3*n_cols)) 
        if n_cols == 1:
            axes = [axes]  

        for ax, col in zip(axes, self.columns_to_check):
            if col not in self.df.columns:
                print(f"Warning: Column '{col}' not found in DataFrame.")
                continue

            sns.boxplot(x=self.df[col], ax=ax)
            ax.set_title(f'Boxplot: {col}')

            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]

            report.append({
                'column': col,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'Lower Bound': lower_bound,
                'Upper Bound': upper_bound,
                'Outlier Count': len(outliers)
            })
        self.report = pd.DataFrame(report)
        self.fig = fig
        print(self.report) 
        plt.tight_layout()
        return self
    
    def plot_boxplot(self, save_path="D:/DA_Google_Advanced/Course7_ProjectCapstone/Project_SalifortMotors/report/boxplot_outliers.png"):
        self.fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(self.fig)
        print(f"✅ Saved boxplot to: {save_path}")

    def remove_outliers(self):
        if self.report is None:
            print("Error: No report found. Run create_boxplot() first.")
            return self

        for _, row in self.report.iterrows():
            col = row['column']
            lower = row['Lower Bound']
            upper = row['Upper Bound']
            self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]

        print(f"✅ Outliers removed. Remaining rows: {len(self.df)}")
        return self