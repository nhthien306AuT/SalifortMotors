import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import logit

class check_LR_logit:
    def __init__(self, eda, target):
        self.df = eda.df
        self.target = target

    def auto_box_tidwell(self, alpha=0.05):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target in numeric_cols:
            numeric_cols.remove(self.target)
        
        results = {}
        for col in numeric_cols:
            self.df['Xlog'] = self.df[col] * np.log(self.df[col] + 1e-6)
            formula = f"{self.target} ~ {col} + Xlog"
            model = logit(formula, data=self.df).fit(disp=False)
            pval = model.pvalues['Xlog']
            results[col] = pval
        
        results_df = pd.DataFrame.from_dict(results, orient='index', columns=['p_value']).sort_values('p_value')
        results_df['violates_linearity'] = results_df['p_value'] < alpha
        self.results = results_df
        
        print("\n✅ Box-Tidwell Test Results ")
        print(self.results)
        print("\n❌ Features violating linearity assumption (p < {:.2f}) ===".format(alpha))
        print(self.results[self.results['violates_linearity']])
        
        return self  


