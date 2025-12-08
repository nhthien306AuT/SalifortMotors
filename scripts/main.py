from load import data_loader
from filepaths import PATHS
from clean import data_cleaner
from eda import data_eda
from boxplot import check_outliers
from LR_logit import check_LR_logit
from random_forest import random_forest_model
from logistic_regression import logistic_model
from decision_tree import decision_tree_model
from xgb import xgboost_model
import matplotlib
matplotlib.use("Agg")


if __name__ == "__main__":

# Cleaning the dataset
    loader = data_loader(PATHS["hr"]).load_csv() 
    cleaner = data_cleaner(loader).handle_nulls().remove_duplicates().trim().standardize()
    checker = check_outliers(cleaner).create_boxplot().remove_outliers() # it's necessary for logistic model
    # checker.plot_boxplot()
 
# Get overview of this dataset 
    eda = data_eda(checker).overview().categorical_summary().left_summary()

# Resolving data leakage
    eda.transform_data("average_monthly_hours").encode()

# Explore Data Analytics dataset
    # eda.plot_correlation()
    # eda.plot_boxplot("number_project","satisfaction_level").plot_boxplot("number_project","last_evaluation")
    # eda.plot_scatter("average_monthly_hours","satisfaction_level").plot_scatter("average_monthly_hours","last_evaluation").plot_boxplot(
    #     "number_project","average_monthly_hours").plot_scatter("average_monthly_hours","promotion_last_5years")
    # eda.plot_boxplot("time_spend_company","satisfaction_level").plot_boxplot("time_spend_company","last_evaluation").plot_boxplot(
    #     "time_spend_company","number_project").plot_boxplot("time_spend_company","average_monthly_hours")
    # eda.plot_histogram("satisfaction_level").plot_histogram("last_evaluation").plot_histogram("number_project").plot_histogram(
    #     "average_monthly_hours").plot_histogram("time_spend_company").plot_histogram("salary")
    # eda.plot_histogram("department")

# Chech logit assumption (Logistic model)
    # logit = check_LR_logit(eda, target='left').auto_box_tidwell() 

# Build & Evaluate model
    # logis = logistic_model(eda).preprocess().train().evaluate()
    # tree= decision_tree_model(eda).preprocess().train("Decision Tree")
    # rf = random_forest_model(eda).preprocess().train().get_best_cv_results("RF").get_scores("RF").feature_importance("RF")
    # xgb = xgboost_model(eda).preprocess().train().get_best_cv_results("XGB").get_scores("XGB").feature_importance("XGB").save_model("XGB")   





    