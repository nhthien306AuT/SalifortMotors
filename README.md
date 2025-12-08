# SalifortMotor Project - Data Analysis - Data Science
1. Project Overview
  - This project analyzes employee attrition at Salifort Motors using a full Data Science workflow.
  - It includes data preprocessing, exploratory analysis, and machine learning modeling to identify key drivers of turnover 
and support HR decision-making.
2. Project Structure
  - Folders:
    + dataset: Raw data used for analysis and modeling.
    + deploy_model: contains the finalized model saved in .joblib format, used during the deployment stage.
    + report: Exploratory analysis, visuals (boxplots, scatterplots, histogram), feature importance, model results.
    + final_report: Final analytical report with insights & recommendations,full data science modeling report: model building, evaluation, and performance analysis.
    + scripts: Python scripts for data cleaning, EDA, feature engineering & modeling.
3. Tech Stack
  - Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.
4. Workflow
  - Data cleaning and preprocessing
  - Exploratory Data Analysis (EDA)
  - Feature engineering
  - Model training and evaluation (Logistic Regression, Decision Tree, Random Forest, XGBoost)
  - Deploy the champion model 
  - Extracting insights to support HR decision-making
5. Results
  - XGBoost achieved the best performance with strong feature importance signals 
highlighting tenure, satisfaction level, number of project, workload status and last evaluation as top predictors of attrition.
