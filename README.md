# Customer Churn Prediction (Machine Learning)

## Overview
This repository contains an end-to-end machine learning project focused on predicting customer churn using structured telecom data. The project emphasizes **clean preprocessing**, **robust model evaluation**, and **business-aware decision making**, rather than model complexity for its own sake.

The primary objective is to identify customers at high risk of churning so that retention strategies can be applied proactively.

---

## Problem Statement
Customer churn directly impacts revenue and growth. Acquiring new customers is significantly more expensive than retaining existing ones.  
This project formulates churn prediction as a **binary classification problem**, with a strong focus on handling **class imbalance** and selecting evaluation metrics aligned with real-world business costs.

---

## Dataset
- Structured telecom customer data
-source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Key challenges addressed:
- Incorrect data types in raw features
- Missing values introduced during type conversion
- High-cardinality categorical variables
- Imbalanced target distribution

---

## Approach

### 1. Data Understanding & Cleaning
- Explicit data type correction for numeric fields
- Controlled handling of missing values
- Clear separation of numerical and categorical features
- No silent row drops or leakage-prone transformations

### 2. Preprocessing Pipeline
- `ColumnTransformer` used for clean feature handling
- Standard scaling for numerical variables
- One-hot encoding for categorical variables
- Fully reproducible preprocessing integrated with modeling

### 3. Modeling Strategy
Models evaluated:
- Decision Tree (baseline)
- Random Forest
- Tuned Random Forest (final model)

Key considerations:
- Avoiding overfitting
- Stability across cross-validation folds
- Interpretability vs performance trade-offs

### 4. Model Evaluation
Given the imbalanced nature of churn data:
- Accuracy is treated as a secondary metric
- Primary focus on:
  - Recall (churn class)
  - Precision
  - F1-score
  - ROC-AUC

Cross-validation and hyperparameter tuning are used to ensure generalization.

---

## Results
- Random Forest significantly outperformed the baseline Decision Tree
- Hyperparameter tuning improved recall and overall stability
- Final model balances churn detection capability with acceptable false positives

The chosen model reflects a **business-aware compromise**, prioritizing the identification of potential churners.

---

## Feature Importance & Interpretability
- Feature importance analysis used to identify key churn drivers
- Results interpreted with awareness of bias in impurity-based importance
- The project is designed to be easily extended with advanced interpretability tools (e.g., SHAP)







---

## Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Jupyter Notebook

---

## Key Takeaways
- Proper preprocessing often has more impact than model complexity
- Metric selection must align with business objectives
- Handling class imbalance is critical in churn prediction problems
- Simpler models, when well-tuned, can outperform more complex baselines

---

## Future Enhancements
- Decision threshold optimization based on business cost
- Precision–Recall curve analysis
- SHAP-based feature explanations
- Model deployment via REST API



