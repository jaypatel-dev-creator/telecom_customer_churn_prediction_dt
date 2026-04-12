# Customer Churn Prediction

A machine learning project to predict customer churn for a telecom provider using demographic data, service usage patterns, and billing information. The project covers the full pipeline — from data cleaning and preprocessing to model training, hyperparameter tuning, evaluation, interpretability, and inference.

---

## Dataset

**Source:** [Telecom Customer Churn by Maven Analytics](https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics)


---


## ML Pipeline

**Preprocessing**
- Structural missing values handled based on service subscription context — not imputed with mean/median
- High-cardinality column (`City`, 1,106 unique values) dropped before encoding
- `OneHotEncoder(drop='first')` for categorical features — avoids multicollinearity
- Numerical features passed through without scaling — tree-based models do not require it
- Full `sklearn Pipeline` used to prevent data leakage

**Models**
Three models built in progression:

| Model | Role |
|-------|------|
| Decision Tree | Baseline |
| Random Forest | Ensemble improvement |
| LightGBM (tuned) | Final model |

**Hyperparameter Tuning**
- `GridSearchCV` with `StratifiedKFold(n_splits=5)`
- Scoring metric: `average_precision` (PR-AUC) — more informative than ROC-AUC for imbalanced data
- 16 parameter combinations, 80 total fits

---

## Results

| Metric | Decision Tree | Random Forest | Tuned LightGBM |
|--------|--------------|---------------|----------------|
| Test Accuracy | 0.84 | 0.84 | 0.85 |
| Test ROC-AUC | 0.8926 | 0.9188 | 0.9201 |
| Test PR-AUC | — | — | 0.8463 |
| Churn Recall | 0.75 | 0.78 | 0.77 |
| Churn Precision | 0.69 | 0.70 | 0.72 |
| Churn F1 | 0.72 | 0.74 | 0.74 |

**Decision threshold tuned to 0.5784** (optimised for F1) — improves churn precision to 0.76 over the default 0.5 threshold.

---

## Interpretability

**Feature Importance** — impurity-based scores from LightGBM identify Monthly Charge, Age, and Tenure as top contributors.

**SHAP Analysis** — TreeExplainer used for more reliable feature attribution. Key findings:
- Low tenure customers have the highest churn risk
- Customers on long-term contracts (One Year, Two Year) are significantly less likely to churn
- Number of referrals is a strong loyalty signal — high referrers rarely churn
- Monthly Charge shows less directional impact than impurity importance suggested — SHAP corrects this overstatement

---

## Inference

```python
import joblib
import pandas as pd

model = joblib.load('churn_model.pkl')

customer = pd.DataFrame([{
    'Gender': 'Male',
    'Age': 35,
    'Tenure in Months': 5,
    'Contract': 'Month-to-Month',
    'Monthly Charge': 95.0,
    # ... remaining features
}])

proba = model.predict_proba(customer)[0][1]
prediction = 'Churned' if proba >= 0.5784 else 'Stayed'
print(f"Churn Probability: {proba:.4f} — {prediction}")
```

---

## Tech Stack

- Python 3.12
- scikit-learn
- LightGBM
- SHAP
- pandas, numpy
- matplotlib, seaborn
- joblib

---

## Future Work

- Expand GridSearch to include LightGBM regularisation parameters (`reg_lambda`, `min_child_samples`) to reduce train-test gap
- Business cost-based threshold tuning using actual retention campaign costs
- FastAPI deployment as a REST inference endpoint
- Scheduled retraining pipeline to handle data drift over time
