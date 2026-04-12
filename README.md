# Customer Churn Prediction

A machine learning project to predict customer churn for a telecom provider using demographic data, service usage patterns, and billing information. The project covers the full pipeline — from data cleaning and preprocessing to model training, hyperparameter tuning, evaluation, interpretability, and inference.

---

## Dataset

**Source:** [Telecom Customer Churn by Maven Analytics](https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics)

- 7,043 customers, 38 features in the raw dataset
- After cleaning: 6,589 customers, 30 features
- Target: `Customer Status` — Churned (1) / Stayed (0)
- Class distribution: ~71.6% Stayed, ~28.4% Churned

---


## Data Cleaning & Preprocessing

**Columns removed before modelling:**
- `Customer ID`, `Zip Code`, `Latitude`, `Longitude` — identifiers with no predictive value
- `City` — 1,106 unique values, OHE would generate 1,100+ binary columns with minimal signal
- `Churn Category`, `Churn Reason` — post-churn labels, using them as features would cause direct data leakage
- Customers with `Customer Status = 'Joined'` removed — insufficient history to label as churned or stayed

**Missing value handling — structural, not statistical:**
- 3,877 customers with no active offer → filled with `'No Offer'`
- 682 customers with no phone plan → `Multiple Lines` filled with `'No Phone Service'`, long distance charges filled with `0`
- 1,526 customers with no internet plan → all internet-related features filled with `'No Internet Service'`, `Avg Monthly GB Download` filled with `0`

Imputing these with mean or median would have introduced factually incorrect information into the dataset.

**Encoding & scaling:**
- `OneHotEncoder(drop='first')` for 18 categorical features — avoids multicollinearity
- Numerical features passed through without scaling — tree-based models split on thresholds, not distances

**Pipeline:**
- Full `sklearn Pipeline` wrapping preprocessor and model — prevents data leakage and ensures consistent transformations during training and inference

---

## Churn Reason Analysis

The dataset includes a `Churn Reason` column — used for business insight only, dropped before modelling.

Top findings:
- Competitor-related reasons (better devices, better offers, more data) account for the majority of churn
- "Attitude of support person" is the third highest reason — a pure service experience issue independent of pricing
- "Price too high" ranks lower than expected, suggesting customers are pulled away by competitors rather than pushed away by dissatisfaction

---

## Models

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

**Feature Importance** — impurity-based scores identify Monthly Charge, Age, and Tenure as top contributors.

**SHAP Analysis** — `TreeExplainer` used for more reliable feature attribution:
- Low tenure customers have the highest churn risk
- Long-term contracts (One Year, Two Year) significantly reduce churn probability
- Number of referrals is a strong loyalty signal — high referrers rarely churn
- Monthly Charge shows less directional impact than impurity importance suggested — SHAP corrects this overstatement

**Key cross-analysis insight:** The model's top feature (Monthly Charge) and the top customer-reported churn reason (competitor offerings) don't tell the same story. The model picks up on high monthly charges as a risk signal — but customers aren't necessarily leaving because the price is too high, they're leaving because a competitor made a better offer. Pricing sensitivity and competitive pressure are related but not the same thing.

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
