"""
Model Building and Evaluation

- Train/test split
- XGBoost model training
- Predictions
- Evaluation metrics (confusion matrix, classification report, ROC-AUC)
"""

# =========================
# 1) Imports
# =========================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)

# =========================
# 2) Train/Test Split
# =========================
# Assuming df_cleaned is loaded from prepration.py or an earlier notebook step
x = df_cleaned.drop(columns=["fraud_reported"])
y = df_cleaned["fraud_reported"]

# Sanity check – ensure no object columns remain
assert x.select_dtypes(include="object").empty, "Object columns found in features."

# Split into training/testing sets (80/20 stratified)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Data split complete: Train shape {X_train.shape}, Test shape {X_test.shape}")

# =========================
# 3) Model Training (XGBoost)
# =========================
Xtr = X_train.to_numpy(dtype=np.float32, copy=True)
Xte = X_test.to_numpy(dtype=np.float32, copy=True)
ytr = y_train.to_numpy()

xgb_model = XGBClassifier(
    learning_rate=0.05,
    n_estimators=2000,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    tree_method="hist"
)

xgb_model.fit(Xtr, ytr)
print("✅ XGBoost model trained successfully.")

# =========================
# 4) Predictions
# =========================
y_pred = xgb_model.predict(Xte)
y_pred_prob = xgb_model.predict_proba(Xte)[:, 1]

# =========================
# 5) Evaluation Metrics
# =========================
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"\nROC AUC: {roc_auc:.3f}")

# Plot ROC Curve
RocCurveDisplay.from_predictions(y_test, y_pred_prob)

# =========================
# 6) Next Steps (optional extensions)
# =========================
# - Add SHAP explainability analysis
# - Apply hyperparameter tuning (GridSearchCV / Optuna)
# - Handle class imbalance (SMOTE or scale_pos_weight)
# - Save model using joblib.dump()
# - Integrate model deployment via Flask API
