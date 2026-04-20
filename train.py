import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from preprocess import preprocess_data

# Load dataset
df = pd.read_csv("../data/train_transaction.csv")

# Target
y = df["isFraud"]
X = df.drop(columns=["isFraud"])

# Preprocess
X = preprocess_data(X, training=True)

# Save feature columns
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "../models/feature_columns.joblib")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    use_label_encoder=False
)
xgb_model.fit(X_res, y_res)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    n_jobs=-1
)
rf_model.fit(X_res, y_res)

# Save models
joblib.dump(xgb_model, "../models/xgboost_fraud.joblib")
joblib.dump(rf_model, "../models/rf_fraud.joblib")

print("✅ Models trained and saved.")