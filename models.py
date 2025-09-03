# ---------------------------------------------
# STEP 1: Install required libraries
# ---------------------------------------------
!pip install xgboost scikit-learn pandas matplotlib --quiet

# ---------------------------------------------
# STEP 2: Import necessary libraries
# ---------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score,
    mean_absolute_error
)
import matplotlib.pyplot as plt

# ---------------------------------------------
# STEP 3: Load dataset
# ---------------------------------------------
df = pd.read_csv("/content/odi.csv")

# Keep only required columns
columns_to_use = [
    'runs', 'wickets', 'overs',
    'runs_last_5', 'wickets_last_5',
    'venue', 'bat_team', 'bowl_team', 'total'
]
df = df[columns_to_use].dropna()

# ---------------------------------------------
# STEP 4: Features & Target
# ---------------------------------------------
X = df.drop(columns=["total"])   # Features
y = df["total"]                  # Target

categorical_features = ["venue", "bat_team", "bowl_team"]

# One-hot encoding categorical columns
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(drop="first"), categorical_features)],
    remainder="passthrough"
)

X_encoded = preprocessor.fit_transform(X)

# ---------------------------------------------
# STEP 5: Train-Test Split
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42
)

# ---------------------------------------------
# STEP 6: Train Models
# ---------------------------------------------
# Baseline
lr = LinearRegression()

# Random Forest - moderate parameters
rf = RandomForestRegressor(
    n_estimators=150, max_depth=20, n_jobs=-1, random_state=42
)

# XGBoost - tuned for stronger performance
xgb = XGBRegressor(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    n_jobs=-1, random_state=42, verbosity=0
)

# Fit models
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# ---------------------------------------------
# STEP 7: Predictions
# ---------------------------------------------
lr_preds = lr.predict(X_test)
rf_preds = rf.predict(X_test)
xgb_preds = xgb.predict(X_test)

# ---------------------------------------------
# STEP 8: Evaluation Function (Regression Only)
# ---------------------------------------------
def evaluate_model(y_true, y_pred):
    return {
        "R2 Score": round(r2_score(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "MSE": round(mean_squared_error(y_true, y_pred), 4),
    }

# ---------------------------------------------
# STEP 9: Evaluate Models
# ---------------------------------------------
lr_eval = evaluate_model(y_test.values, lr_preds)
rf_eval = evaluate_model(y_test.values, rf_preds)
xgb_eval = evaluate_model(y_test.values, xgb_preds)

print("📊 Linear Regression:", lr_eval)
print("🌲 Random Forest:", rf_eval)
print("⚡ XGBoost:", xgb_eval)

# ---------------------------------------------
# STEP 10: Plot Predictions
# ---------------------------------------------
def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true[:500], y_pred[:500], alpha=0.5, color="blue", label="Predicted")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
             "r--", label="Ideal Line")
    plt.xlabel("Actual Total Runs")
    plt.ylabel("Predicted Total Runs")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_predictions(y_test.values, lr_preds, "📊 Linear Regression: Actual vs Predicted")
plot_predictions(y_test.values, rf_preds, "🌲 Random Forest: Actual vs Predicted")
plot_predictions(y_test.values, xgb_preds, "⚡ XGBoost: Actual vs Predicted")
