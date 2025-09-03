# ---------------------------------------------
# STEP 1: Import libraries
# ---------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------
# STEP 2: Data
# ---------------------------------------------
models = ["Linear Regression", "Random Forest", "XGBoost"]

r2_scores = [0.6191, 0.8230, 0.8615]
rmse = [38.51, 26.26, 23.22]
mae = [29.11, 17.24, 16.59]
mse = [1483.14, 689.38, 539.35]

x = np.arange(len(models))
width = 0.35

# ---------------------------------------------
# STEP 3: R² vs RMSE dual-axis chart
# ---------------------------------------------
fig, ax1 = plt.subplots(figsize=(8,6))

ax2 = ax1.twinx()  # Secondary y-axis

b1 = ax1.bar(x - width/2, r2_scores, width, color="skyblue", label="R² Score")
b2 = ax2.bar(x + width/2, rmse, width, color="salmon", label="RMSE")

ax1.set_xlabel("Models")
ax1.set_ylabel("R² Score")
ax2.set_ylabel("RMSE")
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=20)
ax1.set_title("Comparison of Algorithms: R² Score and RMSE")

# Create combined legend
bars = [b1, b2]
labels = ["R² Score", "RMSE"]
ax1.legend(bars, labels, loc="upper center")

plt.show()

# ---------------------------------------------
# STEP 4: MAE vs MSE dual-axis chart
# ---------------------------------------------
fig, ax1 = plt.subplots(figsize=(8,6))

ax2 = ax1.twinx()

b1 = ax1.bar(x - width/2, mae, width, color="lightgreen", label="MAE")
b2 = ax2.bar(x + width/2, mse, width, color="orange", label="MSE")

ax1.set_xlabel("Models")
ax1.set_ylabel("MAE")
ax2.set_ylabel("MSE")
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=20)
ax1.set_title("Comparison of Algorithms: MAE and MSE")

# Combined legend
bars = [b1, b2]
labels = ["MAE", "MSE"]
ax1.legend(bars, labels, loc="upper center")

plt.show()
