# scripts/debug_pipeline.py

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================================
# Load predictions and ground truth
# ================================
pred_file = 'data/processed/predictions_FD001.csv'
rul_file = 'data/raw/RUL_FD001.txt'

if not os.path.exists(pred_file):
    raise FileNotFoundError(f"{pred_file} not found. Run predictions first.")

predictions = pd.read_csv(pred_file)

if not os.path.exists(rul_file):
    raise FileNotFoundError(f"{rul_file} not found. Cannot evaluate RUL.")

rul_true = pd.read_csv(rul_file, header=None).values.flatten()
y_pred = predictions['Predicted_RUL'].values[:len(rul_true)]

# ================================
# Step 1: Scatter plot Predicted vs Actual
# ================================
plt.figure(figsize=(10,6))
plt.scatter(range(len(rul_true)), rul_true, label='Actual RUL', color='blue')
plt.scatter(range(len(y_pred)), y_pred, label='Predicted RUL', color='red', alpha=0.6)
plt.xlabel('Engine Index')
plt.ylabel('RUL')
plt.title('Predicted vs Actual RUL per Engine')
plt.legend()
plt.show()

# ================================
# Step 2: Residuals
# ================================
residuals = rul_true - y_pred
plt.figure(figsize=(8,5))
plt.hist(residuals, bins=20, color='purple', alpha=0.7)
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.show()

print(f"Residual stats: mean={residuals.mean():.2f}, std={residuals.std():.2f}, min={residuals.min():.2f}, max={residuals.max():.2f}")

# ================================
# Step 3: Load training feature info
# ================================
feature_columns_file = 'models/feature_columns.pkl'
rf_model_file = 'models/randomforest_model.pkl'

with open(feature_columns_file, 'rb') as f:
    feature_columns = pickle.load(f)

with open(rf_model_file, 'rb') as f:
    rf_model = pickle.load(f)

importances = rf_model.feature_importances_
sorted_features = sorted(zip(feature_columns, importances), key=lambda x: x[1], reverse=True)
print("\nTop 10 features by importance:")
for f, imp in sorted_features[:10]:
    print(f"{f}: {imp:.4f}")

# ================================
# Step 4: Check for missing values in raw train data
# ================================
raw_train_file = 'data/raw/train_FD001.txt'
df_train = pd.read_csv(raw_train_file, delim_whitespace=True, header=None)
print("\nMissing values per column in training data:")
print(df_train.isna().sum())

# ================================
# Step 5: Compare train/test RUL ranges
# ================================
rul_train = pd.read_csv(raw_train_file, delim_whitespace=True, header=None).iloc[:,-1].values
print(f"\nTrain RUL range: min={rul_train.min()}, max={rul_train.max()}")
print(f"Test RUL range: min={rul_true.min()}, max={rul_true.max()}")

# ================================
# Step 6: Summary metrics
# ================================
from sklearn.metrics import mean_squared_error, r2_score

rmse = np.sqrt(mean_squared_error(rul_true, y_pred))
r2 = r2_score(rul_true, y_pred)
print(f"\nEvaluation on test set: RMSE={rmse:.2f}, R²={r2:.2f}")
