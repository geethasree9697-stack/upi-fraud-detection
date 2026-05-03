import pandas as pd
import joblib
import numpy as np

# -----------------------
# LOAD MODEL
# -----------------------
model = joblib.load("model/xgb_model.pkl")

# -----------------------
# SET THRESHOLD
# -----------------------
THRESHOLD = 0.2

# -----------------------
# LOAD NEW TRANSACTIONS
# -----------------------
new_data = pd.read_csv("dataset/new_transactions.csv")

# -----------------------
# ENSURE COLUMN ORDER MATCHES TRAINING
# -----------------------
feature_names = model.get_booster().feature_names
new_data = new_data[feature_names]

# -----------------------
# PREDICT PROBABILITY
# -----------------------
probs = model.predict_proba(new_data)[:, 1]

# -----------------------
# APPLY THRESHOLD
# -----------------------
predictions = (probs >= THRESHOLD).astype(int)

# -----------------------
# SAVE RESULTS
# -----------------------
results = new_data.copy()
results["fraud_probability"] = probs
results["prediction"] = predictions

results.to_csv("dataset/prediction_results.csv", index=False)

print("✅ Predictions completed")
print("Results saved to dataset/prediction_results.csv")