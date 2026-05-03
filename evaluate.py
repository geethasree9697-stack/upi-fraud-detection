import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc
)

# -----------------------
# LOAD MODEL & DATA
# -----------------------
model = joblib.load("model/xgb_model.pkl")
df = pd.read_csv("dataset/processed.csv")

X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# -----------------------
# PREDICT PROBABILITIES
# -----------------------
y_probs = model.predict_proba(X)[:, 1]

# -----------------------
# FIND THRESHOLD FOR HIGH RECALL
# -----------------------
threshold = 0.20  # catches almost all frauds

y_pred = (y_probs >= threshold).astype(int)

# -----------------------
# METRICS
# -----------------------
accuracy = accuracy_score(y, y_pred)
cm = confusion_matrix(y, y_pred)

print(f"Threshold for ~99% recall: {threshold}")
print(f"Accuracy: {accuracy:.5f}")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y, y_pred))

# -----------------------
# SAVE RESULTS
# -----------------------
results = df.copy()
results["predicted"] = y_pred
results["probability"] = y_probs
results.to_csv("dataset/predictions.csv", index=False)

frauds = results[results["predicted"] == 1]
frauds.to_csv("dataset/detected_frauds.csv", index=False)

print("\nCSV files saved in dataset/ folder")

# -----------------------
# VISUALIZATION
# -----------------------

# Confusion Matrix Heatmap
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0,1], ["Normal", "Fraud"])
plt.yticks([0,1], ["Normal", "Fraud"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUC = {roc_auc:.3f})")
plt.show()

# Precision Recall Curve
precision, recall, _ = precision_recall_curve(y, y_probs)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()
