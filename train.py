import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("dataset/processed.csv")

X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Faster XGBoost model
model = XGBClassifier(
    n_estimators=120,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=10,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/xgb_model.pkl")

print("✅ Model trained & saved successfully")

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))