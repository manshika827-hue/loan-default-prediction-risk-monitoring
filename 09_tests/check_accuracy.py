import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("01_data/Loan_default.csv")

# Remove ID
df = df.drop("LoanID", axis=1)

# Separate features and target
X = df.drop("Default", axis=1)
y = df["Default"]

# Same split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load trained model
model = joblib.load(
    "04_models/loan_default_model.joblib"
)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")

print("Accuracy :", round(accuracy, 4))
print("Accuracy %:", round(accuracy * 100, 2), "%")

print("ROC-AUC  :", round(roc_auc, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))