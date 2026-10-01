import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Load dataset
df = pd.read_csv("01_data/Loan_default.csv")

print("Dataset Shape:", df.shape)

# Remove ID column
df = df.drop(columns=["LoanID"])

print("\nShape after removing LoanID:", df.shape)

# Separate features and target
X = df.drop(columns=["Default"])
y = df["Default"]
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
# Numerical features
numeric_features = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio"
]

# Categorical features
categorical_features = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner"
]
# Numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both preprocessing methods
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("Preprocessing pipeline created successfully.")
# Fit preprocessing on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform testing data
X_test_processed = preprocessor.transform(X_test)

print("Training data after preprocessing:", X_train_processed.shape)
print("Testing data after preprocessing:", X_test_processed.shape)
print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)
print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Display numerical and categorical columns
print("\nNumerical columns:")
print(X.select_dtypes(include=np.number).columns.tolist())

print("\nCategorical columns:")
print(X.select_dtypes(include="object").columns.tolist())
# Numerical features
numeric_features = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio"
]

# Categorical features
categorical_features = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner"
]

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# Create Logistic Regression model
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

# Train the model
model.fit(X_train_processed, y_train)

print("Logistic Regression model trained successfully.")

# Make predictions
y_pred = model.predict(X_test_processed)

# Get probability of default
y_prob = model.predict_proba(X_test_processed)[:, 1]

# Calculate ROC-AUC
auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC:", auc)

# Classification report
print(classification_report(y_test, y_pred))
# ==============================
# Random Forest Model
# ==============================

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train_processed, y_train)

print("Random Forest model trained successfully.")

# Make predictions
rf_pred = rf_model.predict(X_test_processed)

# Get probability of default
rf_prob = rf_model.predict_proba(X_test_processed)[:, 1]

# Calculate ROC-AUC
rf_auc = roc_auc_score(y_test, rf_prob)

print("Random Forest ROC-AUC:", rf_auc)

# Classification report
print(classification_report(y_test, rf_pred))
# ==============================
# Random Forest Model
# ==============================

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_processed, y_train)

print("Random Forest model trained successfully.")

rf_pred = rf_model.predict(X_test_processed)

rf_prob = rf_model.predict_proba(X_test_processed)[:, 1]

rf_auc = roc_auc_score(y_test, rf_prob)

print("Random Forest ROC-AUC:", rf_auc)

print(classification_report(y_test, rf_pred))
# ==============================
# XGBoost Model
# ==============================

from xgboost import XGBClassifier

# Calculate class imbalance ratio
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("Scale Pos Weight:", scale_pos_weight)

# Create XGBoost model
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

# Train
xgb_model.fit(X_train_processed, y_train)

print("XGBoost model trained successfully.")

# Predictions
xgb_pred = xgb_model.predict(X_test_processed)

# Probability of default
xgb_prob = xgb_model.predict_proba(X_test_processed)[:, 1]

# ROC-AUC
xgb_auc = roc_auc_score(y_test, xgb_prob)

print("XGBoost ROC-AUC:", xgb_auc)

# Classification report
print(classification_report(y_test, xgb_pred))
# ==============================
# Tuned XGBoost Model
# ==============================

tuned_xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

tuned_xgb.fit(X_train_processed, y_train)

print("Tuned XGBoost model trained successfully.")

tuned_prob = tuned_xgb.predict_proba(X_test_processed)[:, 1]

tuned_auc = roc_auc_score(y_test, tuned_prob)

print("Tuned XGBoost ROC-AUC:", tuned_auc)

tuned_pred = tuned_xgb.predict(X_test_processed)

print(classification_report(y_test, tuned_pred))
# ==============================
# Save Final Model
# ==============================

import joblib
import os

# Create models folder if it does not exist
os.makedirs("04_models", exist_ok=True)

# Create complete pipeline
final_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", tuned_xgb)
])

# Train complete pipeline on training data
final_pipeline.fit(X_train, y_train)

# Save model
joblib.dump(
    final_pipeline,
    "04_models/loan_default_model.joblib"
)

print("Final model saved successfully.")
print("Model path: 04_models/loan_default_model.joblib")