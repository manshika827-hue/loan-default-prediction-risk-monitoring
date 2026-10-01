import pandas as pd
import shap
import joblib


# Load trained pipeline
model = joblib.load("04_models/loan_default_model.joblib")

# Get preprocessing and XGBoost model
preprocessor = model.named_steps["preprocessor"]
xgb_model = model.named_steps["model"]

# Create SHAP explainer
explainer = shap.TreeExplainer(xgb_model)


def explain_loan(input_data):
    """
    Generate SHAP values for one loan application.
    """

    # Transform input using the same preprocessing
    input_transformed = preprocessor.transform(input_data)

    # Convert sparse matrix to dense
    if hasattr(input_transformed, "toarray"):
        input_transformed = input_transformed.toarray()

    # Get feature names
    feature_names = preprocessor.get_feature_names_out()

    # Calculate SHAP values
    shap_values = explainer.shap_values(input_transformed)

    return shap_values, input_transformed, feature_names