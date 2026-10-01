import sys
import os
import streamlit as st
import requests
import pandas as pd
from importlib.util import spec_from_file_location, module_from_spec


sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)


shap_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "08_monitoring",
    "shap_explanation.py"
)

spec = spec_from_file_location("shap_explanation", shap_path)
shap_module = module_from_spec(spec)
spec.loader.exec_module(shap_module)

explain_loan = shap_module.explain_loan

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Default Prediction System")
st.write("Enter the applicant's loan details below.")

st.divider()

# Applicant details
st.subheader("👤 Applicant Details")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income", min_value=0.0, value=50000.0)
credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=700
)

education = st.selectbox(
    "Education",
    ["High School", "Bachelor's", "Master's", "PhD"]
)

employment_type = st.selectbox(
    "Employment Type",
    ["Full-time", "Part-time", "Self-employed", "Unemployed"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

has_mortgage = st.selectbox(
    "Has Mortgage",
    ["Yes", "No"]
)

has_dependents = st.selectbox(
    "Has Dependents",
    ["Yes", "No"]
)

st.divider()

# Loan details
st.subheader("💰 Loan Details")

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=15000.0
)

months_employed = st.number_input(
    "Months Employed",
    min_value=0,
    value=60
)

num_credit_lines = st.number_input(
    "Number of Credit Lines",
    min_value=0,
    value=3
)

interest_rate = st.number_input(
    "Interest Rate (%)",
    min_value=0.0,
    value=8.5
)

loan_term = st.number_input(
    "Loan Term (months)",
    min_value=1,
    value=36
)

dti_ratio = st.number_input(
    "DTI Ratio",
    min_value=0.0,
    max_value=1.0,
    value=0.30
)

loan_purpose = st.selectbox(
    "Loan Purpose",
    ["Home", "Auto", "Education", "Business", "Other"]
)

has_cosigner = st.selectbox(
    "Has Co-Signer",
    ["Yes", "No"]
)

st.divider()

# Prediction button
if st.button("🔍 Predict Loan Risk", use_container_width=True):

    data = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data
        )

        if response.status_code == 200:

            result = response.json()

            st.success("Prediction completed successfully!")

            st.subheader("📊 Prediction Result")

            st.write(
                "Default Probability:",
                result["default_probability"]
            )

            st.write(
                "Risk Level:",
                result["risk_level"]
            )

            st.write(
                "Default Prediction:",
                result["default_prediction"]
            )

            st.write(
                "Customer ID:",
                result["customer_id"]
            )

            st.write(
                "Loan ID:",
                result["loan_id"]
            )

            # SHAP explanation
            st.divider()
            st.subheader("🔎 Why did the model make this prediction?")

            input_df = pd.DataFrame([data])

            shap_values, input_transformed, feature_names = explain_loan(
                input_df
            )

            if isinstance(shap_values, list):
                values = shap_values[1][0]
            else:
                values = shap_values[0]

            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": values
            })

            # Remove preprocessing prefixes
            shap_df["Feature"] = (
                shap_df["Feature"]
                .str.replace("num__", "", regex=False)
                .str.replace("cat__", "", regex=False)
            )

            # Calculate importance
            shap_df["Importance"] = (
                shap_df["SHAP Value"].abs()
            )

            # Top 10 features
            shap_df = shap_df.sort_values(
                "Importance",
                ascending=False
            ).head(10)

            # Explain direction
            shap_df["Effect"] = shap_df["SHAP Value"].apply(
                lambda x:
                    "⬆️ Toward Default"
                    if x > 0
                    else "⬇️ Away from Default"
            )

            st.write(
                "### Top factors influencing this prediction"
            )

            st.dataframe(
                shap_df[
                    ["Feature", "SHAP Value", "Effect"]
                ],
                use_container_width=True,
                hide_index=True
            )

            st.write("### SHAP Contribution")

            st.bar_chart(
                shap_df.set_index("Feature")["SHAP Value"]
            )

            st.caption(
                "Positive SHAP values push the prediction "
                "toward default, while negative values push "
                "it away from default."
            )

        else:
            st.error(
                f"API Error: {response.status_code}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. "
            "Make sure the FastAPI server is running."
        )