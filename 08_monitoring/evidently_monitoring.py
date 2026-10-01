import os
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset
from sqlalchemy import create_engine, text


# ==========================================
# 1. Load environment variables
# ==========================================

load_dotenv()

password = os.getenv("POSTGRES_PASSWORD")

if not password:
    raise ValueError(
        "POSTGRES_PASSWORD not found in .env file."
    )

encoded_password = quote_plus(password)


# ==========================================
# 2. Load historical data
# ==========================================

df = pd.read_csv(
    "01_data/Loan_default.csv"
)

# Remove ID and target.
# We only want to monitor input features.
reference_data = df.drop(
    ["LoanID", "Default"],
    axis=1
)


# ==========================================
# 3. Connect to PostgreSQL
# ==========================================

DATABASE_URL = (
    f"postgresql+psycopg://postgres:{encoded_password}"
    "@localhost:1737/loan_prediction_db"
)

engine = create_engine(
    DATABASE_URL
)


# ==========================================
# 4. Get real-time application data
# ==========================================

query = """
SELECT
    c.age AS "Age",
    c.income AS "Income",
    la.loan_amount AS "LoanAmount",
    c.credit_score AS "CreditScore",
    la.months_employed AS "MonthsEmployed",
    la.num_credit_lines AS "NumCreditLines",
    la.interest_rate AS "InterestRate",
    la.loan_term AS "LoanTerm",
    la.dti_ratio AS "DTIRatio",
    c.education AS "Education",
    c.employment_type AS "EmploymentType",
    c.marital_status AS "MaritalStatus",
    c.has_mortgage AS "HasMortgage",
    c.has_dependents AS "HasDependents",
    la.loan_purpose AS "LoanPurpose",
    la.has_cosigner AS "HasCoSigner"
FROM customer c
JOIN loan_application la
    ON c.customer_id = la.customer_id
"""


with engine.connect() as connection:

    current_data = pd.read_sql(
        text(query),
        connection
    )


# ==========================================
# 5. Check current data
# ==========================================

if len(current_data) == 0:

    print(
        "No real-time loan applications found."
    )

    print(
        "Make at least one prediction through Streamlit."
    )

    exit()


print(
    f"Current prediction records found: "
    f"{len(current_data)}"
)


# ==========================================
# 6. Create Evidently drift report
# ==========================================

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)


snapshot = report.run(
    reference_data=reference_data,
    current_data=current_data
)


# ==========================================
# 7. Save HTML report
# ==========================================

snapshot.save_html(
    "08_monitoring/data_drift_report.html"
)


# ==========================================
# 8. Success message
# ==========================================

print(
    "Evidently monitoring report "
    "created successfully!"
)

print(
    "Open: 08_monitoring/data_drift_report.html"
)