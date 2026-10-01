from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
 
# -----------------------------
# FastAPI Application
# -----------------------------
app = FastAPI(
    title="Loan Default Prediction API",
    description="API for predicting loan default risk",
    version="1.0"
)


# -----------------------------
# PostgreSQL Database Connection
# -----------------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

# Render provides postgresql://...
# SQLAlchemy + psycopg uses postgresql+psycopg://
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://",
        "postgresql+psycopg://",
        1
    )

engine = create_engine(DATABASE_URL))

# -----------------------------
# Load ML Model
# -----------------------------
model = joblib.load("04_models/loan_default_model.joblib")


# -----------------------------
# Input Data Structure
# -----------------------------
class LoanInput(BaseModel):
    Age: float
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: float
    NumCreditLines: float
    InterestRate: float
    LoanTerm: float
    DTIRatio: float
    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str


# -----------------------------
# Home Endpoint
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Loan Default Prediction API is running"
    }


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict_loan(data: LoanInput):

    input_data = pd.DataFrame([{
        "Age": data.Age,
        "Income": data.Income,
        "LoanAmount": data.LoanAmount,
        "CreditScore": data.CreditScore,
        "MonthsEmployed": data.MonthsEmployed,
        "NumCreditLines": data.NumCreditLines,
        "InterestRate": data.InterestRate,
        "LoanTerm": data.LoanTerm,
        "DTIRatio": data.DTIRatio,
        "Education": data.Education,
        "EmploymentType": data.EmploymentType,
        "MaritalStatus": data.MaritalStatus,
        "HasMortgage": data.HasMortgage,
        "HasDependents": data.HasDependents,
        "LoanPurpose": data.LoanPurpose,
        "HasCoSigner": data.HasCoSigner
    }])

    # ML prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Risk level
    if probability >= 0.70:
        risk = "High"
    elif probability >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    # Save everything to PostgreSQL
    with engine.begin() as connection:

        # 1. Customer
        customer_result = connection.execute(
            text("""
                INSERT INTO customer (
                    age,
                    income,
                    credit_score,
                    education,
                    employment_type,
                    marital_status,
                    has_mortgage,
                    has_dependents
                )
                VALUES (
                    :age,
                    :income,
                    :credit_score,
                    :education,
                    :employment_type,
                    :marital_status,
                    :has_mortgage,
                    :has_dependents
                )
                RETURNING customer_id
            """),
            {
                "age": data.Age,
                "income": data.Income,
                "credit_score": data.CreditScore,
                "education": data.Education,
                "employment_type": data.EmploymentType,
                "marital_status": data.MaritalStatus,
                "has_mortgage": data.HasMortgage,
                "has_dependents": data.HasDependents
            }
        )

        customer_id = customer_result.fetchone()[0]

        # 2. Loan application
        loan_result = connection.execute(
            text("""
                INSERT INTO loan_application (
                    customer_id,
                    loan_amount,
                    interest_rate,
                    loan_term,
                    dti_ratio,
                    num_credit_lines,
                    months_employed,
                    loan_purpose,
                    has_cosigner
                )
                VALUES (
                    :customer_id,
                    :loan_amount,
                    :interest_rate,
                    :loan_term,
                    :dti_ratio,
                    :num_credit_lines,
                    :months_employed,
                    :loan_purpose,
                    :has_cosigner
                )
                RETURNING loan_id
            """),
            {
                "customer_id": customer_id,
                "loan_amount": data.LoanAmount,
                "interest_rate": data.InterestRate,
                "loan_term": data.LoanTerm,
                "dti_ratio": data.DTIRatio,
                "num_credit_lines": data.NumCreditLines,
                "months_employed": data.MonthsEmployed,
                "loan_purpose": data.LoanPurpose,
                "has_cosigner": data.HasCoSigner
            }
        )

        loan_id = loan_result.fetchone()[0]

        # 3. Prediction
        connection.execute(
            text("""
                INSERT INTO loan_prediction (
                    loan_id,
                    default_prediction,
                    default_probability,
                    risk_level
                )
                VALUES (
                    :loan_id,
                    :default_prediction,
                    :default_probability,
                    :risk_level
                )
            """),
            {
                "loan_id": loan_id,
                "default_prediction": int(prediction),
                "default_probability": float(probability),
                "risk_level": risk
            }
        )

    return {
        "customer_id": customer_id,
        "loan_id": loan_id,
        "default_prediction": int(prediction),
        "default_probability": round(float(probability), 4),
        "risk_level": risk,
        "database": "prediction saved successfully"
    }
@app.get("/db-test")
def database_test():
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT version();"))
            version = result.fetchone()[0]

        return {
            "database": "connected",
            "version": version
        }

    except Exception as e:
        return {
            "database": "connection failed",
            "error": str(e)
        }