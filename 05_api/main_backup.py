from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


# Create FastAPI application
app = FastAPI(
    title="Loan Default Prediction API",
    description="API for predicting loan default risk",
    version="1.0"
)


# Load trained ML pipeline
model = joblib.load("04_models/loan_default_model.joblib")


# Input data structure
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


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Loan Default Prediction API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict_loan(data: LoanInput):

    # Convert input into Pandas DataFrame
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

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get probability of default
    probability = model.predict_proba(input_data)[0][1]

    # Determine risk level
    if probability >= 0.70:
        risk = "High"
    elif probability >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    # Return result
    return {
        "default_prediction": int(prediction),
        "default_probability": round(float(probability), 4),
        "risk_level": risk
    }