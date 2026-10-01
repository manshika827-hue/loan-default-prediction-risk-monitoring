import joblib
import pandas as pd

model = joblib.load("04_models/loan_default_model.joblib")

data = pd.DataFrame([{
    "Age": 35,
    "Income": 50000,
    "LoanAmount": 15000,
    "CreditScore": 700,
    "MonthsEmployed": 60,
    "NumCreditLines": 3,
    "InterestRate": 8.5,
    "LoanTerm": 36,
    "DTIRatio": 0.30,
    "Education": "Bachelor's",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Married",
    "HasMortgage": "Yes",
    "HasDependents": "No",
    "LoanPurpose": "Home",
    "HasCoSigner": "No"
}])

prediction = model.predict(data)[0]
probability = model.predict_proba(data)[0][1]

print("MODEL TEST SUCCESSFUL")
print("Prediction:", prediction)
print("Probability:", probability)