# Loan Default Prediction and Risk Monitoring System

## 1. Project Overview

The **Loan Default Prediction and Risk Monitoring System** is an end-to-end machine learning application designed to predict the probability of loan default and classify loan applications into different risk levels.

The system integrates machine learning, data preprocessing, a REST API, a web dashboard, PostgreSQL database storage, Explainable AI using SHAP, and data drift monitoring using Evidently.

Instead of using a machine learning model only inside a notebook, this project implements a complete application workflow.

A user enters borrower and loan information through the Streamlit dashboard. The information is sent to the FastAPI backend, where the trained machine learning pipeline processes the input and generates a default prediction and default probability.

The application data and prediction results are stored in PostgreSQL. SHAP is used to explain the individual prediction, while Evidently is used to monitor changes between historical data and current loan application data.

---

## 2. Problem Statement

Loan default is an important challenge for financial institutions because inaccurate risk assessment can result in financial losses and inefficient lending decisions.

Manual loan assessment can be time-consuming and may not consistently identify relationships between different borrower and loan characteristics.

This project aims to develop a machine learning-based system that can:

- Analyze historical loan application data.
- Identify patterns associated with loan default.
- Predict the probability of loan default.
- Classify loan applications into risk levels.
- Store application and prediction information.
- Explain individual predictions.
- Monitor changes in incoming application data.

---

## 3. Project Objectives

The main objectives of this project are:

- Perform exploratory data analysis on historical loan data.
- Clean and preprocess the dataset.
- Handle numerical and categorical features.
- Train multiple machine learning models.
- Compare Logistic Regression, Random Forest, and XGBoost.
- Tune the XGBoost model.
- Evaluate models using ROC-AUC, recall, precision, and F1-score.
- Save the trained machine learning pipeline.
- Develop a FastAPI REST API.
- Develop a Streamlit web dashboard.
- Store customer, loan, and prediction data in PostgreSQL.
- Provide individual prediction explanations using SHAP.
- Monitor data drift using Evidently.
- Protect database credentials using environment variables.

---

## 4. System Architecture

```text
                         USER
                           |
                           v
                 +-------------------+
                 | Streamlit         |
                 | Dashboard         |
                 +-------------------+
                           |
                           | HTTP Request
                           v
                 +-------------------+
                 | FastAPI Backend   |
                 +-------------------+
                           |
                           v
                 +-------------------+
                 | Input Validation  |
                 | & Preprocessing   |
                 +-------------------+
                           |
                           v
                 +-------------------+
                 | Tuned XGBoost     |
                 | Model             |
                 +-------------------+
                           |
                           v
              +--------------------------+
              | Default Prediction       |
              | Default Probability     |
              | Risk Level               |
              +--------------------------+
                    |              |
                    |              |
                    v              v
          +----------------+  +----------------+
          | PostgreSQL     |  | SHAP           |
          | Database       |  | Explanation    |
          +----------------+  +----------------+
                    |
                    v
          +------------------------+
          | Evidently Monitoring   |
          | Data Drift Analysis    |
          +------------------------+
```

---

## 5. End-to-End Workflow

```text
Historical Loan Data
        |
        v
Data Cleaning
        |
        v
Exploratory Data Analysis
        |
        v
Feature Selection
        |
        v
Train/Test Split
        |
        v
Data Preprocessing
        |
        v
Model Training
        |
        v
Model Comparison
        |
        v
XGBoost Tuning
        |
        v
Final Model
        |
        v
Joblib Model Saving
        |
        v
FastAPI Backend
        |
        v
Streamlit Dashboard
        |
        v
PostgreSQL Database
        |
        +------> SHAP Explanation
        |
        +------> Evidently Monitoring
```

---

## 6. Dataset

The project uses the following dataset:

```text
Loan_default.csv
```

Dataset location:

```text
01_data/Loan_default.csv
```

Dataset details:

- Number of records: **255,347**
- Number of columns: **18**
- Target variable: `Default`

Target interpretation:

```text
Default = 0 → Non-default
Default = 1 → Default
```

The `LoanID` column is removed before model training because it is an identifier rather than a predictive feature.

---

## 7. Dataset Features

| Feature | Description |
|---|---|
| LoanID | Unique loan identifier |
| Age | Borrower's age |
| Income | Borrower's income |
| LoanAmount | Requested loan amount |
| CreditScore | Borrower's credit score |
| MonthsEmployed | Number of months employed |
| NumCreditLines | Number of existing credit lines |
| InterestRate | Loan interest rate |
| LoanTerm | Loan duration |
| DTIRatio | Debt-to-income ratio |
| Education | Borrower's education level |
| EmploymentType | Employment category |
| MaritalStatus | Marital status |
| HasMortgage | Whether the borrower has a mortgage |
| HasDependents | Whether the borrower has dependents |
| LoanPurpose | Purpose of the loan |
| HasCoSigner | Whether the loan has a co-signer |
| Default | Target variable |

---

## 8. Data Analysis

Exploratory Data Analysis was performed to understand:

- Dataset dimensions
- Data types
- Missing values
- Numerical feature distributions
- Categorical feature distributions
- Target distribution
- Relationships between numerical features

The dataset contains no missing values in the analyzed data.

The target distribution is imbalanced:

```text
Default = 0 : 225,694
Default = 1 : 29,653
```

Approximate distribution:

```text
Non-default : 88.39%
Default     : 11.61%
```

Because the target variable is imbalanced, accuracy alone is not sufficient for evaluating the model.

The project therefore considers:

- ROC-AUC
- Recall
- Precision
- F1-score

---

## 9. Data Preprocessing

The `LoanID` column is removed because it is an identifier.

The target variable is separated from the input features.

```text
X = Input Features
y = Default
```

The dataset is divided into training and testing datasets using an 80:20 split.

A stratified split is used to preserve the target distribution between training and testing datasets.

---

## 10. Numerical Features

The numerical features are:

```text
Age
Income
LoanAmount
CreditScore
MonthsEmployed
NumCreditLines
InterestRate
LoanTerm
DTIRatio
```

Numerical features are standardized using:

```text
StandardScaler
```

---

## 11. Categorical Features

The categorical features are:

```text
Education
EmploymentType
MaritalStatus
HasMortgage
HasDependents
LoanPurpose
HasCoSigner
```

Categorical features are transformed using:

```text
OneHotEncoder
```

The encoder uses:

```text
handle_unknown="ignore"
```

This allows the model pipeline to process previously unseen categorical values.

---

## 12. Preprocessing Pipeline

The preprocessing workflow is:

```text
Numerical Features
        |
        v
StandardScaler
        |
        +----------------+
                         |
Categorical Features     |
        |                |
        v                |
OneHotEncoder            |
        |                |
        +-------+--------+
                |
                v
        Processed Features
```

The resulting processed dataset contains:

```text
31 features
```

Training data:

```text
204,277 rows × 31 features
```

Testing data:

```text
51,070 rows × 31 features
```

---

## 13. Machine Learning Models

The project evaluates the following machine learning models:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. Tuned XGBoost

The models are compared using evaluation metrics suitable for the imbalanced classification problem.

---

## 14. Model Evaluation

The models were evaluated using:

- ROC-AUC
- Default Recall
- Precision
- F1-score

Results:

| Model | ROC-AUC | Default Recall | Default F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.7532 | 0.70 | 0.33 |
| Random Forest | 0.7444 | 0.01 | 0.02 |
| XGBoost | 0.7521 | 0.65 | 0.34 |
| Tuned XGBoost | **0.7589** | **0.69** | **0.34** |

The tuned XGBoost model achieved a test ROC-AUC of:

```text
0.7589
```

The initial project target was a ROC-AUC of 0.85. The current validation result of 0.7589 does not reach that target.

Further feature engineering, hyperparameter optimization, and additional validation can be explored in future work.

---

## 15. Class Imbalance

The dataset contains substantially more non-default cases than default cases.

Therefore, accuracy alone may provide a misleading view of model performance.

The project focuses on:

- ROC-AUC
- Recall
- Precision
- F1-score

XGBoost was also configured using class weighting to give additional importance to the minority default class.

---

## 16. Final Model

The final trained pipeline contains:

```text
Preprocessing
     +
Tuned XGBoost Model
```

The complete pipeline is saved using Joblib:

```text
04_models/loan_default_model.joblib
```

Saving the preprocessing and model together ensures that the same preprocessing steps used during training are applied during real-time predictions.

---

## 17. FastAPI Backend

FastAPI is used to provide the machine learning model as a REST API.

The backend is responsible for:

- Receiving loan application data.
- Validating input.
- Loading the trained model.
- Generating predictions.
- Calculating default probability.
- Assigning risk level.
- Storing application information.
- Storing prediction information.
- Returning the prediction result.

Start FastAPI using:

```powershell
uvicorn 05_api.main:app --reload
```

The API runs at:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 18. API Endpoints

### GET `/`

Checks whether the API is running.

### GET `/db-test`

Tests the connection between the FastAPI backend and PostgreSQL.

### POST `/predict`

Accepts loan application information and returns:

- Customer ID
- Loan ID
- Default prediction
- Default probability
- Risk level
- Database storage status

---

## 19. Prediction Workflow

```text
User Input
    |
    v
FastAPI Validation
    |
    v
Pandas DataFrame
    |
    v
Saved ML Pipeline
    |
    v
Default Prediction
    |
    v
Default Probability
    |
    v
Risk Classification
    |
    +------> PostgreSQL
    |
    +------> SHAP Explanation
```

---

## 20. Risk Classification

The application currently uses the following probability thresholds:

```text
Probability < 0.40
        |
        v
      Low


0.40 ≤ Probability < 0.70
        |
        v
     Medium


Probability ≥ 0.70
        |
        v
      High
```

These thresholds are application-level rules and can be adjusted in future versions after additional validation.

---

## 21. Streamlit Dashboard

Streamlit provides the user interface for the system.

The dashboard accepts applicant and loan information.

### Applicant Information

- Age
- Income
- Credit Score
- Education
- Employment Type
- Marital Status
- Has Mortgage
- Has Dependents

### Loan Information

- Loan Amount
- Months Employed
- Number of Credit Lines
- Interest Rate
- Loan Term
- DTI Ratio
- Loan Purpose
- Has Co-Signer

The dashboard sends this information to the FastAPI backend.

---

## 22. Running Streamlit

Start the dashboard using:

```powershell
python -m streamlit run 07_dashboard\app.py
```

The dashboard is available at:

```text
http://localhost:8501
```

The dashboard displays:

- Default prediction
- Default probability
- Risk level
- Top factors influencing the prediction
- SHAP contribution chart

---

## 23. PostgreSQL Database

PostgreSQL is used to store application and prediction information.

The database contains three main tables:

```text
customer
     |
     v
loan_application
     |
     v
loan_prediction
```

---

## 24. Database Tables

### Customer

The `customer` table stores:

- Customer ID
- Age
- Income
- Credit Score
- Education
- Employment Type
- Marital Status
- Mortgage Status
- Dependents Status

### Loan Application

The `loan_application` table stores:

- Loan ID
- Customer ID
- Loan Amount
- Interest Rate
- Loan Term
- DTI Ratio
- Number of Credit Lines
- Months Employed
- Loan Purpose
- Co-Signer Status

### Loan Prediction

The `loan_prediction` table stores:

- Prediction ID
- Loan ID
- Default Prediction
- Default Probability
- Risk Level
- Prediction Time

---

## 25. Database Relationship

```text
+-------------+
|  customer   |
+-------------+
       |
       | 1
       |
       | many
       v
+--------------------+
| loan_application   |
+--------------------+
       |
       | 1
       |
       | many
       v
+--------------------+
| loan_prediction    |
+--------------------+
```

The system has successfully stored real prediction records in PostgreSQL.

The current database validation contains 11 records in each of the main customer, loan application, and prediction tables.

---

## 26. Explainable AI Using SHAP

Machine learning models can be difficult to interpret.

To improve interpretability, the project uses:

```text
SHAP
```

SHAP stands for:

```text
SHapley Additive exPlanations
```

SHAP provides feature-level explanations for individual predictions.

The dashboard displays:

- Feature
- SHAP value
- Effect on prediction

A positive SHAP value indicates that the individual feature contribution moves the prediction toward the default class.

A negative SHAP value indicates that the individual feature contribution moves the prediction away from the default class.

These explanations describe the model's behavior for an individual prediction and should not be interpreted as proof of causal relationships.

---

## 27. SHAP Workflow

```text
Loan Application
       |
       v
Preprocessing
       |
       v
Tuned XGBoost
       |
       v
SHAP TreeExplainer
       |
       v
Feature Contributions
       |
       v
Top Influential Features
       |
       v
Streamlit Dashboard
```

SHAP implementation:

```text
08_monitoring/shap_explanation.py
```

---

## 28. Data Drift Monitoring

Machine learning models are trained using historical data.

Over time, incoming real-world data may change.

For example:

- Income distributions may change.
- Loan amounts may change.
- Credit scores may change.
- Loan purposes may change.
- Employment patterns may change.

Such changes can affect the reliability of a machine learning model.

Therefore, this project uses:

```text
Evidently
```

for data drift monitoring.

---

## 29. Evidently Monitoring Workflow

```text
Historical Loan Dataset
          |
          | Reference Data
          v
     +-----------+
     | Evidently |
     +-----------+
          ^
          |
          | Current Data
          |
    PostgreSQL
          |
          v
Current Loan Applications
```

The monitoring script retrieves current loan application data from PostgreSQL and compares it with the historical dataset.

Monitoring script:

```text
08_monitoring/evidently_monitoring.py
```

Run:

```powershell
python 08_monitoring\evidently_monitoring.py
```

The generated report is:

```text
08_monitoring/data_drift_report.html
```

---

## 30. Current Monitoring Status

The Evidently monitoring pipeline has been successfully executed.

Current validation:

```text
Current prediction records found: 11
```

The HTML data drift report is generated successfully.

Because the current application dataset contains only 11 records compared with the much larger historical dataset, the current drift analysis should be considered preliminary.

A larger production sample would provide a more reliable assessment of data drift.

---

## 31. Environment Variables

Database credentials are stored using environment variables.

The project uses a `.env` file:

```text
POSTGRES_PASSWORD=your_postgres_password
```

The Python application loads the value using:

```python
from dotenv import load_dotenv
import os

load_dotenv()

password = os.getenv("POSTGRES_PASSWORD")
```

The actual password must never be committed to source control.

---

## 32. Security

The project follows basic security practices:

- Database credentials are stored in `.env`.
- `.env` is excluded from Git.
- Virtual environment is excluded from Git.
- API input is validated using Pydantic.
- Database credentials are not hard-coded in application source code.
- Frontend and backend are separated.

---

## 33. Project Structure

```text
HCLproject/
│
├── 01_data/
│   └── Loan_default.csv
│
├── 02_notebooks/
│   └── 01_eda.ipynb
│
├── 03_src/
│   └── data_processing.py
│
├── 04_models/
│   └── loan_default_model.joblib
│
├── 05_api/
│   └── main.py
│
├── 06_database/
│   └── database.py
│
├── 07_dashboard/
│   └── app.py
│
├── 08_monitoring/
│   ├── shap_explanation.py
│   ├── evidently_monitoring.py
│   └── data_drift_report.html
│
├── 09_tests/
│
├── .env
├── .gitignore
├── README.md
├── requirements.txt
└── venv/
```

---

## 34. Technologies Used

### Programming Language

- Python 3.12

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- XGBoost

### Data Visualization

- Matplotlib
- Seaborn
- Streamlit

### Explainable AI

- SHAP

### Backend

- FastAPI
- Uvicorn
- Pydantic

### Database

- PostgreSQL
- SQLAlchemy
- Psycopg

### Monitoring

- Evidently

### Model Serialization

- Joblib

### Configuration

- Python-dotenv

---

## 35. Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd HCLproject
```

### Step 2: Create Virtual Environment

The project uses Python 3.12.

```powershell
py -3.12 -m venv venv
```

### Step 3: Activate Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 5: Configure Environment Variables

Create a `.env` file in the project root:

```text
POSTGRES_PASSWORD=your_postgres_password
```

Do not upload the `.env` file to GitHub.

---

## 36. Running the Complete Project

### Terminal 1 - FastAPI

```powershell
uvicorn 05_api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

### Terminal 2 - Streamlit

```powershell
python -m streamlit run 07_dashboard\app.py
```

Open:

```text
http://localhost:8501
```

### Terminal 3 - Evidently Monitoring

```powershell
python 08_monitoring\evidently_monitoring.py
```

---

## 37. API Testing

FastAPI provides interactive API documentation through Swagger.

Open:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:

```text
GET  /
GET  /db-test
POST /predict
```

The `/db-test` endpoint can be used to verify PostgreSQL connectivity.

The `/predict` endpoint can be used to test the complete machine learning prediction pipeline.

---

## 38. Functional Requirements

### FR-1: Data Processing

The system shall load and preprocess historical loan data.

### FR-2: Prediction

The system shall generate a loan default prediction.

### FR-3: Probability

The system shall return the predicted default probability.

### FR-4: Risk Classification

The system shall classify loan applications into Low, Medium, or High risk.

### FR-5: Database Storage

The system shall store customer, loan application, and prediction information.

### FR-6: Explainability

The system shall provide SHAP-based explanations for individual predictions.

### FR-7: Monitoring

The system shall generate data drift reports using Evidently.

---

## 39. Non-Functional Requirements

### Performance

The API should provide prediction results quickly enough for interactive dashboard use.

### Reliability

The system should provide consistent results for valid input data.

### Maintainability

The application is separated into modules for:

- Data processing
- Machine learning
- API
- Database
- Dashboard
- Monitoring

### Security

Sensitive database credentials are stored using environment variables.

### Scalability

The modular architecture allows the system to be extended with additional services and monitoring functionality.

### Explainability

SHAP provides feature-level information for individual model predictions.

---

## 40. Advantages

The system provides:

1. Automated loan risk prediction.
2. Real-time prediction through FastAPI.
3. User-friendly Streamlit interface.
4. Centralized PostgreSQL storage.
5. Explainable AI using SHAP.
6. Data drift monitoring using Evidently.
7. Modular project architecture.
8. Environment-based credential management.
9. Reusable machine learning pipeline.
10. Separation of frontend, backend, model, database, and monitoring components.

---

## 41. Limitations

The current implementation has the following limitations:

- The current test ROC-AUC is 0.7589, below the initial target of 0.85.
- The dataset contains class imbalance.
- The current monitoring sample contains only 11 application records.
- Current drift analysis should therefore be considered preliminary.
- Automated model retraining has not been implemented.
- Authentication and authorization have not been implemented.
- Risk thresholds are application-defined and may require additional validation.
- The current model may require additional feature engineering to improve performance.

---

## 42. Future Enhancements

### Machine Learning Improvements

- Additional feature engineering.
- Advanced hyperparameter optimization.
- Cross-validation.
- Model calibration.
- Threshold optimization.
- Additional machine learning algorithms.
- Ensemble methods.

### Monitoring Improvements

- Larger production datasets.
- Automated monitoring schedules.
- Model performance monitoring.
- Prediction distribution monitoring.
- Automated drift alerts.
- Monitoring dashboards.

### Application Improvements

- User authentication.
- Role-based access control.
- Admin dashboard.
- Loan history.
- Prediction history.
- Exportable prediction reports.

### MLOps Improvements

- Automated model retraining.
- Model versioning.
- Model registry.
- CI/CD pipeline.
- Automated testing.
- Production deployment.

---

## 43. Testing

The project can be tested at multiple levels.

### Data Testing

- Dataset shape validation.
- Missing-value checking.
- Data-type validation.
- Target distribution checking.

### API Testing

- Test `/`.
- Test `/db-test`.
- Test `/predict`.

### Database Testing

- PostgreSQL connection testing.
- Customer record verification.
- Loan application verification.
- Prediction record verification.

### Model Testing

- ROC-AUC evaluation.
- Recall evaluation.
- Precision evaluation.
- F1-score evaluation.

### Dashboard Testing

- Submit valid loan applications.
- Verify prediction output.
- Verify default probability.
- Verify risk level.
- Verify SHAP explanation.

### Monitoring Testing

- Run Evidently monitoring.
- Verify PostgreSQL records are retrieved.
- Verify HTML report generation.

---

## 44. Current Project Status

The following components have been implemented:

- [x] Dataset loading
- [x] Data cleaning
- [x] Exploratory Data Analysis
- [x] Target analysis
- [x] Numerical preprocessing
- [x] Categorical preprocessing
- [x] Train/test split
- [x] Logistic Regression
- [x] Random Forest
- [x] XGBoost
- [x] XGBoost tuning
- [x] Model evaluation
- [x] Model saving
- [x] FastAPI backend
- [x] `/predict` endpoint
- [x] `/db-test` endpoint
- [x] PostgreSQL integration
- [x] Streamlit dashboard
- [x] SHAP explanation
- [x] Evidently monitoring
- [x] Environment variable configuration
- [x] `.gitignore`
- [x] `requirements.txt`
- [x] Project documentation

---

## 45. Project Results

The current model evaluation results are:

```text
Logistic Regression
ROC-AUC: 0.7532

Random Forest
ROC-AUC: 0.7444

XGBoost
ROC-AUC: 0.7521

Tuned XGBoost
ROC-AUC: 0.7589
```

The tuned XGBoost model currently has the highest ROC-AUC among the evaluated models.

The trained pipeline has been integrated into the FastAPI prediction service.

The complete application workflow has also been integrated with PostgreSQL, Streamlit, SHAP, and Evidently.

---

## 46. Conclusion

The **Loan Default Prediction and Risk Monitoring System** demonstrates how a machine learning model can be transformed into a complete application rather than being used only as an offline notebook.

The project integrates:

```text
Machine Learning
       +
Data Preprocessing
       +
FastAPI
       +
Streamlit
       +
PostgreSQL
       +
SHAP
       +
Evidently
```

The system can accept loan application information through a web dashboard, send it to the FastAPI backend, process the information using the trained machine learning pipeline, generate a default prediction and probability, classify the application into a risk level, and store the result in PostgreSQL.

The system also provides SHAP-based explanations for individual predictions and uses Evidently to monitor changes in current application data.

The tuned XGBoost model achieved a test ROC-AUC of **0.7589**. The initial project target was 0.85 ROC-AUC, so additional model improvement remains a future objective.

Overall, the project provides an end-to-end foundation for a machine learning-based loan risk assessment and monitoring system.

---

## 47. Author

**Manshika**

B.Tech - Computer Science and Engineering (Data Science)

Academic Year: **2026-2027**

---

## 48. License

This project is developed for academic and educational purposes.