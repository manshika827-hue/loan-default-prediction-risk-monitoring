from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

password = "VERMA@123"

encoded_password = quote_plus(password)

DATABASE_URL = (
    f"postgresql+psycopg://postgres:{encoded_password}"
    "@localhost:1737/loan_prediction_db"
)

engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT version();"))

        print("Database connected successfully!")
        print(result.fetchone()[0])

except Exception as e:
    print("Database connection failed!")
    print(e)