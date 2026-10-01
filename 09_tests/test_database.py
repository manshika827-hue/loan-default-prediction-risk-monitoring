import os
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text

load_dotenv()

password = os.getenv("POSTGRES_PASSWORD")
encoded_password = quote_plus(password)

DATABASE_URL = (
    f"postgresql+psycopg://postgres:{encoded_password}"
    "@localhost:1737/loan_prediction_db"
)

engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT version();"))
        version = result.fetchone()[0]

        print("DATABASE CONNECTION SUCCESSFUL")
        print("PostgreSQL:", version)

except Exception as e:
    print("DATABASE CONNECTION FAILED")
    print(e)