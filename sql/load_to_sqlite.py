"""
Loads the raw Telco churn CSV into a local SQLite database for SQL/BI analysis.

This is a separate, additive path from src/preprocessing.py: it does not fit
any encoders or feed the ML pipeline, it just gets clean, snake_case,
analyst-friendly data into `sql/churn.db` so it can be queried directly with
SQL (queries.sql / run_queries.py) or from the Streamlit dashboard.

Run with:  python sql/load_to_sqlite.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "telco_churn.csv"
DB_PATH = Path(__file__).resolve().parent / "churn.db"
TABLE_NAME = "customers"

# Raw CSV column -> clean snake_case column. Explicit mapping (rather than a
# regex) so acronyms like "ID" don't get mangled and the schema is obvious
# at a glance.
COLUMN_RENAME_MAP = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Same known data-quality fix as src/preprocessing.py: TotalCharges is a
    # string column with 11 blank rows for brand-new (tenure == 0) customers.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df = df.rename(columns=COLUMN_RENAME_MAP)
    return df


def write_to_sqlite(df: pd.DataFrame, db_path: Path, table_name: str) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_churn ON {table_name}(churn)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_contract ON {table_name}(contract)"
        )


def main():
    print(f"Loading {DATA_PATH} ...")
    df = load_and_clean(DATA_PATH)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns.")

    print(f"Writing table '{TABLE_NAME}' to {DB_PATH} ...")
    write_to_sqlite(df, DB_PATH, TABLE_NAME)
    print("Done.")


if __name__ == "__main__":
    main()
