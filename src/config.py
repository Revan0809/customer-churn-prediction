"""
Shared constants for the churn pipeline.

Single source of truth for column names and category options so that
preprocessing, training, explainability, and the API request schema
can never drift out of sync with each other.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "telco_churn.csv"
MODELS_DIR = ROOT_DIR / "models"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
COMPARISON_CSV_PATH = MODELS_DIR / "model_comparison.csv"
SHAP_SUMMARY_PLOT_PATH = MODELS_DIR / "shap_summary.png"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.joblib"

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# All treated as categorical (one-hot encoded), including the 0/1 flag
# SeniorCitizen, which is encoded as an int in the raw data but is
# semantically a category, not a magnitude.
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Allowed values per categorical field. Used both to build the frontend
# dropdowns' contract and to validate API input. Mirrors the raw values
# found in the IBM Telco Customer Churn dataset.
CATEGORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

RANDOM_STATE = 42
