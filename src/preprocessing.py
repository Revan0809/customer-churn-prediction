"""
Preprocessing pipeline for the Telco Customer Churn dataset.

Responsibilities:
  - Load and clean the raw CSV (missing values, dtype fixes)
  - Build a scikit-learn ColumnTransformer (impute + scale numeric,
    impute + one-hot encode categorical) that can be fit once on
    training data and reused identically at inference time
  - Apply SMOTE to the *transformed training set only* so evaluation
    always happens on the real, imbalanced class distribution
  - Persist / load the fitted preprocessor with joblib so the API
    never needs to retrain anything at request time
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERIC_FEATURES,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
)


def load_data(path) -> pd.DataFrame:
    """Load the raw Telco churn CSV and fix known data-quality issues.

    Known issue: TotalCharges is stored as a string column and contains
    11 blank-string rows (customers with tenure == 0, i.e. brand new
    accounts that haven't been billed yet). We coerce to numeric and
    fill those with 0, since that's the true billed amount so far.
    """
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)

    # SeniorCitizen arrives as int 0/1 already; cast to a plain Python
    # object dtype so OneHotEncoder treats it as categorical rather than
    # a magnitude, consistently with how a single JSON record from the
    # API will present it.
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    return df


def split_X_y(df: pd.DataFrame):
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy() if TARGET_COLUMN in df.columns else None
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build (but do not fit) the feature preprocessing ColumnTransformer."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Human-readable output feature names, post one-hot encoding."""
    return list(preprocessor.get_feature_names_out())


def apply_smote(X_train: np.ndarray, y_train: np.ndarray):
    """Oversample the minority (churn) class on the *training* split only.

    Must never be called on validation/test data: doing so would let
    synthetic neighbors leak information across the split and produce
    metrics that look better than real-world performance.
    """
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def save_preprocessor(preprocessor: ColumnTransformer, path=PREPROCESSOR_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path=PREPROCESSOR_PATH) -> ColumnTransformer:
    return joblib.load(path)


def transform_record(preprocessor: ColumnTransformer, record: dict) -> np.ndarray:
    """Transform a single raw customer record (as a dict, e.g. from the API)
    into the model's numeric feature space using the fitted preprocessor.
    """
    df = pd.DataFrame([record])[ALL_FEATURES]
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    return preprocessor.transform(df)
