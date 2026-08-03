"""
SHAP-based explainability for the trained churn model.

Provides:
  - A CLI entry point (`python -m src.explain`) that computes SHAP values
    for a sample of the test set and saves a global summary plot.
  - `build_explainer` / `get_top_contributing_features`, used by the
    FastAPI backend to explain a single live prediction.

SHAP explainer choice: `shap.Explainer` auto-selects the fastest exact
algorithm for the model type we hand it -- TreeExplainer for
RandomForest/XGBoost, LinearExplainer for Logistic Regression -- both of
which are cheap enough to run per-request in the API without a
noticeable latency hit.
"""

from __future__ import annotations

import re

import joblib
import numpy as np
import shap

from src.config import (
    DATA_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    SHAP_SUMMARY_PLOT_PATH,
)
from src.preprocessing import build_preprocessor  # noqa: F401 (kept for type context)


def build_explainer(model, background_data: np.ndarray) -> shap.Explainer:
    """Build a SHAP explainer for `model` using a background sample.

    `background_data` should be a small (e.g. 100-row) sample of
    *preprocessed* training features, used by linear/kernel explainers to
    estimate feature baselines. Tree explainers ignore it.
    """
    return shap.Explainer(model, background_data)


def prettify_feature_name(raw_name: str) -> str:
    """Turn a ColumnTransformer output name into something human-readable.

    Examples:
        "num__tenure"                    -> "tenure"
        "cat__Contract_Month-to-month"   -> "Contract = Month-to-month"
        "cat__SeniorCitizen_1"           -> "SeniorCitizen = 1"
    """
    name = re.sub(r"^(num__|cat__)", "", raw_name)
    for original_col in [
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
    ]:
        prefix = original_col + "_"
        if name.startswith(prefix):
            return f"{original_col} = {name[len(prefix):]}"
    return name


def get_top_contributing_features(
    explainer: shap.Explainer,
    x_row: np.ndarray,
    feature_names: list[str],
    top_n: int = 3,
) -> list[dict]:
    """Return the top-`top_n` features driving a single prediction.

    `x_row` must be a single already-preprocessed row, shape (1, n_features).
    Ranking is by absolute SHAP value (magnitude of impact); the sign tells
    us whether that feature pushed the prediction toward churn (+) or
    toward staying (-).
    """
    shap_values = explainer(x_row)

    values = shap_values.values[0]
    # Binary classifiers via shap.Explainer sometimes return shape
    # (n_features, 2) -- one column per class. Take the churn (class 1) column.
    if values.ndim == 2:
        values = values[:, 1]

    order = np.argsort(-np.abs(values))[:top_n]

    top_features = []
    for idx in order:
        top_features.append(
            {
                "feature": prettify_feature_name(feature_names[idx]),
                "impact": round(float(values[idx]), 4),
                "direction": "increases churn risk" if values[idx] > 0 else "decreases churn risk",
            }
        )
    return top_features


def main():
    import matplotlib

    matplotlib.use("Agg")  # headless: no display available on a server/CI box
    import matplotlib.pyplot as plt

    from src.preprocessing import load_data, load_preprocessor, split_X_y, get_feature_names
    from sklearn.model_selection import train_test_split

    print("Loading model, preprocessor, and data ...")
    model = joblib.load(MODEL_PATH)
    preprocessor = load_preprocessor(PREPROCESSOR_PATH)
    feature_names = get_feature_names(preprocessor)

    df = load_data(DATA_PATH)
    X, y = split_X_y(df)
    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_test_t = preprocessor.transform(X_test)

    background = shap.sample(X_test_t, min(100, X_test_t.shape[0]), random_state=RANDOM_STATE)
    explainer = build_explainer(model, background)

    sample_size = min(200, X_test_t.shape[0])
    sample = X_test_t[:sample_size]
    print(f"Computing SHAP values for {sample_size} test rows ...")
    shap_values = explainer(sample)

    values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, 1]

    plt.figure()
    shap.summary_plot(values, sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PLOT_PATH, dpi=150)
    plt.close()
    print(f"Saved global SHAP summary plot to {SHAP_SUMMARY_PLOT_PATH}")

    print("\nExample: top 3 factors for the first test row:")
    top = get_top_contributing_features(explainer, X_test_t[0:1], feature_names, top_n=3)
    for item in top:
        print(f"  {item['feature']}: {item['impact']} ({item['direction']})")


if __name__ == "__main__":
    main()
