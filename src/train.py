"""
Train and compare Logistic Regression, Random Forest, and XGBoost models
for churn prediction.

Design choices worth explaining in an interview:
  - Stratified train/test split so the ~26% churn rate is preserved in
    both splits.
  - SMOTE is fit on the *training* split only (after preprocessing),
    never on the test split -- the test set always reflects the real,
    imbalanced class distribution so metrics are honest.
  - We report precision, recall, F1, and ROC-AUC for the churn (positive)
    class, not just accuracy, because a model that always predicts "No
    churn" would score ~73% accuracy while being useless.
  - Model selection is driven by ROC-AUC (ranking quality across
    thresholds) with recall on the churn class as a tie-breaker, since a
    missed churner (false negative) typically costs the business far more
    than an unnecessary retention offer (false positive).
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from src.config import (
    COMPARISON_CSV_PATH,
    DATA_PATH,
    METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    SHAP_BACKGROUND_PATH,
)
from src.preprocessing import (
    apply_smote,
    build_preprocessor,
    get_feature_names,
    load_data,
    save_preprocessor,
    split_X_y,
)

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    return {
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
    }


def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = load_data(DATA_PATH)
    X, y = split_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Train churn rate: {y_train.mean():.3f} | Test churn rate: {y_test.mean():.3f}")

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)
    save_preprocessor(preprocessor)
    print(f"Preprocessor fit on {X_train_t.shape[1]} encoded features and saved.")

    # Small background sample for SHAP explainers at inference time, so the
    # API doesn't need the original training CSV to build an explainer.
    rng = np.random.default_rng(RANDOM_STATE)
    background_idx = rng.choice(
        X_train_t.shape[0], size=min(100, X_train_t.shape[0]), replace=False
    )
    joblib.dump(X_train_t[background_idx], SHAP_BACKGROUND_PATH)

    print("Applying SMOTE to the training split only ...")
    X_train_res, y_train_res = apply_smote(X_train_t, y_train.to_numpy())
    print(
        f"Resampled train churn rate: {y_train_res.mean():.3f} "
        f"({len(y_train_res)} rows, was {len(y_train)})"
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    fitted_models = {}

    for name, model in MODELS.items():
        print(f"\n=== {name} ===")

        cv_scores = cross_val_score(
            model, X_train_res, y_train_res, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        print(f"5-fold CV ROC-AUC (resampled train): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        model.fit(X_train_res, y_train_res)
        metrics = evaluate(model, X_test_t, y_test)
        metrics["cv_roc_auc_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_roc_auc_std"] = round(cv_scores.std(), 4)

        print(
            f"Test  -> precision: {metrics['precision']}, recall: {metrics['recall']}, "
            f"f1: {metrics['f1']}, roc_auc: {metrics['roc_auc']}"
        )
        print(f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")

        results[name] = metrics
        fitted_models[name] = model

    comparison_df = pd.DataFrame(results).T
    comparison_df.index.name = "model"
    comparison_df = comparison_df[
        ["precision", "recall", "f1", "roc_auc", "cv_roc_auc_mean", "cv_roc_auc_std"]
    ]
    print("\n=== Model comparison ===")
    print(comparison_df.to_string())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(COMPARISON_CSV_PATH)

    best_name = comparison_df["roc_auc"].astype(float).idxmax()
    best_model = fitted_models[best_name]
    print(f"\nSelected best model by test ROC-AUC: {best_name}")

    joblib.dump(best_model, MODEL_PATH)

    metadata = {
        "best_model_name": best_name,
        "feature_names": feature_names,
        "metrics": results,
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "positive_class": "Churn = Yes",
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved best model to {MODEL_PATH}")
    print(f"Saved metadata/comparison to {METADATA_PATH} and {COMPARISON_CSV_PATH}")


if __name__ == "__main__":
    main()
