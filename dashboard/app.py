"""
Streamlit BI dashboard for the Telco churn dataset.

Reads directly from sql/churn.db (built by sql/load_to_sqlite.py) using SQL
queries rather than pandas filtering, and optionally reuses the trained
model + SHAP explainer from src/ (if models/*.joblib exist) to show the
top global churn drivers alongside the SQL-driven views.

Run with:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "sql" / "churn.db"

# So `from src...` works when Streamlit runs this file directly.
sys.path.insert(0, str(ROOT_DIR))

TENURE_BUCKET_SQL = """
    CASE
        WHEN tenure <= 12 THEN '0-12'
        WHEN tenure <= 24 THEN '13-24'
        WHEN tenure <= 48 THEN '25-48'
        ELSE '49+'
    END
"""
TENURE_BUCKET_ORDER_SQL = """
    CASE
        WHEN tenure <= 12 THEN 1
        WHEN tenure <= 24 THEN 2
        WHEN tenure <= 48 THEN 3
        ELSE 4
    END
"""
TENURE_BUCKET_LABELS = ["0-12", "13-24", "25-48", "49+"]

# Single accent color used consistently for every "magnitude" mark (bars,
# lines) in this dashboard -- one hue, not a rainbow, since each chart here
# is a single series.
ACCENT = "#3B82F6"


@st.cache_resource
def get_connection() -> sqlite3.Connection:
    if not DB_PATH.exists():
        st.error(
            f"{DB_PATH} not found. Run `python sql/load_to_sqlite.py` first."
        )
        st.stop()
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def run_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, get_connection(), params=params)


@st.cache_data
def get_filter_options() -> dict[str, list[str]]:
    options = {}
    for col in ["contract", "internet_service", "payment_method"]:
        df = run_query(f"SELECT DISTINCT {col} FROM customers ORDER BY {col}")
        options[col] = df[col].tolist()
    return options


def overall_churn_rate() -> tuple[int, int, float]:
    df = run_query(
        """
        SELECT
            COUNT(*) AS total_customers,
            SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
            ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
        FROM customers
        """
    )
    row = df.iloc[0]
    return int(row.total_customers), int(row.churned_customers), float(row.churn_rate_pct)


def churn_rate_by_segment(dimension: str) -> pd.DataFrame:
    return run_query(
        f"""
        SELECT
            {dimension} AS segment,
            COUNT(*) AS total_customers,
            ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
        FROM customers
        GROUP BY {dimension}
        ORDER BY churn_rate_pct DESC
        """
    )


def churn_rate_by_tenure_bucket() -> pd.DataFrame:
    return run_query(
        f"""
        SELECT
            {TENURE_BUCKET_SQL} AS tenure_bucket,
            {TENURE_BUCKET_ORDER_SQL} AS bucket_order,
            COUNT(*) AS total_customers,
            ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
        FROM customers
        GROUP BY tenure_bucket, bucket_order
        ORDER BY bucket_order
        """
    )


def filtered_customers(contract: str, internet_service: str, payment_method: str) -> pd.DataFrame:
    clauses, params = [], []
    if contract != "All":
        clauses.append("contract = ?")
        params.append(contract)
    if internet_service != "All":
        clauses.append("internet_service = ?")
        params.append(internet_service)
    if payment_method != "All":
        clauses.append("payment_method = ?")
        params.append(payment_method)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return run_query(
        f"""
        SELECT
            customer_id, gender, senior_citizen, partner, dependents, tenure,
            contract, internet_service, payment_method, monthly_charges,
            total_charges, churn
        FROM customers
        {where}
        ORDER BY customer_id
        """,
        tuple(params),
    )


@st.cache_resource
def load_shap_assets():
    """Load the trained model/preprocessor/explainer from models/, if present.

    Returns None if the model hasn't been trained yet (models/*.joblib
    missing) -- this section is optional, the rest of the dashboard works
    without it.
    """
    try:
        import joblib
        import shap

        from src.config import MODEL_PATH, PREPROCESSOR_PATH, DATA_PATH, RANDOM_STATE
        from src.explain import build_explainer, prettify_feature_name
        from src.preprocessing import load_data, load_preprocessor, split_X_y, get_feature_names

        if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
            return None

        model = joblib.load(MODEL_PATH)
        preprocessor = load_preprocessor(PREPROCESSOR_PATH)
        feature_names = get_feature_names(preprocessor)

        df = load_data(DATA_PATH)
        X, _ = split_X_y(df)
        X_t = preprocessor.transform(X)

        sample_size = min(200, X_t.shape[0])
        sample = shap.sample(X_t, sample_size, random_state=RANDOM_STATE)
        explainer = build_explainer(model, sample)
        shap_values = explainer(sample)

        values = shap_values.values
        if values.ndim == 3:
            values = values[:, :, 1]

        mean_abs_impact = pd.DataFrame(
            {
                "feature": [prettify_feature_name(f) for f in feature_names],
                "mean_abs_shap": abs(values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)

        return mean_abs_impact
    except Exception as exc:  # missing model artifacts, shap not installed, etc.
        st.session_state["_shap_load_error"] = str(exc)
        return None


def bar_chart(df: pd.DataFrame, x: str, y: str, x_title: str, y_title: str, sort=None):
    chart = (
        alt.Chart(df)
        .mark_bar(color=ACCENT, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{x}:N", title=x_title, sort=sort),
            y=alt.Y(f"{y}:Q", title=y_title),
            tooltip=list(df.columns),
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")


def line_chart(df: pd.DataFrame, x: str, y: str, x_title: str, y_title: str, sort=None):
    chart = (
        alt.Chart(df)
        .mark_line(color=ACCENT, point=alt.OverlayMarkDef(color=ACCENT, size=60), strokeWidth=2)
        .encode(
            x=alt.X(f"{x}:N", title=x_title, sort=sort),
            y=alt.Y(f"{y}:Q", title=y_title),
            tooltip=list(df.columns),
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")


def main():
    st.set_page_config(page_title="Customer Churn Analytics", layout="wide")
    st.title("Customer Churn — SQL + BI Analytics")
    st.caption(
        "Reads directly from sql/churn.db. Complements the ML prediction app "
        "(frontend/ + backend/) with the retention-team view: where is churn "
        "concentrated, and which segments should be targeted first?"
    )

    total, churned, churn_rate_pct = overall_churn_rate()
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total customers", f"{total:,}")
    kpi2.metric("Churned customers", f"{churned:,}")
    kpi3.metric("Overall churn rate", f"{churn_rate_pct:.2f}%")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn rate by segment")
        dimension_label = st.selectbox(
            "Group by", ["Contract", "Internet service", "Payment method"]
        )
        dimension_col = {
            "Contract": "contract",
            "Internet service": "internet_service",
            "Payment method": "payment_method",
        }[dimension_label]
        segment_df = churn_rate_by_segment(dimension_col)
        bar_chart(
            segment_df,
            x="segment",
            y="churn_rate_pct",
            x_title=dimension_label,
            y_title="Churn rate (%)",
            sort="-y",
        )

    with col2:
        st.subheader("Churn rate by tenure cohort")
        tenure_df = churn_rate_by_tenure_bucket()
        line_chart(
            tenure_df,
            x="tenure_bucket",
            y="churn_rate_pct",
            x_title="Tenure (months)",
            y_title="Churn rate (%)",
            sort=TENURE_BUCKET_LABELS,
        )

    st.divider()

    st.subheader("Top churn drivers (SHAP, from the trained model)")
    shap_importance = load_shap_assets()
    if shap_importance is not None:
        top10 = shap_importance.head(10)
        bar_chart(
            top10.rename(columns={"feature": "segment", "mean_abs_shap": "churn_rate_pct"}),
            x="segment",
            y="churn_rate_pct",
            x_title="Feature",
            y_title="Mean |SHAP value|",
            sort="-y",
        )
    else:
        st.info(
            "Model artifacts not found in models/. Run `python -m src.train` and "
            "`python -m src.explain` to enable this section."
        )

    st.divider()

    st.subheader("Filter customers")
    options = get_filter_options()
    f1, f2, f3 = st.columns(3)
    contract_filter = f1.selectbox("Contract", ["All"] + options["contract"])
    internet_filter = f2.selectbox("Internet service", ["All"] + options["internet_service"])
    payment_filter = f3.selectbox("Payment method", ["All"] + options["payment_method"])

    filtered_df = filtered_customers(contract_filter, internet_filter, payment_filter)
    filtered_total = len(filtered_df)
    filtered_churned = int((filtered_df["churn"] == "Yes").sum())
    filtered_rate = round(100.0 * filtered_churned / filtered_total, 2) if filtered_total else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Customers in filter", f"{filtered_total:,}")
    m2.metric("Churned", f"{filtered_churned:,}")
    m3.metric("Churn rate", f"{filtered_rate:.2f}%")

    st.dataframe(filtered_df, width="stretch", height=400)


if __name__ == "__main__":
    main()
