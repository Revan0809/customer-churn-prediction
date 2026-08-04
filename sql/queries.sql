-- ============================================================================
-- Business-facing SQL queries against sql/churn.db (table: customers)
--
-- Each query is preceded by a machine-readable header so sql/run_queries.py
-- can split this file into named, independently-runnable queries and save
-- each as its own CSV in sql/query_results/. The header format is:
--
--   -- @name: <a_snake_case_identifier>
--   -- @description: <one-line business question this answers>
--
-- Everything between one @name header and the next is that query's SQL.
-- ============================================================================


-- @name: overall_churn_rate
-- @description: What share of the customer base has churned, overall?
SELECT
    COUNT(*)                                            AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)       AS churned_customers,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                    AS churn_rate_pct
FROM customers;


-- @name: churn_rate_by_contract
-- @description: Does contract type (month-to-month vs. annual commitment) predict churn?
SELECT
    contract,
    COUNT(*)                                            AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)       AS churned_customers,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                    AS churn_rate_pct
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC;


-- @name: churn_rate_by_internet_service
-- @description: Which internet service type (DSL / fiber / none) has the highest churn?
SELECT
    internet_service,
    COUNT(*)                                            AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)       AS churned_customers,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                    AS churn_rate_pct
FROM customers
GROUP BY internet_service
ORDER BY churn_rate_pct DESC;


-- @name: churn_rate_by_payment_method
-- @description: Does how a customer pays (electronic check vs. autopay) correlate with churn?
SELECT
    payment_method,
    COUNT(*)                                            AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)       AS churned_customers,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                    AS churn_rate_pct
FROM customers
GROUP BY payment_method
ORDER BY churn_rate_pct DESC;


-- @name: churn_rate_by_tenure_bucket
-- @description: Is churn concentrated in new customers, or does it persist into long tenures?
--   Buckets: 0-12, 13-24, 25-48, 49+ months, ordered chronologically (not alphabetically).
SELECT
    CASE
        WHEN tenure <= 12 THEN '0-12'
        WHEN tenure <= 24 THEN '13-24'
        WHEN tenure <= 48 THEN '25-48'
        ELSE '49+'
    END                                                  AS tenure_bucket,
    CASE
        WHEN tenure <= 12 THEN 1
        WHEN tenure <= 24 THEN 2
        WHEN tenure <= 48 THEN 3
        ELSE 4
    END                                                  AS bucket_order,
    COUNT(*)                                            AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)       AS churned_customers,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                    AS churn_rate_pct
FROM customers
GROUP BY tenure_bucket, bucket_order
ORDER BY bucket_order;


-- @name: tenure_cohort_churn_trend_by_contract
-- @description: Cohort-style view using window functions: for each contract type,
--   how does churn rate move from one tenure cohort to the next, and by how much
--   (LAG + PARTITION BY contract)? Surfaces which contract type sees the steepest
--   early-tenure drop-off vs. a flatter, more stable curve.
WITH tenure_cohorts AS (
    SELECT
        contract,
        CASE
            WHEN tenure <= 12 THEN '0-12'
            WHEN tenure <= 24 THEN '13-24'
            WHEN tenure <= 48 THEN '25-48'
            ELSE '49+'
        END                                              AS tenure_bucket,
        CASE
            WHEN tenure <= 12 THEN 1
            WHEN tenure <= 24 THEN 2
            WHEN tenure <= 48 THEN 3
            ELSE 4
        END                                              AS bucket_order,
        churn
    FROM customers
),
cohort_churn AS (
    SELECT
        contract,
        tenure_bucket,
        bucket_order,
        COUNT(*)                                        AS total_customers,
        ROUND(
            100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
        )                                                AS churn_rate_pct
    FROM tenure_cohorts
    GROUP BY contract, tenure_bucket, bucket_order
)
SELECT
    contract,
    tenure_bucket,
    total_customers,
    churn_rate_pct,
    LAG(churn_rate_pct) OVER (
        PARTITION BY contract ORDER BY bucket_order
    )                                                    AS prev_cohort_churn_rate_pct,
    ROUND(
        churn_rate_pct - LAG(churn_rate_pct) OVER (
            PARTITION BY contract ORDER BY bucket_order
        ), 2
    )                                                    AS churn_rate_change_pct
FROM cohort_churn
ORDER BY contract, bucket_order;


-- @name: tenure_bucket_over_bucket_trend
-- @description: CTE-based, contract-agnostic version of the same question: across
--   the whole customer base, does churn risk fall as tenure increases, and where is
--   the biggest drop? Useful as the single trend line for an exec-facing chart.
WITH bucketed AS (
    SELECT
        CASE
            WHEN tenure <= 12 THEN '0-12'
            WHEN tenure <= 24 THEN '13-24'
            WHEN tenure <= 48 THEN '25-48'
            ELSE '49+'
        END                                              AS tenure_bucket,
        CASE
            WHEN tenure <= 12 THEN 1
            WHEN tenure <= 24 THEN 2
            WHEN tenure <= 48 THEN 3
            ELSE 4
        END                                              AS bucket_order,
        churn
    FROM customers
),
bucket_churn_rate AS (
    SELECT
        tenure_bucket,
        bucket_order,
        COUNT(*)                                        AS total_customers,
        ROUND(
            100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
        )                                                AS churn_rate_pct
    FROM bucketed
    GROUP BY tenure_bucket, bucket_order
)
SELECT
    tenure_bucket,
    total_customers,
    churn_rate_pct,
    ROUND(
        churn_rate_pct - LAG(churn_rate_pct) OVER (ORDER BY bucket_order), 2
    )                                                    AS change_vs_prev_bucket_pct
FROM bucket_churn_rate
ORDER BY bucket_order;


-- @name: top_3_highest_risk_segments
-- @description: Which specific Contract + InternetService + PaymentMethod combos
--   should retention target first? Ranked by churn rate, but only among segments
--   with at least 30 customers so a tiny, noisy segment (e.g. 4 customers, 3 churned)
--   doesn't outrank a real, actionable population.
WITH segments AS (
    SELECT
        contract,
        internet_service,
        payment_method,
        COUNT(*)                                        AS customer_count,
        SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)   AS churned_customers,
        ROUND(
            100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
        )                                                AS churn_rate_pct
    FROM customers
    GROUP BY contract, internet_service, payment_method
    HAVING COUNT(*) >= 30
)
SELECT
    contract,
    internet_service,
    payment_method,
    customer_count,
    churned_customers,
    churn_rate_pct
FROM segments
ORDER BY churn_rate_pct DESC, customer_count DESC
LIMIT 3;
