-- transform/models/mart/mart_merchant_intelligence.sql
-- Mart layer: business-ready aggregations for RAG retrieval
-- One row per merchant — aggregated transaction + rating intelligence

{{ config(materialized='table', schema='mart') }}

WITH stg AS (
    SELECT * FROM {{ ref('stg_merchant_transactions') }}
),

merchant_agg AS (
    SELECT
        merchant_id,
        merchant_name,
        city,
        state,
        neighborhood,
        latitude,
        longitude,
        mcc_code,
        mcc_description,
        categories,
        price_range,
        price_symbol,
        is_open,

        -- Transaction aggregates
        COUNT(*)                                        AS total_transactions,
        ROUND(AVG(transaction_amount), 2)               AS avg_transaction_amount,
        ROUND(SUM(transaction_amount), 2)               AS total_transaction_volume,
        MIN(transaction_amount)                         AS min_transaction,
        MAX(transaction_amount)                         AS max_transaction,
        ROUND(STDDEV(transaction_amount), 2)            AS stddev_transaction,

        -- Acceptance method distribution
        COUNT(CASE WHEN acceptance_method = 'chip'        THEN 1 END) AS chip_count,
        COUNT(CASE WHEN acceptance_method = 'contactless' THEN 1 END) AS contactless_count,
        COUNT(CASE WHEN acceptance_method = 'online'      THEN 1 END) AS online_count,
        COUNT(CASE WHEN acceptance_method = 'swipe'       THEN 1 END) AS swipe_count,
        COUNT(CASE WHEN acceptance_method = 'mobile_pay'  THEN 1 END) AS mobile_pay_count,

        -- Value tier distribution
        COUNT(CASE WHEN transaction_value_tier = 'low'  THEN 1 END) AS low_value_txns,
        COUNT(CASE WHEN transaction_value_tier = 'mid'  THEN 1 END) AS mid_value_txns,
        COUNT(CASE WHEN transaction_value_tier = 'high' THEN 1 END) AS high_value_txns,

        -- Quarterly breakdown
        COUNT(CASE WHEN transaction_quarter = 'Q1' THEN 1 END) AS q1_transactions,
        COUNT(CASE WHEN transaction_quarter = 'Q2' THEN 1 END) AS q2_transactions,
        COUNT(CASE WHEN transaction_quarter = 'Q3' THEN 1 END) AS q3_transactions,
        COUNT(CASE WHEN transaction_quarter = 'Q4' THEN 1 END) AS q4_transactions,

        -- Review / rating fields
        AVG(stars)                   AS avg_stars,
        MAX(review_count)            AS review_count,
        MAX(review_velocity_30d)     AS review_velocity_30d,

        -- Temporal
        MIN(transaction_at)          AS first_seen_at,
        MAX(transaction_at)          AS last_seen_at,
        MAX(ingested_at)             AS last_ingested_at

    FROM stg
    GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14
),

-- Anomaly detection: flag merchants with unusually high volume
anomaly_flags AS (
    SELECT
        merchant_id,
        AVG(avg_transaction_amount) OVER ()             AS global_avg_amount,
        STDDEV(avg_transaction_amount) OVER ()          AS global_stddev_amount,
        CASE
            WHEN avg_transaction_amount > (
                AVG(avg_transaction_amount) OVER () +
                2 * STDDEV(avg_transaction_amount) OVER ()
            ) THEN TRUE ELSE FALSE
        END AS is_high_volume_anomaly,
        CASE
            WHEN q4_transactions > (
                (q1_transactions + q2_transactions + q3_transactions) / 3.0 * 1.5
            ) THEN TRUE ELSE FALSE
        END AS is_q4_spending_spike
    FROM merchant_agg
)

SELECT
    m.*,
    ROUND((m.avg_transaction_amount - a.global_avg_amount) / NULLIF(a.global_stddev_amount, 0), 2) AS z_score_amount,
    a.is_high_volume_anomaly,
    a.is_q4_spending_spike
FROM merchant_agg m
JOIN anomaly_flags a USING (merchant_id)
