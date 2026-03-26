-- transform/models/staging/stg_merchant_transactions.sql
-- Staging layer: type casting, null handling, deduplication
-- Adds derived fields: transaction_value_tier, is_late_night, etc.

{{ config(materialized='table', schema='staging') }}

WITH source AS (
    SELECT * FROM {{ ref('raw_merchant_transactions') }}
),

cleaned AS (
    SELECT
        transaction_id,
        merchant_id,
        TRIM(merchant_name)                         AS merchant_name,
        ROUND(CAST(transaction_amount AS FLOAT), 2) AS transaction_amount,
        LPAD(CAST(mcc_code AS TEXT), 4, '0')        AS mcc_code,
        mcc_description,
        TRIM(city)                                  AS city,
        UPPER(TRIM(state))                          AS state,
        COALESCE(NULLIF(TRIM(neighborhood), ''), city) AS neighborhood,
        CAST(latitude AS FLOAT)                     AS latitude,
        CAST(longitude AS FLOAT)                    AS longitude,
        CAST(timestamp AS TIMESTAMP)                AS transaction_at,
        LOWER(acceptance_method)                    AS acceptance_method,
        CAST(stars AS FLOAT)                        AS stars,
        CAST(review_count AS INTEGER)               AS review_count,
        CAST(review_velocity_30d AS INTEGER)        AS review_velocity_30d,
        categories,
        CAST(is_open AS BOOLEAN)                    AS is_open,
        CAST(COALESCE(price_range, 2) AS INTEGER)   AS price_range,
        CAST(ingested_at AS TIMESTAMP)              AS ingested_at
    FROM source
    WHERE
        transaction_id IS NOT NULL
        AND merchant_id IS NOT NULL
        AND transaction_amount > 0
        AND transaction_amount < 100000   -- sanity cap
        AND city IS NOT NULL
),

enriched AS (
    SELECT
        *,

        -- Transaction value tier (for metadata filtering)
        CASE
            WHEN transaction_amount < 20  THEN 'low'
            WHEN transaction_amount < 75  THEN 'mid'
            ELSE 'high'
        END AS transaction_value_tier,

        -- Price symbol
        REPEAT('$', price_range) AS price_symbol,

        -- Quarter for analytics
        CASE CAST(strftime('%m', transaction_at) AS INTEGER)
            WHEN 1 THEN 'Q1' WHEN 2 THEN 'Q1' WHEN 3 THEN 'Q1'
            WHEN 4 THEN 'Q2' WHEN 5 THEN 'Q2' WHEN 6 THEN 'Q2'
            WHEN 7 THEN 'Q3' WHEN 8 THEN 'Q3' WHEN 9 THEN 'Q3'
            ELSE 'Q4'
        END AS transaction_quarter,

        strftime('%Y', transaction_at) AS transaction_year

    FROM cleaned
),

-- Deduplicate on transaction_id (keep latest ingested)
deduped AS (
    SELECT *
    FROM enriched
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY transaction_id ORDER BY ingested_at DESC
    ) = 1
)

SELECT * FROM deduped
