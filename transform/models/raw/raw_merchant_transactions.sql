-- transform/models/raw/raw_merchant_transactions.sql
-- Raw layer: direct source reference with minimal transformation
-- Materialized as VIEW — always reflects latest warehouse data

{{ config(materialized='view', schema='raw') }}

SELECT
    transaction_id,
    merchant_id,
    merchant_name,
    transaction_amount,
    mcc_code,
    mcc_description,
    city,
    state,
    neighborhood,
    latitude,
    longitude,
    timestamp,
    acceptance_method,
    stars,
    review_count,
    review_velocity_30d,
    categories,
    is_open,
    price_range,
    ingested_at
FROM {{ source('raw', 'raw_merchant_transactions') }}
