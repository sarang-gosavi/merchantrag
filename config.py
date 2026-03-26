"""
config.py — Environment-driven settings for MerchantRAG.

Every service connection is toggled via env vars:
  USE_LOCAL_*=true  →  local fallback (SQLite, TF-IDF, mock Kafka)
  USE_LOCAL_*=false →  production service (Snowflake, Pinecone, real Kafka)

Copy .env.example → .env and fill in API keys before running.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────────────────────
    APP_NAME:    str = "MerchantRAG"
    APP_VERSION: str = "1.0.0"
    DEBUG:       bool = False
    LOG_LEVEL:   str = "INFO"
    API_HOST:    str = "0.0.0.0"
    API_PORT:    int = 8000
    WORKERS:     int = 4
    SECRET_KEY:  str = "change-me-in-production"
    RATE_LIMIT_PER_MIN: int = 60

    # ── LLM ──────────────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY:    Optional[str] = None
    LLM_PROVIDER:      str = "anthropic"   # anthropic | openai
    LLM_MODEL:         str = "claude-sonnet-4-20250514"
    LLM_MAX_TOKENS:    int = 1024
    LLM_TEMPERATURE:   float = 0.0

    # ── Embeddings ────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "text-embedding-3-small"   # or sentence-transformers
    EMBEDDING_DIM:   int = 1536
    EMBEDDING_PROVIDER: str = "openai"    # openai | sentence_transformers
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE:      int = 512            # 256 | 512 | 1024

    # ── Kafka ─────────────────────────────────────────────────────────────
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_TRANSACTIONS: str = "merchant.transactions"
    KAFKA_TOPIC_ALERTS:       str = "merchant.alerts"
    KAFKA_GROUP_ID:           str = "merchantrag-consumer"
    KAFKA_MOCK_MODE:          bool = True   # flip for real broker
    KAFKA_BATCH_SIZE:         int = 200
    KAFKA_RETENTION_MS:       int = 604800000   # 7 days

    # ── Snowflake + Snowpark ──────────────────────────────────────────────
    SNOWFLAKE_ACCOUNT:    Optional[str] = None
    SNOWFLAKE_USER:       Optional[str] = None
    SNOWFLAKE_PASSWORD:   Optional[str] = None
    SNOWFLAKE_DATABASE:   str = "MERCHANTRAG_DB"
    SNOWFLAKE_SCHEMA:     str = "PUBLIC"
    SNOWFLAKE_WAREHOUSE:  str = "COMPUTE_WH"
    SNOWFLAKE_ROLE:       str = "SYSADMIN"
    SNOWFLAKE_STAGE:      str = "MERCHANT_STAGE"       # for Snowpipe
    SNOWFLAKE_PIPE:       str = "MERCHANT_TRANSACTION_PIPE"
    USE_LOCAL_DB:         bool = True      # flip for Snowflake
    LOCAL_DB_PATH:        str = "./data/merchantrag.sqlite"

    # ── Vector Store ──────────────────────────────────────────────────────
    PINECONE_API_KEY:     Optional[str] = None
    PINECONE_ENV:         str = "gcp-starter"
    PINECONE_INDEX:       str = "merchantrag-index"
    PGVECTOR_URL:         Optional[str] = None   # postgres://user:pass@host/db
    USE_LOCAL_VECTORS:    bool = True      # flip for Pinecone or pgvector
    VECTOR_BACKEND:       str = "local"    # local | pinecone | pgvector

    # ── dbt ───────────────────────────────────────────────────────────────
    DBT_PROJECT_DIR:      str = "./transform"
    DBT_PROFILES_DIR:     str = "./transform"
    DBT_TARGET:           str = "dev"

    # ── AWS ───────────────────────────────────────────────────────────────
    AWS_ACCESS_KEY_ID:    Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION:           str = "us-east-1"
    AWS_S3_BUCKET:        str = "merchantrag-data"
    AWS_LAMBDA_FUNCTION:  str = "merchantrag-query"
    USE_LOCAL_STORAGE:    bool = True      # flip for S3

    # ── LangSmith ─────────────────────────────────────────────────────────
    LANGCHAIN_API_KEY:    Optional[str] = None   # LangSmith key
    LANGCHAIN_PROJECT:    str = "merchantrag"
    LANGCHAIN_TRACING_V2: bool = False      # flip to enable LangSmith tracing
    LANGCHAIN_ENDPOINT:   str = "https://api.smith.langchain.com"

    # ── Data ──────────────────────────────────────────────────────────────
    YELP_DATASET_DIR:     str = "./data/yelp_dataset"
    YELP_BUSINESS_FILE:   str = "yelp_academic_dataset_business.json"
    USE_SAMPLE_DATA:      bool = True
    SYNTHETIC_RECORDS:    int = 10000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


ENV_EXAMPLE = """
# ── LLM ──────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
# LLM_PROVIDER=openai

# ── LangSmith (tracing + eval) ────────────────────────────────────────────────
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true

# ── Pinecone ──────────────────────────────────────────────────────────────────
# PINECONE_API_KEY=...
# VECTOR_BACKEND=pinecone
# USE_LOCAL_VECTORS=false

# ── pgvector (local Postgres) ──────────────────────────────────────────────────
# PGVECTOR_URL=postgresql://postgres:password@localhost:5432/merchantrag
# VECTOR_BACKEND=pgvector
# USE_LOCAL_VECTORS=false

# ── Snowflake ─────────────────────────────────────────────────────────────────
# SNOWFLAKE_ACCOUNT=myaccount.us-east-1
# SNOWFLAKE_USER=myuser
# SNOWFLAKE_PASSWORD=...
# USE_LOCAL_DB=false

# ── Kafka ─────────────────────────────────────────────────────────────────────
# KAFKA_BOOTSTRAP_SERVERS=localhost:9092
# KAFKA_MOCK_MODE=false

# ── AWS ───────────────────────────────────────────────────────────────────────
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
# USE_LOCAL_STORAGE=false

# ── Yelp dataset ──────────────────────────────────────────────────────────────
# USE_SAMPLE_DATA=false
# YELP_DATASET_DIR=./data/yelp_dataset

DEBUG=false
LOG_LEVEL=INFO
""".strip()
