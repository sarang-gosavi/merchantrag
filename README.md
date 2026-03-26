# 🏪 MerchantRAG

> **Production-grade RAG pipeline over 10M+ merchant transaction records.**
> Kafka + Snowflake + Pinecone + LangChain + LangGraph + LangSmith.
>
> Built on the Yelp Open Dataset — demonstrating enterprise RAG architecture patterns
> applicable to any merchant intelligence, payments, or location-based analytics platform.

---

## 🎯 What it does

```
"Which merchant categories had anomalous spending in Q4?"
"Find coffee shops open past 10pm in the West Loop with contactless payment"
"Now filter that by merchants with 4.5+ stars only"  ← multi-turn memory
"Show me merchants with unusually high review velocity this month"
"Which acceptance methods dominate in the Streeterville neighborhood?"
```

The platform answers these in natural language by:
1. **Streaming** Yelp business records through a Kafka producer (real or mock)
2. **Transforming** raw data through dbt models (raw → staging → mart)
3. **Validating** data quality with Great Expectations before indexing
4. **Embedding** merchant summaries via OpenAI or sentence-transformers
5. **Storing** vectors in Pinecone (cloud) or pgvector (local Postgres)
6. **Retrieving** top-k relevant merchants via hybrid semantic + keyword search
7. **Generating** grounded answers via LangChain RetrievalQA + Claude/GPT-4o
8. **Monitoring** query quality with LangSmith + RAGAS (precision@k, faithfulness)
9. **Running agents** autonomously via LangGraph (validation, freshness SLA, anomalies)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MerchantRAG Architecture                            │
│                                                                             │
│  Yelp Open Dataset (150K+ businesses)                                       │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────┐  produce  ┌─────────────────┐  consume               │
│  │  YelpLoader /   │ ────────► │  Kafka Topic    │ ────────►               │
│  │  BatchLoader    │           │ merchant.txns   │                         │
│  │  (Ab Initio     │           └─────────────────┘                        │
│  │   bridge)       │                    │                                  │
│  └─────────────────┘         ┌──────────┴───────────┐                     │
│                               ▼                     ▼                      │
│                     ┌──────────────────┐  ┌──────────────────┐            │
│                     │  SnowflakeWriter │  │  BatchLoader     │            │
│                     │  (Snowpipe /     │  │  (CSV staging →  │            │
│                     │   SQLite local)  │  │   archive)       │            │
│                     └────────┬─────────┘  └──────────────────┘            │
│                              │                                             │
│                              ▼                                             │
│                     ┌──────────────────────────────────────┐              │
│                     │           dbt Models                  │              │
│                     │  raw → staging → mart_merchant_intel  │              │
│                     │  + Great Expectations quality suites  │              │
│                     └────────┬─────────────────────────────┘              │
│                              │                                             │
│                              ▼                                             │
│                     ┌──────────────────────────────────────┐              │
│                     │     Embedding Pipeline                │              │
│                     │  Chunking (256/512/1024 tokens)       │              │
│                     │  OpenAI text-embedding-3-small        │              │
│                     │  OR sentence-transformers (local)     │              │
│                     └────────┬─────────────────────────────┘              │
│                              │                                             │
│                   ┌──────────┴──────────┐                                  │
│                   ▼                     ▼                                  │
│         ┌──────────────────┐  ┌──────────────────┐                        │
│         │  Pinecone Cloud  │  │  pgvector        │                        │
│         │  (serverless)    │  │  (Postgres local)│                        │
│         └────────┬─────────┘  └────────┬─────────┘                       │
│                  └──────────┬───────────┘                                  │
│                             ▼                                              │
│                  ┌───────────────────────────────────────┐                │
│                  │          RAG Engine                    │                │
│                  │  LangChain RetrievalQA                 │                │
│                  │  Multi-turn ConversationMemory         │                │
│                  │  LangSmith tracing                     │                │
│                  │  RAGAS evaluation (precision@k)        │                │
│                  └────────────────┬──────────────────────┘                │
│                                   │                                        │
│                  ┌────────────────┴──────────────────────┐                │
│                  │        LangGraph Agents               │                │
│                  │  DataValidationAgent                  │                │
│                  │  DataFreshnessAgent                   │                │
│                  │  AnomalyDetectionAgent                │                │
│                  └────────────────┬──────────────────────┘                │
│                                   │                                        │
│                                   ▼                                        │
│                         FastAPI REST API                                   │
│                  POST /query   POST /ingest   GET /merchants               │
│                  POST /agents/run   POST /analytics   GET /health          │
│                                   │                                        │
│                         AWS Lambda (serverless)                            │
│                   API Gateway + EventBridge + S3 trigger                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Default (demo) | Production |
|---|---|---|
| **Ingestion** | Mock Kafka (asyncio.Queue) | Apache Kafka / Confluent Cloud |
| **Batch loading** | CSV staging dir | Ab Initio-style file landing + Snowpipe |
| **Warehouse** | SQLite | Snowflake + Snowpark |
| **Transform** | dbt SQLite adapter | dbt Snowflake adapter |
| **Data quality** | GE heuristic suite | Great Expectations with HTML reports |
| **Vectors** | TF-IDF in-memory | Pinecone serverless / pgvector |
| **Embeddings** | sentence-transformers | OpenAI text-embedding-3-small |
| **LLM** | Claude Sonnet | Claude Opus / GPT-4o |
| **Agents** | Sequential fallback | LangGraph compiled graph |
| **Observability** | Local JSONL log | LangSmith + RAGAS |
| **Deploy** | Docker Compose | AWS Lambda + Terraform |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/sarang-gosavi/merchantrag.git
cd merchantrag
python -m venv .venv && source .venv/bin/activate

# Minimal install (demo mode)
pip install fastapi uvicorn pydantic pydantic-settings anthropic \
    langchain langchain-core langchain-anthropic langsmith langgraph \
    sentence-transformers aiosqlite aiokafka rich pytest pytest-asyncio
```

### 2. Configure

```bash
cp .env.example .env
# Set ANTHROPIC_API_KEY (minimum required for LLM answers)
# All other services have local fallbacks — no cloud accounts needed to start
```

### 3. Run

```bash
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for Swagger UI.

### 4. Run the test suite

```bash
pytest tests/ -v
```

### 5. Docker

```bash
docker-compose up api                        # minimal
docker-compose --profile kafka up            # with real Kafka
docker-compose --profile pgvector up         # with pgvector
```

---

## 🔌 API Reference

### `POST /query` — RAG query with multi-turn memory

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Which merchant categories had anomalous spending in Q4?",
    "top_k": 5,
    "session_id": "sess_abc123"
  }'
```

**Follow-up (multi-turn):**
```bash
curl -X POST http://localhost:8000/query \
  -d '{"query": "Now filter that by West Coast merchants only", "session_id": "sess_abc123"}'
```

### `POST /ingest` — Real-time ingest + re-index

```bash
curl -X POST http://localhost:8000/ingest \
  -d '{"records": [{"business_id": "new_001", "name": "New Cafe", ...}], "reindex": true}'
```

### `POST /agents/run` — LangGraph pipeline

```bash
curl -X POST "http://localhost:8000/agents/run?task=anomaly_scan"
```

### `POST /analytics` — Aggregate metrics

```bash
curl -X POST "http://localhost:8000/analytics?metric=review_velocity&group_by=mcc_code"
curl -X POST "http://localhost:8000/analytics?metric=anomaly_flags"
curl -X POST "http://localhost:8000/analytics?metric=q4_spending"
curl -X POST "http://localhost:8000/analytics?metric=acceptance_methods"
```

---

## 📈 Scaling to 10M+ Records

| Component | Config change |
|---|---|
| Vectors | `VECTOR_BACKEND=pinecone` + `PINECONE_API_KEY` |
| Kafka | `KAFKA_MOCK_MODE=false` + `KAFKA_BOOTSTRAP_SERVERS` |
| Warehouse | `USE_LOCAL_DB=false` + Snowflake creds |
| Embeddings | `OPENAI_API_KEY` + `EMBEDDING_PROVIDER=openai` |
| Deploy | `terraform apply` in `infra/terraform/` |
| LangSmith | `LANGCHAIN_API_KEY` + `LANGCHAIN_TRACING_V2=true` |

Architecture is identical — only env vars change.

---

## 📂 Repository Structure

```
merchantrag/
├── ingestion/                  ← Kafka producers + Snowpipe batch loaders
│   ├── kafka_producer.py       MCC mapping, Yelp→merchant schema bridge
│   └── kafka_consumer.py       Consumer + SnowflakeWriter + BatchLoader
├── transform/                  ← dbt models (raw → staging → mart)
│   ├── dbt_project.yml
│   ├── profiles.yml            Snowflake + SQLite targets
│   ├── models/
│   │   ├── raw/                Source views
│   │   ├── staging/            Cleaned, typed, deduplicated
│   │   └── mart/               Aggregated merchant intelligence + anomaly flags
│   └── expectations/
│       └── ge_suites.py        Great Expectations: null, MCC, amount, integrity
├── embeddings/
│   └── embedder.py             Chunking + OpenAI/ST + Pinecone/pgvector/local
├── rag_api/
│   └── query.py                LangChain RetrievalQA + multi-turn memory + LangSmith
├── agents/
│   └── graph.py                LangGraph: Validation + Freshness + Anomaly agents
├── observability/
│   └── langsmith_tracer.py     LangSmith tracing + RAGAS eval metrics
├── infra/
│   ├── lambda_handler.py       AWS Lambda entry point (API Gateway + S3 trigger)
│   └── terraform/
│       └── main.tf             AWS Lambda + API Gateway + S3 + EventBridge + Snowflake
├── data/
│   └── sample_merchants.py     40-record Chicago dataset (Yelp schema-compatible)
├── tests/
│   └── test_merchantrag.py     35 tests: models, pipeline, store, agents, API
├── models.py                   Pydantic schemas (MerchantTransaction, AgentState, ...)
├── config.py                   pydantic-settings — all env vars documented
├── main.py                     FastAPI app — all endpoints
├── requirements.txt            Full dependency list with install notes
├── Dockerfile                  Multi-stage Python 3.11 container
├── docker-compose.yml          API + Kafka + pgvector profiles
├── AGENTS.md                   LangGraph decision trees (most shareable content)
├── DESIGN.md                   Prompt engineering + data contract rationale
└── ARCHITECTURE.md             Full system design for interview prep
```

---

## 📊 Datasets

| Dataset | Records | Source |
|---|---|---|
| **Yelp Open Dataset** | 150K+ businesses, 6M+ reviews | yelp.com/dataset |
| Kaggle: CC Fraud Detection | 284K transactions | kaggle.com |
| Kaggle: Online Retail II | 500K+ merchant transactions | kaggle.com |
| Python Faker | Unlimited synthetic | `pip install faker` |

See `get_yelp_data.py` for automated download instructions.

---

*Built by Sarang Gosavi · 2026 AI Data Platform Portfolio · Chicago, IL*
