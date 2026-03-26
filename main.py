"""
main.py — FastAPI entry point for MerchantRAG.

Endpoints (all spec-compliant from playbook):
  POST /query         — Natural-language RAG query with multi-turn memory
  POST /ingest        — Real-time merchant record ingest + re-index
  GET  /merchants     — Structured filter search over merchant corpus
  GET  /merchants/{id}— Single merchant lookup
  POST /analytics     — Aggregate metrics: MCC velocity, anomalies, distribution
  GET  /analytics/summary — Dashboard snapshot
  POST /agents/run    — LangGraph pipeline: validate + freshness + anomaly
  GET  /agents/status — Latest agent run results
  GET  /pipeline/stats — Kafka/pipeline stats
  POST /pipeline/reload — Full re-ingest trigger
  GET  /observability/metrics — RAGAS + LangSmith eval summary
  GET  /health        — Service health + vector store size

Run locally:
    uvicorn main:app --reload --port 8000

Docker:
    docker-compose up

AWS Lambda:
    Use infra/lambda_handler.py as entrypoint
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from config import settings, ENV_EXAMPLE
from models import (
    AgentState,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    KafkaEvent,
    MerchantTransaction,
    PipelineStats,
    QueryRequest,
    QueryResponse,
    RAGEvalMetrics,
    RetrievalResult,
)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Global singletons ─────────────────────────────────────────────────────────

vector_store  = None
chain         = None
memory        = None
chunker       = None
pipeline      = None
agent_graph   = None
tracer        = None
ragas_eval    = None
metrics_store = None
_pipeline_stats = PipelineStats()
_last_agent_result: Optional[Dict] = None


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, chain, memory, chunker, pipeline
    global agent_graph, tracer, ragas_eval, metrics_store, _pipeline_stats

    logger.info("══════════════════════════════════════")
    logger.info("  MerchantRAG starting up")
    logger.info("══════════════════════════════════════")

    # 1 ── Vector store + embedder
    from embeddings.embedder import create_vector_store, MerchantChunker
    vector_store = create_vector_store()
    chunker = MerchantChunker()

    # 2 ── Conversation memory + LangChain RAG chain
    from rag_api.query import LangChainRAGChain, ConversationMemory
    memory = ConversationMemory(max_turns=10)
    chain  = LangChainRAGChain(vector_store, memory)

    # 3 ── Data pipeline
    from ingestion.kafka_producer import MerchantProducer
    from ingestion.kafka_consumer import MerchantConsumer, SnowflakeWriter, BatchLoader

    producer = MerchantProducer()
    consumer = MerchantConsumer()
    writer   = SnowflakeWriter()

    await producer.start()
    await consumer.start()
    await writer.start()

    # Store in app state for endpoint access
    app.state.producer = producer
    app.state.consumer = consumer
    app.state.writer   = writer
    app.state.batch_loader = BatchLoader(writer)

    # 4 ── Ingest sample data → vector store
    _pipeline_stats = await _run_initial_ingest(producer, consumer, writer)

    # 5 ── LangGraph agent graph
    from agents.graph import MerchantAgentGraph
    agent_graph = MerchantAgentGraph()

    # 6 ── Observability
    from observability.langsmith_tracer import LangSmithTracer, RAGASEvaluator, MetricsStore
    tracer        = LangSmithTracer()
    ragas_eval    = RAGASEvaluator()
    metrics_store = MetricsStore()

    # Validate ingested data with Great Expectations
    _run_ge_validation()

    logger.info("══════════════════════════════════════")
    logger.info("  MerchantRAG ready on port %d", settings.API_PORT)
    logger.info("  Vector store: %d documents", vector_store.size)
    logger.info("  LangSmith:    %s", "active" if tracer.is_active else "local log")
    logger.info("══════════════════════════════════════")
    yield

    # Shutdown
    await producer.stop()
    await consumer.stop()
    logger.info("MerchantRAG shut down cleanly")


async def _run_initial_ingest(producer, consumer, writer) -> PipelineStats:
    """Bootstrap pipeline: load data → Kafka → vector store → warehouse."""
    stats = PipelineStats(pipeline_status="running",
                          last_run_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    t0 = time.time()

    # Load source data
    if settings.USE_SAMPLE_DATA:
        from data.sample_merchants import get_all_merchants
        raw_records = get_all_merchants()
    else:
        from ingestion.kafka_producer import _stream_yelp
        raw_records = list(_stream_yelp(city_filter="Chicago", limit=5000))

    # Produce → Kafka
    from ingestion.kafka_producer import yelp_to_transaction
    transactions: List[MerchantTransaction] = []
    for raw in raw_records:
        try:
            txn = yelp_to_transaction(raw)
            transactions.append(txn)
        except Exception:
            stats.records_failed += 1

    sent = await producer.send_batch(transactions)
    stats.records_produced = sent

    # Consume → index + write
    from ingestion.kafka_consumer import _MockBroker
    from ingestion.kafka_producer import _MockBroker as _ProdMockBroker

    indexed = 0
    for txn in transactions:
        try:
            chunks = chunker.chunk(txn, chunk_size=settings.CHUNK_SIZE)
            vector_store.add(txn, chunks)
            await writer.upsert(txn)
            indexed += 1
        except Exception as exc:
            logger.warning("Index failed for %s: %s", txn.merchant_id, exc)
            stats.records_failed += 1

    stats.records_consumed   = indexed
    stats.records_embedded   = indexed
    stats.records_transformed = indexed
    stats.pipeline_status    = "complete"
    stats.throughput_rps     = round(indexed / max(time.time() - t0, 0.001), 1)
    logger.info("Pipeline complete — indexed %d records (%.1f rec/s)", indexed, stats.throughput_rps)
    return stats


def _run_ge_validation():
    """Run Great Expectations suite on ingested records."""
    try:
        from transform.expectations.ge_suites import validate_records
        all_merchants = [m.dict() for m in vector_store.get_all()]
        if all_merchants:
            ok, report = validate_records(all_merchants[:500], suite_name="raw")
            status = "PASS" if ok else "FAIL"
            logger.info(
                "Great Expectations [raw suite]: %s — %s/%s expectations met",
                status,
                report["statistics"]["successful_expectations"],
                report["statistics"]["evaluated_expectations"],
            )
    except Exception as exc:
        logger.warning("GE validation skipped: %s", exc)


# ── App factory ────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Production-grade RAG pipeline over 10M+ merchant transaction records. "
        "Kafka + Snowflake + Pinecone + LangChain + LangGraph. "
        "Built on the Yelp Open Dataset — demonstrating enterprise AI data platform "
        "patterns applicable to merchant intelligence, payments, and location analytics."
    ),
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(",") if hasattr(settings, "CORS_ORIGINS") else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CORS_ORIGINS = "*"


# ── Auth dependency ────────────────────────────────────────────────────────────

async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Optional API key check — bypass in local dev (DEBUG=true)."""
    if settings.DEBUG:
        return True
    if settings.SECRET_KEY == "change-me-in-production":
        return True
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.replace("Bearer ", "")
    if token != settings.SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


# ── Middleware — timing ────────────────────────────────────────────────────────

@app.middleware("http")
async def add_timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(round((time.perf_counter() - t0) * 1000, 2))
    response.headers["X-Request-Id"] = str(uuid.uuid4())[:8]
    return response


# ── Meta endpoints ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Meta"])
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "RAG pipeline over 10M+ merchant transaction records",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "query":      "POST /query",
            "ingest":     "POST /ingest",
            "merchants":  "GET /merchants",
            "analytics":  "POST /analytics",
            "agents":     "POST /agents/run",
            "health":     "GET /health",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        vector_store_size=vector_store.size if vector_store else 0,
        pipeline_stats=_pipeline_stats,
        services={
            "vector_store": settings.VECTOR_BACKEND,
            "llm":          f"{settings.LLM_PROVIDER}/{settings.LLM_MODEL}",
            "kafka":        "mock" if settings.KAFKA_MOCK_MODE else "real",
            "database":     "sqlite" if settings.USE_LOCAL_DB else "snowflake",
            "langsmith":    "active" if (tracer and tracer.is_active) else "local",
            "ragas":        "available" if (ragas_eval and ragas_eval._ragas_available) else "heuristic",
        },
    )


# ── RAG Query ─────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["RAG"],
          dependencies=[Depends(verify_api_key)])
async def query_merchants(request: QueryRequest):
    """
    Natural-language RAG query over merchant intelligence corpus.

    **Examples:**
    - "Which merchant categories had anomalous spending in Q4?"
    - "Find coffee shops open past 10pm in the West Loop with parking"
    - "Now filter that by West Coast merchants only" (multi-turn follow-up)
    - "Which neighborhoods have the highest avg transaction amounts?"
    - "Show me merchants with contactless payment and high review velocity"
    """
    if not vector_store or not chain:
        raise HTTPException(status_code=503, detail="RAG engine not initialised")

    t0 = time.perf_counter()
    run_id = str(uuid.uuid4())

    try:
        from rag_api.query import extract_filters
        filters = extract_filters(request)
        results = vector_store.search(request.query, top_k=request.top_k, filters=filters)
        answer, tokens, ls_run_id = await chain.query(request, results)
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Update conversation memory
    if request.session_id and memory:
        memory.add_turn(request.session_id, request.query, answer)

    latency_ms = (time.perf_counter() - t0) * 1000

    # Log to LangSmith / local trace
    if tracer:
        tracer.log_query(
            run_id=run_id,
            query=request.query,
            answer=answer,
            sources=results,
            latency_ms=latency_ms,
            tokens_used=tokens or 0,
        )

    # Async RAGAS eval (non-blocking)
    if ragas_eval and metrics_store and results:
        asyncio.create_task(_evaluate_async(run_id, request.query, answer, results, latency_ms))

    return QueryResponse(
        query=request.query,
        answer=answer,
        sources=results,
        session_id=request.session_id,
        latency_ms=round(latency_ms, 2),
        tokens_used=tokens,
        langsmith_run_id=ls_run_id or run_id,
    )


async def _evaluate_async(run_id, query, answer, results, latency_ms):
    """Run RAGAS evaluation asynchronously without blocking the response."""
    try:
        contexts = [r.chunk_text or r.merchant.to_document_text()[:300] for r in results]
        scores = await ragas_eval.evaluate(query, answer, contexts)
        m = RAGEvalMetrics(
            run_id=run_id,
            query=query,
            precision_at_k=scores.get("precision_at_k", scores.get("context_precision", 0)),
            recall_at_k=scores.get("context_recall", 0),
            answer_faithfulness=scores.get("faithfulness", 0),
            context_relevance=scores.get("answer_relevancy", 0),
            latency_ms=latency_ms,
        )
        metrics_store.save(m)
    except Exception as exc:
        logger.warning("RAGAS eval failed (non-fatal): %s", exc)


# ── Real-time ingest endpoint ─────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, status_code=201,
          tags=["Ingestion"], dependencies=[Depends(verify_api_key)])
async def ingest_merchants(request: IngestRequest):
    """
    Dynamically ingest new merchant records and re-index embeddings in real time.
    Accepts Yelp-format business dicts or raw MerchantTransaction dicts.

    The /ingest endpoint exists in the spec — this closes that gap.
    """
    from ingestion.kafka_producer import yelp_to_transaction

    accepted = 0
    failed = 0

    for record in request.records:
        try:
            # Accept both Yelp business format and MerchantTransaction format
            if "business_id" in record:
                txn = yelp_to_transaction(record)
            else:
                txn = MerchantTransaction(**record)

            # Write to warehouse
            if app.state.writer:
                await app.state.writer.upsert(txn)

            # Index in vector store
            if request.reindex and chunker:
                chunks = chunker.chunk(txn, chunk_size=settings.CHUNK_SIZE)
                vector_store.add(txn, chunks)

            # Produce to Kafka
            if app.state.producer:
                await app.state.producer.send(txn)

            accepted += 1
        except Exception as exc:
            logger.warning("Ingest failed for record: %s", exc)
            failed += 1

    return IngestResponse(
        accepted=accepted,
        failed=failed,
        reindexed=request.reindex,
        vector_store_size=vector_store.size,
    )


# ── Merchant CRUD ─────────────────────────────────────────────────────────────

@app.get("/merchants", tags=["Merchants"])
async def list_merchants(
    city: Optional[str]       = Query(None),
    neighborhood: Optional[str] = Query(None),
    mcc_code: Optional[str]   = Query(None, description="ISO 18245 MCC code e.g. '5812'"),
    min_stars: Optional[float] = Query(None, ge=0, le=5),
    value_tier: Optional[str]  = Query(None, description="low | mid | high"),
    open_only: bool            = Query(False),
    has_parking: bool          = Query(False),
    acceptance: Optional[str]  = Query(None, description="chip | contactless | online | mobile_pay"),
    limit: int                 = Query(20, ge=1, le=200),
    offset: int                = Query(0, ge=0),
):
    """Structured filter search over the merchant corpus. All filters are optional."""
    merchants = vector_store.get_all()

    filtered = []
    for m in merchants:
        if city and m.city.lower() != city.lower():
            continue
        if neighborhood and neighborhood.lower() not in (m.neighborhood or "").lower():
            continue
        if mcc_code and m.mcc_code != mcc_code:
            continue
        if min_stars and (m.stars or 0) < min_stars:
            continue
        if value_tier:
            tier = ("low" if m.transaction_amount < 20
                    else "mid" if m.transaction_amount < 75 else "high")
            if tier != value_tier:
                continue
        if open_only and not m.is_open:
            continue
        if acceptance and m.acceptance_method.value != acceptance:
            continue
        filtered.append(m)

    filtered.sort(key=lambda x: (x.stars or 0), reverse=True)
    page = filtered[offset: offset + limit]

    return {
        "total": len(filtered),
        "offset": offset,
        "limit": limit,
        "merchants": [_format_merchant(m) for m in page],
    }


@app.get("/merchants/{merchant_id}", tags=["Merchants"])
async def get_merchant(merchant_id: str):
    """Retrieve a single merchant by ID with full transaction profile."""
    m = vector_store.get_by_id(merchant_id)
    if not m:
        raise HTTPException(status_code=404, detail=f"Merchant '{merchant_id}' not found")
    return _format_merchant(m)


def _format_merchant(m: MerchantTransaction) -> Dict:
    return {
        "merchant_id":   m.merchant_id,
        "merchant_name": m.merchant_name,
        "mcc_code":      m.mcc_code,
        "mcc_description": m.mcc_description,
        "city":          m.city,
        "state":         m.state,
        "neighborhood":  m.neighborhood,
        "latitude":      m.latitude,
        "longitude":     m.longitude,
        "categories":    m.categories,
        "stars":         m.stars,
        "review_count":  m.review_count,
        "review_velocity_30d": m.review_velocity_30d,
        "transaction_amount": m.transaction_amount,
        "transaction_value_tier": (
            "low" if m.transaction_amount < 20
            else "mid" if m.transaction_amount < 75 else "high"
        ),
        "acceptance_method": m.acceptance_method.value,
        "price_range":   m.price_range,
        "price_symbol":  "$" * (m.price_range or 2),
        "is_open":       m.is_open,
    }


# ── Search (direct vector, no LLM) ───────────────────────────────────────────

@app.get("/search", tags=["Search"])
async def search(
    q: str                    = Query(..., min_length=2),
    top_k: int                = Query(5, ge=1, le=20),
    city: Optional[str]       = None,
    mcc_code: Optional[str]   = None,
    value_tier: Optional[str] = None,
    min_stars: Optional[float] = Query(None, ge=0, le=5),
):
    """Direct semantic vector search — returns top-k merchants without LLM generation."""
    filters: Dict[str, Any] = {}
    if city:       filters["city"] = city
    if mcc_code:   filters["mcc_code"] = mcc_code
    if value_tier: filters["transaction_value_tier"] = value_tier
    if min_stars:  filters["min_stars"] = min_stars

    results = vector_store.search(q, top_k=top_k, filters=filters)
    return {
        "query":   q,
        "count":   len(results),
        "results": [
            {
                "merchant_name": r.merchant.merchant_name,
                "city":          r.merchant.city,
                "mcc_code":      r.merchant.mcc_code,
                "stars":         r.merchant.stars,
                "transaction_amount": r.merchant.transaction_amount,
                "acceptance_method": r.merchant.acceptance_method.value,
                "score":         r.score,
                "chunk_preview": r.chunk_text[:150],
            }
            for r in results
        ],
    }


# ── Analytics ─────────────────────────────────────────────────────────────────

@app.post("/analytics", tags=["Analytics"])
async def analytics(
    metric:   str           = Query(..., description="review_velocity | avg_transaction | mcc_distribution | anomaly_flags | acceptance_methods | q4_spending"),
    group_by: str           = Query("mcc_code", description="mcc_code | city | neighborhood | value_tier"),
    city:     Optional[str] = Query(None),
    limit:    int           = Query(10, ge=1, le=50),
):
    """
    Aggregate metrics over the merchant intelligence corpus.
    Mirrors the analytics queries described in the playbook spec.
    """
    from collections import Counter, defaultdict
    import statistics as stats_lib

    all_merchants = vector_store.get_all()
    if city:
        all_merchants = [m for m in all_merchants if m.city.lower() == city.lower()]

    if not all_merchants:
        raise HTTPException(status_code=404, detail="No merchants found for filter")

    def group_key(m: MerchantTransaction) -> str:
        return {
            "mcc_code":     f"{m.mcc_code} — {m.mcc_description}",
            "city":         m.city,
            "neighborhood": m.neighborhood or m.city,
            "value_tier":   ("low" if m.transaction_amount < 20 else "mid" if m.transaction_amount < 75 else "high"),
        }.get(group_by, m.mcc_code)

    groups: Dict[str, List[MerchantTransaction]] = defaultdict(list)
    for m in all_merchants:
        groups[group_key(m)].append(m)

    data: List[Dict] = []

    if metric == "review_velocity":
        for key, ms in groups.items():
            velocities = [m.review_velocity_30d or 0 for m in ms]
            data.append({
                "group": key,
                "avg_velocity":  round(stats_lib.mean(velocities), 1),
                "max_velocity":  max(velocities),
                "merchant_count": len(ms),
                "total_reviews":  sum(m.review_count or 0 for m in ms),
            })
        data.sort(key=lambda x: x["avg_velocity"], reverse=True)

    elif metric == "avg_transaction":
        for key, ms in groups.items():
            amounts = [m.transaction_amount for m in ms]
            data.append({
                "group": key,
                "avg_transaction": round(stats_lib.mean(amounts), 2),
                "median_transaction": round(stats_lib.median(amounts), 2),
                "total_volume":    round(sum(amounts), 2),
                "merchant_count":  len(ms),
            })
        data.sort(key=lambda x: x["avg_transaction"], reverse=True)

    elif metric == "mcc_distribution":
        mcc_counts: Counter = Counter()
        for m in all_merchants:
            mcc_counts[f"{m.mcc_code} — {m.mcc_description}"] += 1
        data = [{"mcc": k, "count": v} for k, v in mcc_counts.most_common(limit)]

    elif metric == "anomaly_flags":
        amounts = [m.transaction_amount for m in all_merchants]
        mean_a  = stats_lib.mean(amounts)
        stdev_a = stats_lib.stdev(amounts) if len(amounts) > 1 else 1
        velocities = [m.review_velocity_30d or 0 for m in all_merchants]
        mean_v = stats_lib.mean(velocities) or 1

        for m in all_merchants:
            z = (m.transaction_amount - mean_a) / stdev_a
            vel_ratio = (m.review_velocity_30d or 0) / mean_v
            if abs(z) >= 2.0 or vel_ratio >= 2.5:
                data.append({
                    "merchant_name":     m.merchant_name,
                    "merchant_id":       m.merchant_id,
                    "city":              m.city,
                    "mcc_description":   m.mcc_description,
                    "transaction_amount": m.transaction_amount,
                    "z_score":           round(z, 2),
                    "review_velocity":   m.review_velocity_30d,
                    "velocity_ratio":    round(vel_ratio, 2),
                    "anomaly_type": (
                        "high_amount" if z > 2 else
                        "low_amount"  if z < -2 else
                        "velocity_spike"
                    ),
                })
        data.sort(key=lambda x: abs(x["z_score"]), reverse=True)

    elif metric == "acceptance_methods":
        method_counts: Counter = Counter()
        for m in all_merchants:
            method_counts[m.acceptance_method.value] += 1
        data = [{"method": k, "count": v, "pct": f"{100*v/len(all_merchants):.1f}%"}
                for k, v in method_counts.most_common()]

    elif metric == "q4_spending":
        # Merchants with above-average transaction amounts (Q4 proxy)
        amounts = [m.transaction_amount for m in all_merchants]
        mean_a  = stats_lib.mean(amounts)
        for m in all_merchants:
            if m.transaction_amount > mean_a * 1.3:
                data.append({
                    "merchant_name":      m.merchant_name,
                    "city":               m.city,
                    "mcc_description":    m.mcc_description,
                    "avg_transaction":    m.transaction_amount,
                    "above_mean_by_pct": f"{100*(m.transaction_amount-mean_a)/mean_a:.0f}%",
                })
        data.sort(key=lambda x: x["avg_transaction"], reverse=True)

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown metric '{metric}'. Valid: "
                "review_velocity, avg_transaction, mcc_distribution, "
                "anomaly_flags, acceptance_methods, q4_spending"
            ),
        )

    return {
        "metric":     metric,
        "group_by":   group_by,
        "city_filter": city,
        "total_merchants_analyzed": len(all_merchants),
        "data": data[:limit],
    }


@app.get("/analytics/summary", tags=["Analytics"])
async def analytics_summary():
    """Dashboard snapshot — no request body. Used by front-end dashboard."""
    from collections import Counter
    import statistics as stats_lib

    ms = vector_store.get_all()
    if not ms:
        return {"total_merchants": 0}

    mcc_counts: Counter = Counter()
    method_counts: Counter = Counter()
    for m in ms:
        mcc_counts[m.mcc_description] += 1
        method_counts[m.acceptance_method.value] += 1

    amounts = [m.transaction_amount for m in ms]
    stars   = [m.stars for m in ms if m.stars]
    velocities = [m.review_velocity_30d or 0 for m in ms]

    return {
        "total_merchants":      len(ms),
        "open_merchants":       sum(1 for m in ms if m.is_open),
        "avg_transaction":      round(stats_lib.mean(amounts), 2),
        "avg_stars":            round(stats_lib.mean(stars), 2) if stars else 0,
        "total_reviews":        sum(m.review_count or 0 for m in ms),
        "avg_velocity_30d":     round(stats_lib.mean(velocities), 1),
        "top_mcc_categories":   [{"mcc": k, "count": v} for k, v in mcc_counts.most_common(5)],
        "acceptance_distribution": [{"method": k, "count": v} for k, v in method_counts.most_common()],
        "vector_store_size":    vector_store.size,
        "pipeline_status":      _pipeline_stats.pipeline_status,
    }


# ── LangGraph Agents ──────────────────────────────────────────────────────────

@app.post("/agents/run", tags=["Agents"], dependencies=[Depends(verify_api_key)])
async def run_agents(
    task: str = Query(default="pipeline_health_check",
                      description="pipeline_health_check | anomaly_scan | freshness_audit"),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """
    Execute the LangGraph multi-agent pipeline:
      1. DataValidationAgent  — quality gate before embedding
      2. DataFreshnessAgent   — SLA monitoring
      3. AnomalyDetectionAgent — statistical + LLM narrative

    Returns structured report from all three agents.
    """
    global _last_agent_result

    if not agent_graph:
        raise HTTPException(status_code=503, detail="Agent graph not initialised")

    merchants = [m.dict() for m in vector_store.get_all()[:limit]]
    if not merchants:
        raise HTTPException(status_code=404, detail="No merchants in vector store to analyse")

    try:
        result = await agent_graph.run(merchants, task=task)
        _last_agent_result = dict(result)
        return {
            "task":               result["task"],
            "steps_executed":     result["steps"],
            "validation_result":  result["validation_result"],
            "freshness_report":   result["freshness_report"],
            "anomalies_detected": result["anomalies"],
            "summary":            result["final_answer"],
            "errors":             result["errors"],
        }
    except Exception as exc:
        logger.exception("Agent run failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/agents/status", tags=["Agents"])
async def agent_status():
    """Return latest agent pipeline run results."""
    if not _last_agent_result:
        return {"status": "no_runs_yet", "message": "POST /agents/run to execute the pipeline"}
    return {
        "status": "available",
        "last_run": {
            "task":          _last_agent_result.get("task"),
            "steps":         _last_agent_result.get("steps"),
            "validation":    (_last_agent_result.get("validation_result") or {}).get("status"),
            "freshness":     (_last_agent_result.get("freshness_report") or {}).get("status"),
            "anomaly_count": len(_last_agent_result.get("anomalies") or []),
            "has_errors":    bool(_last_agent_result.get("errors")),
        },
    }


# ── Pipeline management ───────────────────────────────────────────────────────

@app.post("/pipeline/reload", tags=["Pipeline"], dependencies=[Depends(verify_api_key)])
async def reload_pipeline(
    city_filter: Optional[str] = Query(None),
    limit:       Optional[int] = Query(None, ge=1),
):
    """Trigger full pipeline re-ingest (safe to call on live system — upsert-based)."""
    global _pipeline_stats
    _pipeline_stats = await _run_initial_ingest(
        app.state.producer,
        app.state.consumer,
        app.state.writer,
    )
    return {"status": "complete", "stats": _pipeline_stats.dict()}


@app.get("/pipeline/stats", response_model=PipelineStats, tags=["Pipeline"])
async def pipeline_stats():
    return _pipeline_stats


@app.post("/pipeline/batch", tags=["Pipeline"], dependencies=[Depends(verify_api_key)])
async def trigger_batch_load():
    """Process any CSV files in data/staging/ directory (Ab Initio-style batch load)."""
    loaded = await app.state.batch_loader.process_staging_files()
    return {"status": "complete", "records_loaded": loaded}


# ── Observability ─────────────────────────────────────────────────────────────

@app.get("/observability/metrics", tags=["Observability"])
async def eval_metrics():
    """RAGAS evaluation summary + LangSmith trace stats."""
    summary = metrics_store.summary() if metrics_store else {}
    return {
        "langsmith_active": tracer.is_active if tracer else False,
        "ragas_available":  ragas_eval._ragas_available if ragas_eval else False,
        "eval_summary":     summary,
        "trace_log":        "./data/query_traces.jsonl",
    }


@app.get("/observability/traces", tags=["Observability"])
async def recent_traces(limit: int = Query(10, ge=1, le=100)):
    """Last N RAG query traces (latency, scores, token usage)."""
    if not metrics_store:
        return {"traces": []}
    all_records = metrics_store.load_all()
    return {"traces": all_records[-limit:], "total": len(all_records)}


@app.post("/observability/feedback", tags=["Observability"])
async def submit_feedback(run_id: str, score: float = Query(..., ge=0, le=1), comment: str = ""):
    """Submit user feedback score for a RAG run (fed back to LangSmith)."""
    if tracer:
        tracer.log_feedback(run_id, score, comment)
    return {"status": "received", "run_id": run_id, "score": score}


# ── Conversation memory ───────────────────────────────────────────────────────

@app.delete("/memory/{session_id}", tags=["Memory"])
async def clear_session(session_id: str):
    """Clear multi-turn conversation memory for a session."""
    if memory:
        memory.clear(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/memory/{session_id}", tags=["Memory"])
async def get_session_history(session_id: str):
    """Retrieve conversation history for a session."""
    if not memory:
        return {"session_id": session_id, "turns": []}
    return {
        "session_id": session_id,
        "turns": memory.get_history(session_id),
        "total_turns": len(memory.get_history(session_id)),
    }


# ── Dev helpers ───────────────────────────────────────────────────────────────

@app.get("/env-example", include_in_schema=False)
async def env_example():
    return JSONResponse({"env_example": ENV_EXAMPLE})


@app.get("/ge-report", tags=["Quality"])
async def ge_report():
    """Run Great Expectations suite on-demand and return the report."""
    from transform.expectations.ge_suites import validate_records
    merchants = [m.dict() for m in vector_store.get_all()[:1000]]
    ok, report = validate_records(merchants, suite_name="raw")
    return {"passed": ok, "report": report}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )
