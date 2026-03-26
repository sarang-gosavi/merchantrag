"""
infra/lambda_handler.py — AWS Lambda entry point for serverless RAG API.

Handles two event types:
  1. API Gateway events   → /query and /ingest endpoints
  2. S3 trigger events    → batch ingest on file arrival

Deploy:
  zip -r function.zip . && aws lambda update-function-code \
    --function-name merchantrag-query \
    --zip-file fileb://function.zip

Environment variables to set in Lambda console:
  ANTHROPIC_API_KEY, PINECONE_API_KEY, USE_LOCAL_VECTORS=false,
  KAFKA_MOCK_MODE=true, USE_LOCAL_DB=false, + Snowflake creds
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ── Lambda response helper ────────────────────────────────────────────────────

def response(status: int, body: Any, headers: Dict = None) -> Dict:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            **(headers or {}),
        },
        "body": json.dumps(body) if not isinstance(body, str) else body,
    }


def error(status: int, message: str) -> Dict:
    return response(status, {"error": message, "status": status})


# ── Lazy-loaded app state (persists across warm invocations) ──────────────────

_app_state = {}


def _get_state():
    """Initialize vector store and RAG chain once per Lambda container lifetime."""
    if "vector_store" not in _app_state:
        logger.info("Cold start — initialising vector store and RAG chain")

        from embeddings.embedder import create_vector_store, create_embedder, MerchantChunker
        from rag_api.query import LangChainRAGChain, ConversationMemory

        vector_store = create_vector_store()
        memory = ConversationMemory()
        chain = LangChainRAGChain(vector_store, memory)
        chunker = MerchantChunker()

        _app_state.update({
            "vector_store": vector_store,
            "memory": memory,
            "chain": chain,
            "chunker": chunker,
        })
        logger.info("Cold start complete")

    return _app_state


# ── API Gateway handler ────────────────────────────────────────────────────────

async def _handle_query(body: Dict) -> Dict:
    import asyncio
    from models import QueryRequest, QueryResponse, RetrievalResult
    from rag_api.query import extract_filters
    import time

    state = _get_state()
    vector_store = state["vector_store"]
    chain = state["chain"]

    try:
        req = QueryRequest(**body)
    except Exception as exc:
        return error(422, f"Invalid request: {exc}")

    t0 = time.perf_counter()
    filters = extract_filters(req)
    results = vector_store.search(req.query, top_k=req.top_k, filters=filters)

    answer, tokens, ls_run_id = await chain.query(req, results)

    if req.session_id:
        state["memory"].add_turn(req.session_id, req.query, answer)

    latency_ms = (time.perf_counter() - t0) * 1000

    return response(200, {
        "query": req.query,
        "answer": answer,
        "sources": [
            {
                "merchant_name": r.merchant.merchant_name,
                "city": r.merchant.city,
                "mcc_code": r.merchant.mcc_code,
                "stars": r.merchant.stars,
                "score": r.score,
            }
            for r in results
        ],
        "session_id": req.session_id,
        "latency_ms": round(latency_ms, 2),
        "tokens_used": tokens,
        "langsmith_run_id": ls_run_id,
    })


async def _handle_ingest(body: Dict) -> Dict:
    from models import IngestRequest, MerchantTransaction
    from ingestion.kafka_producer import yelp_to_transaction

    state = _get_state()
    vector_store = state["vector_store"]
    chunker = state["chunker"]

    try:
        req = IngestRequest(**body)
    except Exception as exc:
        return error(422, f"Invalid ingest request: {exc}")

    accepted = 0
    failed = 0
    for record in req.records:
        try:
            txn = yelp_to_transaction(record) if "business_id" in record else MerchantTransaction(**record)
            from config import settings
            chunks = chunker.chunk(txn, chunk_size=settings.CHUNK_SIZE)
            vector_store.add(txn, chunks)
            accepted += 1
        except Exception as exc:
            logger.warning("Failed to ingest record: %s", exc)
            failed += 1

    return response(200, {
        "accepted": accepted,
        "failed": failed,
        "reindexed": req.reindex,
        "vector_store_size": vector_store.size,
    })


# ── S3 trigger handler ────────────────────────────────────────────────────────

async def _handle_s3_event(event: Dict) -> Dict:
    """
    Triggered when a new file lands in the S3 staging bucket.
    Downloads the file, ingests transactions, updates vector store.
    """
    import boto3
    import csv
    import io

    s3 = boto3.client("s3")
    state = _get_state()
    vector_store = state["vector_store"]
    chunker = state["chunker"]

    from config import settings
    from ingestion.kafka_producer import yelp_to_transaction

    total_ingested = 0

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]

        logger.info("Processing S3 file: s3://%s/%s", bucket, key)

        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            content = obj["Body"].read().decode("utf-8")

            if key.endswith(".json"):
                import json
                lines = [json.loads(line) for line in content.splitlines() if line.strip()]
            elif key.endswith(".csv"):
                reader = csv.DictReader(io.StringIO(content))
                lines = list(reader)
            else:
                logger.warning("Unsupported file type: %s", key)
                continue

            for raw in lines:
                try:
                    txn = yelp_to_transaction(raw)
                    chunks = chunker.chunk(txn, chunk_size=settings.CHUNK_SIZE)
                    vector_store.add(txn, chunks)
                    total_ingested += 1
                except Exception as exc:
                    logger.warning("Failed to process record: %s", exc)

            logger.info("Ingested %d records from %s", total_ingested, key)

            # Move processed file to archive prefix
            archive_key = key.replace("staging/", "archive/")
            s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": key}, Key=archive_key)
            s3.delete_object(Bucket=bucket, Key=key)

        except Exception as exc:
            logger.error("Failed to process S3 file %s: %s", key, exc)

    return response(200, {"ingested": total_ingested})


# ── Lambda entrypoint ─────────────────────────────────────────────────────────

def handler(event: Dict, context: Any) -> Dict:
    """
    Main Lambda handler.
    Dispatches to query, ingest, or S3 trigger based on event type.
    """
    import asyncio

    # S3 trigger
    if event.get("Records") and event["Records"][0].get("s3"):
        return asyncio.run(_handle_s3_event(event))

    # API Gateway
    http_method = event.get("httpMethod", "").upper()
    path = event.get("path", "/")

    # Handle CORS preflight
    if http_method == "OPTIONS":
        return response(200, {})

    # Check auth header
    auth_header = (event.get("headers") or {}).get("Authorization", "")
    api_key = os.environ.get("API_KEY", "")
    if api_key and not auth_header.startswith(f"Bearer {api_key}"):
        return error(401, "Unauthorized — provide valid Bearer token")

    body = {}
    raw_body = event.get("body", "{}")
    if raw_body:
        try:
            body = json.loads(raw_body)
        except Exception:
            return error(400, "Invalid JSON body")

    if http_method == "POST" and path == "/query":
        return asyncio.run(_handle_query(body))

    if http_method == "POST" and path == "/ingest":
        return asyncio.run(_handle_ingest(body))

    if http_method == "GET" and path == "/health":
        state = _get_state()
        return response(200, {
            "status": "ok",
            "vector_store_size": state["vector_store"].size,
            "active_sessions": state["memory"].active_sessions(),
        })

    return error(404, f"Route not found: {http_method} {path}")
