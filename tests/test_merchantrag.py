"""
tests/test_merchantrag.py — Comprehensive test suite for MerchantRAG.

Coverage:
  - Merchant data model (Pydantic validation, computed properties)
  - Kafka mock producer/consumer cycle
  - Vector store: upsert, search, filters, scores
  - Query parser: intent, MCC hints, filter extraction
  - Analytics engine: velocity, anomaly detection, distribution
  - Great Expectations suites: raw + staging validation
  - LangGraph agents: validation, freshness, anomaly detection
  - RAGAS heuristic evaluator
  - API integration: /health, /query, /ingest, /merchants, /analytics, /agents

Run:
    pytest tests/ -v
    pytest tests/ -v --cov=. --cov-report=term-missing
"""
from __future__ import annotations

import asyncio
import json
import pytest
import sys
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

# Make project importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.sample_merchants import get_all_merchants, get_by_neighborhood
from models import (
    AcceptanceMethod, AgentState, MerchantTransaction,
    QueryRequest, RetrievalResult, PipelineStats,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_merchants():
    return get_all_merchants()


@pytest.fixture
def transactions(raw_merchants):
    from ingestion.kafka_producer import yelp_to_transaction
    return [yelp_to_transaction(r) for r in raw_merchants]


@pytest.fixture
def store(transactions):
    from embeddings.embedder import LocalTFIDFStore, MerchantChunker
    vs = LocalTFIDFStore()
    chunker = MerchantChunker()
    for txn in transactions:
        chunks = chunker.chunk(txn, chunk_size=512)
        vs.add(txn, chunks)
    return vs


@pytest.fixture
def memory():
    from rag_api.query import ConversationMemory
    return ConversationMemory(max_turns=5)


# ── 1. Sample data ────────────────────────────────────────────────────────────

class TestSampleData:
    def test_returns_merchants(self, raw_merchants):
        assert len(raw_merchants) >= 20

    def test_required_schema_fields(self, raw_merchants):
        required = {"business_id", "name", "city", "stars", "review_count", "categories"}
        for m in raw_merchants:
            assert required.issubset(m.keys()), f"Missing fields in {m.get('name')}"

    def test_neighborhood_filter(self):
        wl = get_by_neighborhood("West Loop")
        assert len(wl) > 0
        assert all("West Loop" in m.get("neighborhood", "") for m in wl)

    def test_stars_in_valid_range(self, raw_merchants):
        for m in raw_merchants:
            assert 0 <= m["stars"] <= 5, f"Stars out of range: {m['stars']}"

    def test_review_count_non_negative(self, raw_merchants):
        for m in raw_merchants:
            assert m["review_count"] >= 0


# ── 2. Merchant model ─────────────────────────────────────────────────────────

class TestMerchantTransaction:
    def test_yelp_to_transaction(self, transactions):
        assert len(transactions) > 0
        for txn in transactions:
            assert isinstance(txn, MerchantTransaction)

    def test_mcc_code_is_padded(self, transactions):
        for txn in transactions:
            assert len(txn.mcc_code) == 4, f"MCC not 4 chars: {txn.mcc_code}"

    def test_acceptance_method_valid(self, transactions):
        valid = {m.value for m in AcceptanceMethod}
        for txn in transactions:
            assert txn.acceptance_method.value in valid

    def test_transaction_amount_positive(self, transactions):
        for txn in transactions:
            assert txn.transaction_amount > 0

    def test_to_document_text(self, transactions):
        for txn in transactions:
            doc = txn.to_document_text()
            assert txn.merchant_name in doc
            assert txn.mcc_code in doc
            assert txn.city in doc
            assert len(doc) > 80

    def test_price_range_defaults(self, transactions):
        for txn in transactions:
            assert 1 <= (txn.price_range or 2) <= 4


# ── 3. Kafka mock pipeline ────────────────────────────────────────────────────

class TestKafkaPipeline:
    def test_produce_and_consume(self, transactions):
        async def _run():
            from ingestion.kafka_producer import MerchantProducer
            from ingestion.kafka_consumer import MerchantConsumer

            producer = MerchantProducer()
            consumer = MerchantConsumer()
            await producer.start()
            await consumer.start()

            sent = await producer.send_batch(transactions[:5])
            assert sent == 5

            received = []
            async for txn in consumer.consume(max_records=5):
                received.append(txn)

            await producer.stop()
            assert len(received) == 5
            assert all(isinstance(t, MerchantTransaction) for t in received)

        asyncio.run(_run())

    def test_yelp_to_transaction_mcc_mapping(self):
        from ingestion.kafka_producer import yelp_to_transaction, CATEGORY_TO_MCC
        raw = {
            "business_id": "test_001",
            "name": "Test Pizza",
            "categories": "Pizza, Restaurants",
            "city": "Chicago",
            "state": "IL",
            "latitude": 41.88,
            "longitude": -87.63,
            "stars": 4.2,
            "review_count": 100,
            "is_open": 1,
            "review_velocity_30d": 20,
            "attributes": {"RestaurantsPriceRange2": 2},
        }
        txn = yelp_to_transaction(raw)
        assert txn.mcc_code == "5812"   # Pizza → Eating Places

    def test_staging_file_writer(self, transactions, tmp_path):
        import asyncio
        from ingestion.kafka_consumer import SnowflakeWriter, BatchLoader

        async def _run():
            writer = SnowflakeWriter()
            await writer.start()
            loader = BatchLoader(writer)
            loader.STAGING_DIR = tmp_path / "staging"
            loader.ARCHIVE_DIR = tmp_path / "archive"
            loader.STAGING_DIR.mkdir(parents=True)
            loader.ARCHIVE_DIR.mkdir(parents=True)
            path = loader.write_staging_file(transactions[:3])
            assert path.exists()
            loaded = await loader._load_file(path)
            assert loaded == 3

        asyncio.run(_run())


# ── 4. Embedding + vector store ───────────────────────────────────────────────

class TestLocalTFIDFStore:
    def test_upsert_and_size(self, store, transactions):
        assert store.size > 0

    def test_search_returns_results(self, store):
        results = store.search("coffee shop", top_k=3)
        assert len(results) <= 3
        assert all(hasattr(r, "merchant") for r in results)

    def test_search_coffee_finds_coffee_shops(self, store):
        results = store.search("coffee", top_k=5)
        coffee_hits = [r for r in results
                       if any("coffee" in c.lower() for c in r.merchant.categories)]
        assert len(coffee_hits) > 0

    def test_city_filter(self, store):
        results = store.search("restaurant", top_k=10, filters={"city": "Chicago"})
        assert all(r.merchant.city == "Chicago" for r in results)

    def test_mcc_filter(self, store):
        results = store.search("restaurant", top_k=10, filters={"mcc_code": "5812"})
        assert all(r.merchant.mcc_code == "5812" for r in results)

    def test_value_tier_filter(self, store):
        results = store.search("food", top_k=10, filters={"transaction_value_tier": "high"})
        for r in results:
            assert r.merchant.transaction_amount >= 75

    def test_scores_in_range(self, store):
        results = store.search("Italian pizza dinner", top_k=5)
        for r in results:
            assert 0 <= r.score <= 2.0, f"Unexpected score: {r.score}"

    def test_get_by_id(self, store, transactions):
        txn = transactions[0]
        found = store.get_by_id(txn.merchant_id)
        assert found is not None
        assert found.merchant_id == txn.merchant_id

    def test_get_all_returns_merchants(self, store):
        all_m = store.get_all()
        assert len(all_m) > 0
        assert all(isinstance(m, MerchantTransaction) for m in all_m)

    def test_chunker_produces_chunks(self, transactions):
        from embeddings.embedder import MerchantChunker
        chunker = MerchantChunker()
        chunks = chunker.chunk(transactions[0], chunk_size=512)
        assert len(chunks) >= 2
        for c in chunks:
            assert "text" in c
            assert "doc_id" in c
            assert "metadata" in c
            assert len(c["text"]) > 10


# ── 5. Query filter extraction ────────────────────────────────────────────────

class TestQueryFilters:
    def test_city_chicago(self):
        from rag_api.query import extract_filters
        req = QueryRequest(query="pizza in chicago", top_k=5)
        filters = extract_filters(req)
        assert filters.get("city") == "Chicago"

    def test_mcc_restaurant_hint(self):
        from rag_api.query import extract_filters
        req = QueryRequest(query="best restaurant for group dinner", top_k=5)
        filters = extract_filters(req)
        assert filters.get("mcc_code") == "5812"

    def test_mcc_bar_hint(self):
        from rag_api.query import extract_filters
        req = QueryRequest(query="cocktail bar with good drinks", top_k=5)
        filters = extract_filters(req)
        assert filters.get("mcc_code") == "5813"

    def test_value_tier_budget(self):
        from rag_api.query import extract_filters
        req = QueryRequest(query="cheap budget lunch spots", top_k=5)
        filters = extract_filters(req)
        assert filters.get("transaction_value_tier") == "low"

    def test_value_tier_high(self):
        from rag_api.query import extract_filters
        req = QueryRequest(query="luxury upscale fine dining experience", top_k=5)
        filters = extract_filters(req)
        assert filters.get("transaction_value_tier") == "high"

    def test_explicit_filters_override(self):
        from rag_api.query import extract_filters
        req = QueryRequest(
            query="any food",
            top_k=5,
            city_filter="New York",
            mcc_filter="5411",
            value_tier="mid",
        )
        filters = extract_filters(req)
        assert filters["city"] == "New York"
        assert filters["mcc_code"] == "5411"
        assert filters["transaction_value_tier"] == "mid"


# ── 6. Multi-turn conversation memory ────────────────────────────────────────

class TestConversationMemory:
    def test_add_and_retrieve_turns(self, memory):
        memory.add_turn("sess_1", "find coffee shops", "Here are some coffee shops...")
        history = memory.get_history("sess_1")
        assert len(history) == 1
        assert history[0]["human"] == "find coffee shops"

    def test_max_turns_trimming(self, memory):
        for i in range(8):
            memory.add_turn("sess_2", f"query {i}", f"answer {i}")
        assert len(memory.get_history("sess_2")) <= 5

    def test_format_history_non_empty(self, memory):
        memory.add_turn("sess_3", "show me pizza", "Here are pizza spots...")
        memory.add_turn("sess_3", "now only West Loop", "Filtering to West Loop...")
        fmt = memory.format_history("sess_3")
        assert "pizza" in fmt
        assert "West Loop" in fmt

    def test_clear_session(self, memory):
        memory.add_turn("sess_4", "q", "a")
        memory.clear("sess_4")
        assert memory.get_history("sess_4") == []

    def test_separate_sessions(self, memory):
        memory.add_turn("sess_A", "qa", "aa")
        memory.add_turn("sess_B", "qb", "ab")
        assert len(memory.get_history("sess_A")) == 1
        assert len(memory.get_history("sess_B")) == 1


# ── 7. Analytics ──────────────────────────────────────────────────────────────

class TestAnalytics:
    def test_review_velocity_by_mcc(self, store):
        merchants = store.get_all()
        from collections import defaultdict
        import statistics
        groups = defaultdict(list)
        for m in merchants:
            groups[m.mcc_code].append(m.review_velocity_30d or 0)
        for mcc, velocities in groups.items():
            avg = statistics.mean(velocities)
            assert avg >= 0

    def test_anomaly_detection_finds_outliers(self, store):
        from agents.graph import AnomalyDetectionAgent
        merchants = [m.dict() for m in store.get_all()]

        async def _run():
            agent = AnomalyDetectionAgent()
            state = {
                "task": "anomaly_scan",
                "merchants": merchants,
                "validation_result": None,
                "anomalies": None,
                "freshness_report": None,
                "final_answer": None,
                "errors": [],
                "steps": [],
            }
            return await agent.run(state)

        result = asyncio.run(_run())
        # Should find at least some anomalies in a real dataset
        assert "anomalies" in result
        assert result["anomalies"] is not None

    def test_ge_raw_suite(self, transactions):
        from transform.expectations.ge_suites import build_raw_suite
        records = [t.dict() for t in transactions]
        # Convert to raw dict format
        for r in records:
            r["mcc_code"] = r["mcc_code"]
            if "acceptance_method" in r:
                r["acceptance_method"] = r["acceptance_method"] if isinstance(
                    r["acceptance_method"], str) else r["acceptance_method"].value
        suite = build_raw_suite(records)
        report = suite.to_report()
        assert "statistics" in report
        assert report["statistics"]["evaluated_expectations"] > 5


# ── 8. LangGraph agents ───────────────────────────────────────────────────────

class TestLangGraphAgents:
    def _make_state(self, merchants):
        return {
            "task": "test",
            "merchants": merchants,
            "validation_result": None,
            "anomalies": None,
            "freshness_report": None,
            "final_answer": None,
            "errors": [],
            "steps": [],
        }

    def test_validation_agent_all_valid(self, transactions):
        from agents.graph import DataValidationAgent
        records = []
        for t in transactions[:10]:
            d = t.dict()
            d["mcc_code"] = t.mcc_code
            d["acceptance_method"] = t.acceptance_method.value
            records.append(d)

        async def _run():
            agent = DataValidationAgent()
            state = self._make_state(records)
            return await agent.run(state)

        result = asyncio.run(_run())
        vr = result["validation_result"]
        assert vr is not None
        assert vr["total"] == 10
        assert "pass_rate" in vr

    def test_validation_agent_catches_missing_fields(self):
        from agents.graph import DataValidationAgent
        bad_records = [{"name": "Missing ID merchant"}]

        async def _run():
            agent = DataValidationAgent()
            state = self._make_state(bad_records)
            return await agent.run(state)

        result = asyncio.run(_run())
        vr = result["validation_result"]
        assert vr["failed"] > 0

    def test_freshness_agent_detects_stale(self):
        from agents.graph import DataFreshnessAgent
        stale = [
            {
                "merchant_id": "m1",
                "merchant_name": "Old Merchant",
                "timestamp": (datetime.utcnow() - timedelta(days=10)).isoformat(),
            }
        ]

        async def _run():
            agent = DataFreshnessAgent()
            state = self._make_state(stale)
            return await agent.run(state)

        result = asyncio.run(_run())
        fr = result["freshness_report"]
        assert fr["stale_critical"] == 1
        assert fr["status"] in ("critical", "warning")

    def test_freshness_agent_fresh_data(self):
        from agents.graph import DataFreshnessAgent
        fresh = [
            {
                "merchant_id": "m2",
                "merchant_name": "Fresh Merchant",
                "timestamp": datetime.utcnow().isoformat(),
            }
        ]

        async def _run():
            agent = DataFreshnessAgent()
            state = self._make_state(fresh)
            return await agent.run(state)

        result = asyncio.run(_run())
        fr = result["freshness_report"]
        assert fr["fresh"] == 1

    def test_full_agent_graph(self, transactions):
        from agents.graph import MerchantAgentGraph
        records = []
        for t in transactions[:15]:
            d = t.dict()
            d["acceptance_method"] = t.acceptance_method.value
            d["timestamp"] = t.timestamp.isoformat()
            records.append(d)

        async def _run():
            graph = MerchantAgentGraph()
            return await graph.run(records, task="test_run")

        result = asyncio.run(_run())
        assert result["validation_result"] is not None
        assert result["freshness_report"] is not None
        assert result["anomalies"] is not None
        assert result["final_answer"] is not None
        assert len(result["steps"]) == 3


# ── 9. RAGAS heuristic evaluator ─────────────────────────────────────────────

class TestRAGASEvaluator:
    def test_heuristic_evaluate_returns_scores(self):
        from observability.langsmith_tracer import RAGASEvaluator
        eval = RAGASEvaluator()
        scores = eval._heuristic_evaluate(
            query="coffee shops in West Loop",
            answer="Intelligentsia Coffee is a great coffee shop in the West Loop.",
            contexts=["Intelligentsia Coffee is located at 53 W Jackson, West Loop, Chicago."],
        )
        assert "faithfulness" in scores
        assert "context_precision" in scores
        assert all(0 <= v <= 1 for v in scores.values() if v is not None)

    def test_precision_at_k(self):
        from observability.langsmith_tracer import RAGASEvaluator
        eval = RAGASEvaluator()
        relevant = ["doc_1", "doc_2", "doc_3"]
        retrieved = ["doc_1", "doc_99", "doc_2", "doc_100", "doc_3"]
        p_at_5 = eval.precision_at_k(relevant, retrieved, k=5)
        assert p_at_5 == pytest.approx(3/5)

    def test_recall_at_k(self):
        from observability.langsmith_tracer import RAGASEvaluator
        eval = RAGASEvaluator()
        relevant = ["doc_1", "doc_2", "doc_3"]
        retrieved = ["doc_1", "doc_2"]
        r_at_5 = eval.recall_at_k(relevant, retrieved, k=5)
        assert r_at_5 == pytest.approx(2/3)


# ── 10. FastAPI integration (no LLM key needed) ───────────────────────────────

class TestAPIIntegration:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        # Patch lifespan to be a no-op for speed
        import main as m

        # Ensure store is populated
        from embeddings.embedder import LocalTFIDFStore, MerchantChunker
        from ingestion.kafka_producer import yelp_to_transaction
        from data.sample_merchants import get_all_merchants

        vs = LocalTFIDFStore()
        chunker = MerchantChunker()
        for raw in get_all_merchants():
            txn = yelp_to_transaction(raw)
            vs.add(txn, chunker.chunk(txn))

        m.vector_store = vs
        m.memory = __import__("rag_api.query", fromlist=["ConversationMemory"]).ConversationMemory()
        m.chain = __import__("rag_api.query", fromlist=["LangChainRAGChain"]).LangChainRAGChain(vs, m.memory)
        m.chunker = chunker
        m._pipeline_stats = PipelineStats(pipeline_status="complete", records_produced=30)
        from agents.graph import MerchantAgentGraph
        m.agent_graph = MerchantAgentGraph()
        from observability.langsmith_tracer import LangSmithTracer, RAGASEvaluator, MetricsStore
        m.tracer = LangSmithTracer()
        m.ragas_eval = RAGASEvaluator()
        m.metrics_store = MetricsStore()

        # Stub startup/shutdown
        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def _noop(app):
            yield
        m.app.router.lifespan_context = _noop

        class MockAppState:
            producer = None
            consumer = None
            writer = None
            batch_loader = None
        m.app.state = MockAppState()

        return TestClient(m.app)

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "MerchantRAG" in r.json()["name"]

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["vector_store_size"] > 0

    def test_merchants_list(self, client):
        r = client.get("/merchants?limit=5")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] > 0
        assert len(body["merchants"]) <= 5

    def test_merchants_city_filter(self, client):
        r = client.get("/merchants?city=Chicago&limit=20")
        assert r.status_code == 200
        for m in r.json()["merchants"]:
            assert m["city"] == "Chicago"

    def test_merchant_by_id(self, client):
        # Get first merchant id
        merchants = client.get("/merchants?limit=1").json()["merchants"]
        mid = merchants[0]["merchant_id"]
        r = client.get(f"/merchants/{mid}")
        assert r.status_code == 200
        assert r.json()["merchant_id"] == mid

    def test_merchant_not_found(self, client):
        r = client.get("/merchants/nonexistent_id_xyz")
        assert r.status_code == 404

    def test_search_endpoint(self, client):
        r = client.get("/search?q=pizza&top_k=3")
        assert r.status_code == 200
        body = r.json()
        assert "results" in body
        assert body["count"] <= 3

    def test_query_no_llm_key(self, client):
        """Query should return retrieval-only answer when no LLM is configured."""
        r = client.post("/query", json={
            "query": "show me coffee shops in Chicago",
            "top_k": 3,
        })
        assert r.status_code == 200
        body = r.json()
        assert "answer" in body
        assert "sources" in body
        assert "latency_ms" in body

    def test_ingest_endpoint(self, client):
        r = client.post("/ingest", json={
            "records": [{
                "business_id": "test_new_001",
                "name": "New Test Cafe",
                "categories": "Coffee & Tea, Cafes",
                "city": "Chicago",
                "state": "IL",
                "latitude": 41.88,
                "longitude": -87.63,
                "stars": 4.3,
                "review_count": 500,
                "is_open": 1,
                "review_velocity_30d": 25,
                "attributes": {"RestaurantsPriceRange2": 2},
            }],
            "reindex": True,
        })
        assert r.status_code == 201
        body = r.json()
        assert body["accepted"] == 1
        assert body["failed"] == 0

    def test_analytics_review_velocity(self, client):
        r = client.post("/analytics?metric=review_velocity&group_by=mcc_code")
        assert r.status_code == 200
        body = r.json()
        assert len(body["data"]) > 0
        assert all("avg_velocity" in d for d in body["data"])

    def test_analytics_acceptance_methods(self, client):
        r = client.post("/analytics?metric=acceptance_methods")
        assert r.status_code == 200
        body = r.json()
        methods = {d["method"] for d in body["data"]}
        assert len(methods) > 0

    def test_analytics_summary(self, client):
        r = client.get("/analytics/summary")
        assert r.status_code == 200
        body = r.json()
        assert body["total_merchants"] > 0
        assert "top_mcc_categories" in body
        assert "acceptance_distribution" in body

    def test_agents_status_no_runs(self, client):
        r = client.get("/agents/status")
        assert r.status_code == 200

    def test_pipeline_stats(self, client):
        r = client.get("/pipeline/stats")
        assert r.status_code == 200
        body = r.json()
        assert "pipeline_status" in body

    def test_memory_endpoints(self, client):
        session_id = "test_session_abc"
        r = client.get(f"/memory/{session_id}")
        assert r.status_code == 200
        r = client.delete(f"/memory/{session_id}")
        assert r.status_code == 200

    def test_ge_report(self, client):
        r = client.get("/ge-report")
        assert r.status_code == 200
        body = r.json()
        assert "passed" in body
        assert "report" in body

    def test_observability_metrics(self, client):
        r = client.get("/observability/metrics")
        assert r.status_code == 200
