"""
models.py — Pydantic schemas for MerchantRAG.

Merchant schema as specified in playbook:
  merchant_id, transaction_amount, MCC code, location,
  timestamp, acceptance_method

Also covers: Kafka events, RAG request/response,
LangGraph agent state, observability metrics.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


# ── Merchant / Transaction schema ────────────────────────────────────────────

class AcceptanceMethod(str, Enum):
    swipe      = "swipe"
    chip       = "chip"
    contactless = "contactless"
    online     = "online"
    keyed      = "keyed"
    mobile_pay = "mobile_pay"


class MerchantTransaction(BaseModel):
    """Core merchant transaction event — mirrors Yelp→merchant mapping."""
    transaction_id:    str   = Field(default_factory=lambda: str(uuid.uuid4()))
    merchant_id:       str
    merchant_name:     str
    transaction_amount: float = Field(..., ge=0)
    mcc_code:          str   = Field(..., description="Merchant Category Code (ISO 18245)")
    mcc_description:   str
    city:              str
    state:             str
    neighborhood:      Optional[str] = None
    latitude:          float
    longitude:         float
    timestamp:         datetime      = Field(default_factory=datetime.utcnow)
    acceptance_method: AcceptanceMethod = AcceptanceMethod.chip
    stars:             Optional[float] = None   # Yelp rating mapped in
    review_count:      Optional[int]   = None
    review_velocity_30d: Optional[int] = None
    categories:        List[str]       = Field(default_factory=list)
    is_open:           bool            = True
    price_range:       Optional[int]   = None   # 1-4

    @validator("mcc_code")
    def pad_mcc(cls, v):
        return str(v).zfill(4)

    def to_document_text(self) -> str:
        """Rich text for embedding — merges all queryable fields."""
        cats = ", ".join(self.categories)
        return (
            f"Merchant: {self.merchant_name} | "
            f"ID: {self.merchant_id} | "
            f"Location: {self.neighborhood or self.city}, {self.city}, {self.state} | "
            f"Categories: {cats} | "
            f"MCC: {self.mcc_code} ({self.mcc_description}) | "
            f"Stars: {self.stars} | Reviews: {self.review_count} | "
            f"Velocity (30d): {self.review_velocity_30d} | "
            f"Amount: ${self.transaction_amount:.2f} | "
            f"Acceptance: {self.acceptance_method.value} | "
            f"Price tier: {'$' * (self.price_range or 2)} | "
            f"Open: {'yes' if self.is_open else 'no'}"
        )


class MerchantSummary(BaseModel):
    """Aggregated merchant summary — used as embedding unit."""
    merchant_id:      str
    merchant_name:    str
    city:             str
    state:            str
    neighborhood:     Optional[str]
    mcc_code:         str
    mcc_description:  str
    avg_transaction:  float
    total_transactions: int
    avg_stars:        float
    total_reviews:    int
    review_velocity:  int
    categories:       List[str]
    acceptance_methods: List[str]
    price_range:      int
    is_open:          bool
    lat:              float
    lon:              float


# ── Kafka events ─────────────────────────────────────────────────────────────

class KafkaEvent(BaseModel):
    event_id:    str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type:  str = "merchant_transaction"
    source:      str = "merchantrag_pipeline"
    schema_version: str = "1.0"
    payload:     Dict[str, Any]
    timestamp_ms: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp() * 1000))


class PipelineStats(BaseModel):
    records_produced:  int = 0
    records_consumed:  int = 0
    records_transformed: int = 0
    records_embedded:  int = 0
    records_failed:    int = 0
    pipeline_status:   str = "idle"
    last_run_at:       Optional[str] = None
    throughput_rps:    float = 0.0


# ── Embedding ─────────────────────────────────────────────────────────────────

class ChunkSize(int, Enum):
    small  = 256
    medium = 512
    large  = 1024


class VectorDocument(BaseModel):
    doc_id:    str
    text:      str
    embedding: Optional[List[float]] = None
    metadata:  Dict[str, Any] = Field(default_factory=dict)
    chunk_size: int = 512


class RetrievalResult(BaseModel):
    merchant:  MerchantTransaction
    score:     float
    chunk_text: str
    highlights: Optional[List[str]] = None


# ── RAG API ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:         str    = Field(..., min_length=3, max_length=500)
    top_k:         int    = Field(default=5, ge=1, le=20)
    session_id:    Optional[str]  = None   # for multi-turn memory
    city_filter:   Optional[str]  = None
    mcc_filter:    Optional[str]  = None
    date_from:     Optional[str]  = None
    date_to:       Optional[str]  = None
    value_tier:    Optional[str]  = None   # low | mid | high
    stream:        bool = False

    class Config:
        schema_extra = {"example": {
            "query": "Which merchant categories had anomalous spending in Q4?",
            "top_k": 5,
            "session_id": "sess_abc123",
        }}


class QueryResponse(BaseModel):
    query:           str
    answer:          str
    sources:         List[RetrievalResult]
    session_id:      Optional[str]
    latency_ms:      float
    tokens_used:     Optional[int]
    retrieval_scores: Optional[Dict[str, float]] = None   # precision@k, recall@k
    langsmith_run_id: Optional[str] = None


class IngestRequest(BaseModel):
    records:      List[Dict[str, Any]] = Field(..., min_items=1)
    reindex:      bool = True
    source:       str = "api_upload"


class IngestResponse(BaseModel):
    accepted:     int
    failed:       int
    reindexed:    bool
    vector_store_size: int


# ── LangGraph agent state ─────────────────────────────────────────────────────

class AgentState(BaseModel):
    """Shared state passed between LangGraph nodes."""
    task:            str
    merchant_data:   Optional[List[Dict]] = None
    validation_result: Optional[Dict]    = None
    anomalies:       Optional[List[Dict]] = None
    freshness_report: Optional[Dict]     = None
    final_answer:    Optional[str]       = None
    errors:          List[str]           = Field(default_factory=list)
    steps:           List[str]           = Field(default_factory=list)


# ── Observability ─────────────────────────────────────────────────────────────

class RAGEvalMetrics(BaseModel):
    run_id:          str
    query:           str
    precision_at_k:  float
    recall_at_k:     float
    answer_faithfulness: float
    context_relevance:   float
    latency_ms:      float
    timestamp:       datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status:          str
    version:         str
    vector_store_size: int
    pipeline_stats:  PipelineStats
    services:        Dict[str, str]
