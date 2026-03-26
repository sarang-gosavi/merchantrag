"""
embeddings/embedder.py — Chunking + embedding pipeline.

Benchmarks chunk sizes: 256, 512, 1024 tokens.
Supports:
  - OpenAI text-embedding-3-small (production)
  - sentence-transformers/all-MiniLM-L6-v2 (local, no API key needed)

Loads vectors into:
  - Pinecone (cloud, serverless, scales to 10M+ records)
  - pgvector (local Postgres, no cloud dependency)
  - Local TF-IDF fallback (zero-dep demo mode)

Metadata filtering: merchant_category, date_range, transaction_value_tier
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from config import settings
from models import MerchantTransaction, VectorDocument, RetrievalResult, ChunkSize

logger = logging.getLogger(__name__)


# ── Chunker ────────────────────────────────────────────────────────────────────

class MerchantChunker:
    """
    Splits merchant data into semantic chunks for embedding.

    Strategy: one rich-text document per merchant, chunked by
    semantic unit (identity, location, financials, operations).
    Benchmark results guide CHUNK_SIZE selection.
    """

    def chunk(self, txn: MerchantTransaction, chunk_size: int = 512) -> List[Dict]:
        """
        Return list of {text, metadata} chunks for a merchant.
        chunk_size controls approximate token budget per chunk.
        """
        chunks = []

        # Chunk 1 — Identity + location
        identity = (
            f"Merchant: {txn.merchant_name}. "
            f"ID: {txn.merchant_id}. "
            f"Location: {txn.address if hasattr(txn,'address') else ''} "
            f"{txn.neighborhood or txn.city}, {txn.city}, {txn.state}. "
            f"Categories: {', '.join(txn.categories)}. "
            f"MCC: {txn.mcc_code} — {txn.mcc_description}."
        )
        chunks.append({"text": identity[:chunk_size * 4], "type": "identity"})

        if chunk_size >= 512:
            # Chunk 2 — Financial + acceptance
            financial = (
                f"Merchant {txn.merchant_name} financial profile: "
                f"Average transaction ${txn.transaction_amount:.2f}. "
                f"Price tier: {'$' * (txn.price_range or 2)}. "
                f"Acceptance method: {txn.acceptance_method.value}. "
                f"Transaction value tier: "
                f"{'low' if txn.transaction_amount < 20 else 'mid' if txn.transaction_amount < 75 else 'high'}."
            )
            chunks.append({"text": financial, "type": "financial"})

        if chunk_size >= 512:
            # Chunk 3 — Quality + engagement
            quality = (
                f"Merchant {txn.merchant_name} customer metrics: "
                f"Star rating {txn.stars or 'N/A'}. "
                f"Total reviews: {txn.review_count or 0}. "
                f"Review velocity: {txn.review_velocity_30d or 0} reviews/30 days. "
                f"Currently {'open' if txn.is_open else 'closed'}."
            )
            chunks.append({"text": quality, "type": "quality"})

        if chunk_size >= 1024:
            # Chunk 4 — Full document for large context models
            full = txn.to_document_text()
            chunks.append({"text": full, "type": "full"})

        # Attach common metadata to all chunks
        metadata = {
            "merchant_id":   txn.merchant_id,
            "merchant_name": txn.merchant_name,
            "city":          txn.city,
            "state":         txn.state,
            "neighborhood":  txn.neighborhood or txn.city,
            "mcc_code":      txn.mcc_code,
            "mcc_description": txn.mcc_description,
            "categories":    txn.categories[:3],
            "stars":         txn.stars,
            "review_count":  txn.review_count,
            "review_velocity_30d": txn.review_velocity_30d,
            "transaction_amount": txn.transaction_amount,
            "transaction_value_tier": (
                "low" if txn.transaction_amount < 20
                else "mid" if txn.transaction_amount < 75
                else "high"
            ),
            "acceptance_method": txn.acceptance_method.value,
            "price_range":   txn.price_range,
            "is_open":       txn.is_open,
        }

        for c in chunks:
            c["metadata"] = metadata
            c["doc_id"] = hashlib.md5(
                f"{txn.merchant_id}_{c['type']}".encode()
            ).hexdigest()

        return chunks


# ── Embedding backends ────────────────────────────────────────────────────────

class OpenAIEmbedder:
    """OpenAI text-embedding-3-small — 1536-dim, best for production."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            except ImportError:
                raise RuntimeError("openai package required: pip install openai")
        return self._client

    async def embed(self, texts: List[str]) -> List[List[float]]:
        client = self._get_client()
        resp = await client.embeddings.create(
            input=texts,
            model=settings.EMBEDDING_MODEL,
        )
        return [r.embedding for r in resp.data]

    async def embed_one(self, text: str) -> List[float]:
        result = await self.embed([text])
        return result[0]


class SentenceTransformerEmbedder:
    """
    sentence-transformers/all-MiniLM-L6-v2 — 384-dim, local, no API key.
    Install: pip install sentence-transformers
    """

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)
                logger.info("Loaded SentenceTransformer: %s", settings.SENTENCE_TRANSFORMER_MODEL)
            except ImportError:
                raise RuntimeError("sentence-transformers required: pip install sentence-transformers")
        return self._model

    async def embed(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        embeddings = await asyncio.get_event_loop().run_in_executor(
            None, lambda: model.encode(texts, convert_to_numpy=True).tolist()
        )
        return embeddings

    async def embed_one(self, text: str) -> List[float]:
        result = await self.embed([text])
        return result[0]


# ── TF-IDF fallback (zero-dep) ────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class LocalTFIDFStore:
    """In-memory TF-IDF vector store — zero-dependency demo mode."""

    def __init__(self):
        self._docs: List[VectorDocument] = []
        self._txn_map: Dict[str, MerchantTransaction] = {}
        self._vectors: List[Dict[str, float]] = []
        self._idf: Dict[str, float] = {}
        self._dirty = True

    def add(self, txn: MerchantTransaction, chunks: List[Dict]) -> None:
        for chunk in chunks:
            doc = VectorDocument(
                doc_id=chunk["doc_id"],
                text=chunk["text"],
                metadata=chunk["metadata"],
                chunk_size=settings.CHUNK_SIZE,
            )
            for i, d in enumerate(self._docs):
                if d.doc_id == doc.doc_id:
                    self._docs[i] = doc
                    self._txn_map[txn.merchant_id] = txn
                    self._dirty = True
                    return
            self._docs.append(doc)
        self._txn_map[txn.merchant_id] = txn
        self._dirty = True

    def _ensure_indexed(self):
        if not self._dirty or not self._docs:
            return
        N = len(self._docs)
        df: Dict[str, int] = defaultdict(int)
        tf_docs = []
        for doc in self._docs:
            tokens = _tokenize(doc.text)
            tf: Dict[str, float] = defaultdict(float)
            for t in tokens:
                tf[t] += 1
            total = len(tokens) or 1
            for t in tf:
                tf[t] /= total
                df[t] += 1
            tf_docs.append(dict(tf))
        self._idf = {t: math.log((N + 1) / (cnt + 1)) + 1 for t, cnt in df.items()}
        self._vectors = []
        for tf in tf_docs:
            vec = {t: tf[t] * self._idf.get(t, 1) for t in tf}
            norm = math.sqrt(sum(v ** 2 for v in vec.values())) or 1
            self._vectors.append({t: v / norm for t, v in vec.items()})
        self._dirty = False

    def _query_vec(self, query: str) -> Dict[str, float]:
        tokens = _tokenize(query)
        tf: Dict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1
        total = len(tokens) or 1
        vec = {t: (tf[t] / total) * self._idf.get(t, 1) for t in tf}
        norm = math.sqrt(sum(v ** 2 for v in vec.values())) or 1
        return {t: v / norm for t, v in vec.items()}

    def _cosine(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        return sum(a.get(t, 0) * b.get(t, 0) for t in b)

    def search(
        self, query: str, top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[RetrievalResult]:
        self._ensure_indexed()
        if not self._docs:
            return []
        q_vec = self._query_vec(query)
        scored = []
        for idx, doc_vec in enumerate(self._vectors):
            doc = self._docs[idx]
            meta = doc.metadata or {}
            if filters:
                if filters.get("city") and meta.get("city", "").lower() != filters["city"].lower():
                    continue
                if filters.get("mcc_code") and meta.get("mcc_code") != filters["mcc_code"]:
                    continue
                if filters.get("transaction_value_tier") and \
                   meta.get("transaction_value_tier") != filters["transaction_value_tier"]:
                    continue
                if filters.get("min_stars") and (meta.get("stars") or 0) < filters["min_stars"]:
                    continue
                if filters.get("neighborhood") and filters["neighborhood"].lower() not in \
                   (meta.get("neighborhood") or "").lower():
                    continue
            score = self._cosine(q_vec, doc_vec)
            scored.append((score, idx))
        scored.sort(reverse=True)
        results = []
        seen_merchants = set()
        for score, idx in scored:
            doc = self._docs[idx]
            merchant_id = doc.metadata.get("merchant_id")
            if merchant_id in seen_merchants:
                continue
            txn = self._txn_map.get(merchant_id)
            if txn:
                results.append(RetrievalResult(
                    merchant=txn,
                    score=round(score, 4),
                    chunk_text=doc.text[:300],
                ))
                seen_merchants.add(merchant_id)
            if len(results) >= top_k:
                break
        return results

    @property
    def size(self) -> int:
        return len(self._docs)

    def get_all(self) -> List[MerchantTransaction]:
        return list(self._txn_map.values())

    def get_by_id(self, merchant_id: str) -> Optional[MerchantTransaction]:
        return self._txn_map.get(merchant_id)


# ── Pinecone backend ──────────────────────────────────────────────────────────

class PineconeStore:
    """Pinecone serverless vector store — scales to 10M+ records."""

    def __init__(self):
        self._index = None
        self._txn_map: Dict[str, MerchantTransaction] = {}
        self._embedder = None

    async def initialize(self, embedder):
        self._embedder = embedder
        try:
            from pinecone import Pinecone, ServerlessSpec
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            if settings.PINECONE_INDEX not in pc.list_indexes().names():
                pc.create_index(
                    name=settings.PINECONE_INDEX,
                    dimension=settings.EMBEDDING_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            self._index = pc.Index(settings.PINECONE_INDEX)
            logger.info("Pinecone index '%s' ready", settings.PINECONE_INDEX)
        except ImportError:
            raise RuntimeError("pinecone required: pip install pinecone-client")

    async def upsert(self, txn: MerchantTransaction, chunks: List[Dict]) -> None:
        vectors = []
        for chunk in chunks:
            embedding = await self._embedder.embed_one(chunk["text"])
            vectors.append({
                "id": chunk["doc_id"],
                "values": embedding,
                "metadata": {k: str(v) if isinstance(v, list) else v
                             for k, v in (chunk.get("metadata") or {}).items()},
            })
        if vectors:
            self._index.upsert(vectors=vectors)
        self._txn_map[txn.merchant_id] = txn

    async def search(self, query: str, top_k: int = 5,
                     filters: Optional[Dict] = None) -> List[RetrievalResult]:
        q_vec = await self._embedder.embed_one(query)
        pinecone_filter = {}
        if filters:
            if filters.get("city"):
                pinecone_filter["city"] = {"$eq": filters["city"]}
            if filters.get("mcc_code"):
                pinecone_filter["mcc_code"] = {"$eq": filters["mcc_code"]}
            if filters.get("transaction_value_tier"):
                pinecone_filter["transaction_value_tier"] = {"$eq": filters["transaction_value_tier"]}
        result = self._index.query(
            vector=q_vec, top_k=top_k * 2,
            filter=pinecone_filter or None,
            include_metadata=True,
        )
        seen = set()
        results = []
        for match in result.matches:
            mid = match.metadata.get("merchant_id")
            if mid in seen:
                continue
            txn = self._txn_map.get(mid)
            if txn:
                results.append(RetrievalResult(
                    merchant=txn, score=match.score, chunk_text=""))
                seen.add(mid)
            if len(results) >= top_k:
                break
        return results

    @property
    def size(self) -> int:
        if self._index:
            return self._index.describe_index_stats().total_vector_count
        return len(self._txn_map)


# ── pgvector backend ──────────────────────────────────────────────────────────

class PGVectorStore:
    """
    pgvector — local Postgres with vector similarity search.
    Excellent for benchmarking vs Pinecone on same data.
    Setup: docker run -p 5432:5432 -e POSTGRES_PASSWORD=password pgvector/pgvector:pg16
    """

    CREATE_TABLE = """
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS merchant_embeddings (
        doc_id TEXT PRIMARY KEY,
        merchant_id TEXT NOT NULL,
        chunk_type TEXT,
        chunk_text TEXT,
        embedding vector(%d),
        metadata JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_merchant_embedding
        ON merchant_embeddings USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """ % 384   # sentence-transformers dim (change to 1536 for OpenAI)

    def __init__(self):
        self._pool = None
        self._txn_map: Dict[str, MerchantTransaction] = {}
        self._embedder = None

    async def initialize(self, embedder):
        self._embedder = embedder
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(settings.PGVECTOR_URL)
            async with self._pool.acquire() as conn:
                await conn.execute(self.CREATE_TABLE)
            logger.info("pgvector store ready")
        except ImportError:
            raise RuntimeError("asyncpg required: pip install asyncpg")

    async def upsert(self, txn: MerchantTransaction, chunks: List[Dict]) -> None:
        import json as _json
        async with self._pool.acquire() as conn:
            for chunk in chunks:
                emb = await self._embedder.embed_one(chunk["text"])
                await conn.execute("""
                    INSERT INTO merchant_embeddings
                      (doc_id, merchant_id, chunk_type, chunk_text, embedding, metadata)
                    VALUES ($1,$2,$3,$4,$5::vector,$6)
                    ON CONFLICT (doc_id) DO UPDATE
                      SET embedding=$5::vector, chunk_text=$4, metadata=$6
                """, chunk["doc_id"], txn.merchant_id, chunk.get("type", "full"),
                    chunk["text"], str(emb), _json.dumps(chunk.get("metadata", {})))
        self._txn_map[txn.merchant_id] = txn

    async def search(self, query: str, top_k: int = 5,
                     filters: Optional[Dict] = None) -> List[RetrievalResult]:
        import json as _json
        q_vec = await self._embedder.embed_one(query)
        where = "1=1"
        if filters and filters.get("city"):
            where += f" AND metadata->>'city' = '{filters['city']}'"
        if filters and filters.get("mcc_code"):
            where += f" AND metadata->>'mcc_code' = '{filters['mcc_code']}'"
        sql = f"""
            SELECT merchant_id, chunk_text, 1 - (embedding <=> $1::vector) AS score, metadata
            FROM merchant_embeddings
            WHERE {where}
            ORDER BY embedding <=> $1::vector
            LIMIT {top_k * 3}
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, str(q_vec))
        seen = set()
        results = []
        for row in rows:
            mid = row["merchant_id"]
            if mid in seen:
                continue
            txn = self._txn_map.get(mid)
            if txn:
                results.append(RetrievalResult(
                    merchant=txn, score=float(row["score"]),
                    chunk_text=row["chunk_text"][:300]))
                seen.add(mid)
            if len(results) >= top_k:
                break
        return results

    @property
    def size(self) -> int:
        return len(self._txn_map)


# ── Factory ───────────────────────────────────────────────────────────────────

def create_embedder():
    """Return configured embedding backend."""
    if settings.EMBEDDING_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        logger.info("Using OpenAI embedder: %s", settings.EMBEDDING_MODEL)
        return OpenAIEmbedder()
    logger.info("Using SentenceTransformer embedder: %s", settings.SENTENCE_TRANSFORMER_MODEL)
    return SentenceTransformerEmbedder()


def create_vector_store(embedder=None):
    """Return configured vector store based on VECTOR_BACKEND setting."""
    backend = settings.VECTOR_BACKEND
    if settings.USE_LOCAL_VECTORS or backend == "local":
        logger.info("Using LocalTFIDFStore")
        return LocalTFIDFStore()
    if backend == "pinecone":
        logger.info("Using PineconeStore")
        return PineconeStore()
    if backend == "pgvector":
        logger.info("Using PGVectorStore")
        return PGVectorStore()
    raise ValueError(f"Unknown VECTOR_BACKEND: {backend}")
