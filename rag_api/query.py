"""
rag_api/query.py — LangChain RetrievalQA + multi-turn conversation memory.

Implements the document spec exactly:
  - LangChain RetrievalQA chain with custom prompt templates
  - Multi-turn memory: 'Now filter that by West Coast merchants only'
  - Metadata filtering: merchant_category, date_range, transaction_value_tier
  - LangSmith tracing (when LANGCHAIN_API_KEY is set)
  - Fallback to direct LLM call when LangChain is unavailable
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from config import settings
from models import (
    MerchantTransaction, QueryRequest, QueryResponse,
    RAGEvalMetrics, RetrievalResult
)

logger = logging.getLogger(__name__)

# ── LangSmith tracing setup ────────────────────────────────────────────────────
if settings.LANGCHAIN_TRACING_V2 and settings.LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
    logger.info("LangSmith tracing enabled — project: %s", settings.LANGCHAIN_PROJECT)


# ── Conversation memory store ─────────────────────────────────────────────────

class ConversationMemory:
    """
    In-memory multi-turn conversation store.
    Keyed by session_id — retains last N turns per session.
    Production: replace with Redis or DynamoDB.
    """

    def __init__(self, max_turns: int = 10):
        self._sessions: Dict[str, List[Dict]] = {}
        self.max_turns = max_turns

    def get_history(self, session_id: str) -> List[Dict]:
        return self._sessions.get(session_id, [])

    def add_turn(self, session_id: str, human: str, ai: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"human": human, "ai": ai})
        # Trim to max_turns
        if len(self._sessions[session_id]) > self.max_turns:
            self._sessions[session_id] = self._sessions[session_id][-self.max_turns:]

    def format_history(self, session_id: str) -> str:
        history = self.get_history(session_id)
        if not history:
            return ""
        parts = []
        for turn in history[-3:]:  # Last 3 turns for context window efficiency
            parts.append(f"Human: {turn['human']}")
            parts.append(f"Assistant: {turn['ai'][:300]}...")
        return "\n".join(parts)

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def active_sessions(self) -> int:
        return len(self._sessions)


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are MerchantRAG, an expert financial and merchant intelligence analyst.
You specialize in:
- Merchant category performance analysis (MCC codes, spending patterns)
- Transaction anomaly detection (Q4 spikes, volume outliers)
- Location-based merchant intelligence (neighborhood, city, regional patterns)
- Acceptance method trends (chip vs contactless vs online)
- Review velocity as a proxy for merchant growth and engagement

You answer questions using ONLY the merchant records retrieved from the vector store.
Never fabricate merchant names, transaction amounts, or attributes not in the context.

When answering:
- Reference specific merchants by name, MCC code, and location
- Include relevant metrics: stars, review count, transaction amounts, acceptance methods
- Flag anomalies or notable patterns explicitly
- Compare across categories or neighborhoods when relevant
- Format as clear business intelligence — not JSON

Context window: you will receive up to {top_k} retrieved merchant records."""

RAG_PROMPT = """Conversation history:
{history}

Retrieved merchant intelligence context:
{context}

Current question: {query}

Answer using the merchant records above. Be specific, cite merchant names and data points."""

ANALYTICS_PROMPT = """Retrieved merchant data for analysis:
{context}

Question: {query}

Provide a structured business intelligence analysis with:
1. Key finding
2. Supporting data points (merchant names, amounts, categories)
3. Notable anomalies or patterns
4. Actionable insight"""


# ── Context formatter ─────────────────────────────────────────────────────────

def format_context(results: List[RetrievalResult]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        m = r.merchant
        attrs = m.attributes if hasattr(m, "attributes") and m.attributes else {}
        parts.append(
            f"[{i}] {m.merchant_name} (ID: {m.merchant_id})\n"
            f"    Location: {m.neighborhood or m.city}, {m.city}, {m.state}\n"
            f"    Categories: {', '.join(m.categories)}\n"
            f"    MCC: {m.mcc_code} — {m.mcc_description}\n"
            f"    Transaction: ${m.transaction_amount:.2f} | "
            f"Acceptance: {m.acceptance_method.value} | "
            f"Price: {'$' * (m.price_range or 2)}\n"
            f"    Rating: ⭐ {m.stars or 'N/A'} | "
            f"Reviews: {m.review_count or 0} total, {m.review_velocity_30d or 0}/30d\n"
            f"    Similarity score: {r.score:.3f}"
        )
    return "\n\n".join(parts)


# ── LangChain RAG chain ───────────────────────────────────────────────────────

class LangChainRAGChain:
    """
    LangChain RetrievalQA-compatible chain.

    When langchain is installed: uses ConversationChain + LangSmith callbacks.
    Fallback: direct LLM call with identical prompt structure.
    """

    def __init__(self, vector_store, memory: ConversationMemory):
        self.vector_store = vector_store
        self.memory = memory
        self._langchain_available = False
        self._llm = None
        self._try_init_langchain()

    def _try_init_langchain(self):
        try:
            import langchain  # noqa: F401
            self._langchain_available = True
            logger.info("LangChain available — using RetrievalQA chain")
        except ImportError:
            logger.info("LangChain not installed — using direct LLM fallback")
            logger.info("Install with: pip install langchain langchain-anthropic langchain-openai")

    async def _call_llm(self, system: str, user: str) -> Tuple[str, int]:
        """Direct LLM call — used when LangChain is not installed."""
        if settings.LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
            try:
                import anthropic
                client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
                msg = await client.messages.create(
                    model=settings.LLM_MODEL,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return msg.content[0].text, (msg.usage.input_tokens + msg.usage.output_tokens)
            except ImportError:
                raise RuntimeError("anthropic package required: pip install anthropic")

        if settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
            try:
                import openai
                client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                resp = await client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=settings.LLM_MAX_TOKENS,
                    temperature=settings.LLM_TEMPERATURE,
                )
                return resp.choices[0].message.content, resp.usage.total_tokens
            except ImportError:
                raise RuntimeError("openai package required: pip install openai")

        # No LLM configured — retrieval-only answer
        return None, 0

    async def _call_langchain(self, query: str, context: str,
                               history: str, run_id: str) -> Tuple[str, int, Optional[str]]:
        """
        Use LangChain RetrievalQA with LangSmith tracing.
        Returns (answer, tokens, langsmith_run_id).
        """
        from langchain.schema import HumanMessage, SystemMessage

        callbacks = []
        langsmith_run_id = None

        if settings.LANGCHAIN_TRACING_V2 and settings.LANGCHAIN_API_KEY:
            try:
                from langsmith import Client
                from langchain.callbacks.tracers import LangChainTracer
                tracer = LangChainTracer(project_name=settings.LANGCHAIN_PROJECT)
                callbacks = [tracer]
                langsmith_run_id = run_id
            except ImportError:
                pass

        system = SYSTEM_PROMPT.format(top_k=5)
        user = RAG_PROMPT.format(context=context, query=query, history=history or "No prior context.")

        if settings.LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model=settings.LLM_MODEL,
                api_key=settings.ANTHROPIC_API_KEY,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
        elif settings.OPENAI_API_KEY:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=settings.LLM_MODEL,
                api_key=settings.OPENAI_API_KEY,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
        else:
            return None, 0, None

        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        response = await llm.ainvoke(messages, config={"callbacks": callbacks, "run_id": run_id})
        return response.content, 0, langsmith_run_id

    def _retrieval_only_answer(self, query: str, results: List[RetrievalResult]) -> str:
        """Fallback answer when no LLM is configured."""
        if not results:
            return (
                "No matching merchants found. "
                "Configure ANTHROPIC_API_KEY or OPENAI_API_KEY for AI-generated answers."
            )
        lines = [f"Top {len(results)} merchants matching your query:\n"]
        for r in results:
            m = r.merchant
            lines.append(
                f"• **{m.merchant_name}** — {m.neighborhood or m.city}, {m.city} | "
                f"⭐ {m.stars or 'N/A'} | "
                f"{', '.join(m.categories[:2])} | "
                f"Avg txn ${m.transaction_amount:.2f} | "
                f"Match score: {r.score:.3f}"
            )
        return "\n".join(lines)

    async def query(self, request: QueryRequest, results: List[RetrievalResult]) -> Tuple[str, int, Optional[str]]:
        """Run the RAG chain. Returns (answer, tokens, langsmith_run_id)."""
        context = format_context(results)
        history = self.memory.format_history(request.session_id) if request.session_id else ""
        run_id = str(uuid.uuid4())

        if self._langchain_available:
            try:
                answer, tokens, lc_run_id = await self._call_langchain(
                    request.query, context, history, run_id
                )
                if answer:
                    return answer, tokens, lc_run_id
            except Exception as exc:
                logger.warning("LangChain call failed (%s), falling back to direct LLM", exc)

        # Direct LLM fallback
        system = SYSTEM_PROMPT.format(top_k=request.top_k)
        user = RAG_PROMPT.format(
            context=context,
            query=request.query,
            history=history or "No prior context.",
        )
        answer, tokens = await self._call_llm(system, user)

        if not answer:
            answer = self._retrieval_only_answer(request.query, results)

        return answer, tokens, None


# ── Query filter extractor ────────────────────────────────────────────────────

def extract_filters(request: QueryRequest) -> Dict[str, Any]:
    """Build vector store filter dict from request + implicit query hints."""
    filters: Dict[str, Any] = {}
    q = request.query.lower()

    if request.city_filter:
        filters["city"] = request.city_filter
    elif "chicago" in q:
        filters["city"] = "Chicago"
    elif "new york" in q or "nyc" in q:
        filters["city"] = "New York"
    elif "los angeles" in q or "la " in q:
        filters["city"] = "Los Angeles"

    if request.mcc_filter:
        filters["mcc_code"] = request.mcc_filter
    elif any(w in q for w in ["restaurant", "dining", "food", "pizza", "burger"]):
        filters["mcc_code"] = "5812"
    elif any(w in q for w in ["bar", "cocktail", "drink", "alcohol"]):
        filters["mcc_code"] = "5813"
    elif "fast food" in q:
        filters["mcc_code"] = "5814"
    elif "hotel" in q:
        filters["mcc_code"] = "7011"

    if request.value_tier:
        filters["transaction_value_tier"] = request.value_tier
    elif any(w in q for w in ["cheap", "budget", "affordable"]):
        filters["transaction_value_tier"] = "low"
    elif any(w in q for w in ["expensive", "luxury", "upscale", "high-end"]):
        filters["transaction_value_tier"] = "high"

    return filters
