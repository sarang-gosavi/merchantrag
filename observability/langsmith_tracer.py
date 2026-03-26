"""
observability/langsmith_tracer.py — LangSmith tracing + RAGAS evaluation.

Tracks every RAG query:
  - Latency, retrieval quality, LLM response scores
  - precision@k, answer faithfulness, context relevance (RAGAS framework)
  - Custom eval metrics for merchant intelligence domain

LangSmith setup:
  1. Get API key from smith.langchain.com
  2. Set LANGCHAIN_API_KEY + LANGCHAIN_TRACING_V2=true in .env

RAGAS setup:
  pip install ragas
  Uses OpenAI for evaluation LLM (or anthropic via ragas adapter)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings
from models import RAGEvalMetrics, RetrievalResult

logger = logging.getLogger(__name__)


# ── LangSmith tracer ───────────────────────────────────────────────────────────

class LangSmithTracer:
    """
    Wraps LangSmith tracing for RAG query observability.
    Falls back to local JSONL logging when LangSmith is not configured.
    """

    def __init__(self):
        self._client = None
        self._log_path = Path("./data/query_traces.jsonl")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._try_init()

    def _try_init(self):
        if not (settings.LANGCHAIN_TRACING_V2 and settings.LANGCHAIN_API_KEY):
            logger.info("LangSmith not configured — using local trace log")
            return
        try:
            from langsmith import Client
            self._client = Client(
                api_url=settings.LANGCHAIN_ENDPOINT,
                api_key=settings.LANGCHAIN_API_KEY,
            )
            logger.info("LangSmith client connected — project: %s", settings.LANGCHAIN_PROJECT)
        except ImportError:
            logger.info("langsmith package not installed: pip install langsmith")

    def log_query(
        self,
        run_id: str,
        query: str,
        answer: str,
        sources: List[RetrievalResult],
        latency_ms: float,
        tokens_used: int,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Log a RAG query run to LangSmith or local file."""
        trace = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "answer_preview": answer[:200],
            "num_sources": len(sources),
            "top_score": sources[0].score if sources else 0,
            "latency_ms": latency_ms,
            "tokens_used": tokens_used,
            "metadata": metadata or {},
        }

        if self._client:
            try:
                self._client.create_run(
                    name="merchantrag_query",
                    run_type="chain",
                    inputs={"query": query, "top_k": len(sources)},
                    outputs={"answer": answer, "num_sources": len(sources)},
                    extra={"latency_ms": latency_ms, "tokens": tokens_used},
                    project_name=settings.LANGCHAIN_PROJECT,
                )
            except Exception as exc:
                logger.warning("LangSmith log failed: %s", exc)

        # Always write to local log
        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace) + "\n")

    def log_feedback(self, run_id: str, score: float, comment: str = "") -> None:
        """Log user feedback score (0.0 to 1.0) back to LangSmith."""
        if self._client:
            try:
                self._client.create_feedback(
                    run_id=run_id,
                    key="user_score",
                    score=score,
                    comment=comment,
                )
            except Exception:
                pass

    @property
    def is_active(self) -> bool:
        return self._client is not None


# ── RAGAS evaluator ───────────────────────────────────────────────────────────

class RAGASEvaluator:
    """
    RAGAS evaluation framework integration.
    Computes: precision@k, answer faithfulness, context relevance.

    With ragas installed:  Uses full RAGAS pipeline
    Without ragas:         Local heuristic approximations
    """

    def __init__(self):
        self._ragas_available = False
        try:
            import ragas  # noqa: F401
            self._ragas_available = True
            logger.info("RAGAS framework available")
        except ImportError:
            logger.info("ragas not installed — using heuristic metrics")
            logger.info("Install: pip install ragas")

    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute RAGAS metrics for a single RAG run."""
        if self._ragas_available:
            return await self._ragas_evaluate(query, answer, contexts, ground_truth)
        return self._heuristic_evaluate(query, answer, contexts)

    async def _ragas_evaluate(self, query, answer, contexts, ground_truth) -> Dict[str, float]:
        """Full RAGAS evaluation pipeline."""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset

            data = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]

            ds = Dataset.from_dict(data)
            metrics = [faithfulness, answer_relevancy, context_precision]
            if ground_truth:
                metrics.append(context_recall)

            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: evaluate(ds, metrics=metrics)
            )
            return {
                "faithfulness":      float(result.get("faithfulness", 0)),
                "answer_relevancy":  float(result.get("answer_relevancy", 0)),
                "context_precision": float(result.get("context_precision", 0)),
                "context_recall":    float(result.get("context_recall", 0)),
            }
        except Exception as exc:
            logger.warning("RAGAS evaluation failed: %s", exc)
            return self._heuristic_evaluate(query, answer, contexts)

    def _heuristic_evaluate(self, query: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """
        Heuristic approximations for precision@k, faithfulness, context relevance.
        Used when RAGAS is not installed or evaluation LLM is unavailable.
        """
        import re

        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        a_tokens = set(re.findall(r"[a-z0-9]+", answer.lower()))

        # Precision@k: fraction of contexts that have query token overlap
        relevant_contexts = sum(
            1 for ctx in contexts
            if len(q_tokens & set(re.findall(r"[a-z0-9]+", ctx.lower()))) > 1
        )
        precision_k = relevant_contexts / len(contexts) if contexts else 0

        # Faithfulness: fraction of answer tokens that appear in any context
        all_ctx_tokens = set()
        for ctx in contexts:
            all_ctx_tokens |= set(re.findall(r"[a-z0-9]+", ctx.lower()))
        faithfulness = len(a_tokens & all_ctx_tokens) / len(a_tokens) if a_tokens else 0

        # Context relevance: mean token overlap between query and each context
        relevances = []
        for ctx in contexts:
            ctx_tokens = set(re.findall(r"[a-z0-9]+", ctx.lower()))
            overlap = len(q_tokens & ctx_tokens) / max(len(q_tokens), 1)
            relevances.append(overlap)
        context_relevance = sum(relevances) / len(relevances) if relevances else 0

        return {
            "faithfulness":         round(min(faithfulness * 1.2, 1.0), 3),
            "answer_relevancy":     round(min(context_relevance * 1.1, 1.0), 3),
            "context_precision":    round(precision_k, 3),
            "context_recall":       0.0,   # requires ground truth
            "precision_at_k":       round(precision_k, 3),
        }

    def precision_at_k(self, relevant_ids: List[str], retrieved_ids: List[str], k: int = 5) -> float:
        """Precision@k: fraction of top-k retrieved docs that are relevant."""
        relevant_set = set(relevant_ids)
        top_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
        return hits / k if k else 0.0

    def recall_at_k(self, relevant_ids: List[str], retrieved_ids: List[str], k: int = 5) -> float:
        """Recall@k: fraction of relevant docs found in top-k."""
        relevant_set = set(relevant_ids)
        if not relevant_set:
            return 0.0
        top_k = set(retrieved_ids[:k])
        return len(relevant_set & top_k) / len(relevant_set)


# ── Metrics store (local JSONL) ───────────────────────────────────────────────

class MetricsStore:
    """Persists RAGEvalMetrics to a local JSONL file for dashboard consumption."""

    def __init__(self, path: str = "./data/eval_metrics.jsonl"):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, metrics: RAGEvalMetrics) -> None:
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(metrics.json() + "\n")

    def load_all(self) -> List[Dict]:
        if not self._path.exists():
            return []
        records = []
        with open(self._path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
        return records

    def summary(self) -> Dict:
        records = self.load_all()
        if not records:
            return {"total_runs": 0}
        return {
            "total_runs": len(records),
            "avg_precision_at_k":   sum(r.get("precision_at_k", 0) for r in records) / len(records),
            "avg_faithfulness":     sum(r.get("answer_faithfulness", 0) for r in records) / len(records),
            "avg_context_relevance": sum(r.get("context_relevance", 0) for r in records) / len(records),
            "avg_latency_ms":       sum(r.get("latency_ms", 0) for r in records) / len(records),
        }
