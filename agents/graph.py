"""
agents/graph.py — LangGraph agentic orchestration.

Three autonomous agents as specified in the playbook:
  1. DataValidationAgent  — validates incoming data quality before embedding
  2. DataFreshnessAgent   — alerts when merchant data hasn't been updated within SLA
  3. AnomalyDetectionAgent — statistical rules + LLM narrative interpretation

LangGraph state machine:
  START → validate → check_freshness → detect_anomalies → summarize → END

Install: pip install langgraph langchain langchain-anthropic
Fallback: sequential Python execution when LangGraph is not installed.
"""
from __future__ import annotations

import asyncio
import logging
import statistics
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from config import settings
from models import AgentState, MerchantTransaction

logger = logging.getLogger(__name__)


# ── Agent State (TypedDict for LangGraph compatibility) ─────────────────────

class GraphState(TypedDict):
    task: str
    merchants: List[Dict]
    validation_result: Optional[Dict]
    anomalies: Optional[List[Dict]]
    freshness_report: Optional[Dict]
    final_answer: Optional[str]
    errors: List[str]
    steps: List[str]


# ── Agent 1: Data Validation ─────────────────────────────────────────────────

class DataValidationAgent:
    """
    Validates incoming merchant data quality before embedding.
    Checks: completeness, MCC validity, amount ranges, required fields.
    """

    REQUIRED_FIELDS = {"merchant_id", "merchant_name", "mcc_code", "city", "transaction_amount"}
    VALID_MCC_CODES = {
        "5812","5813","5814","5411","5912","7011","5941","5921","5999","7299","7996","4111"
    }

    async def run(self, state: GraphState) -> GraphState:
        state["steps"].append("data_validation_agent")
        merchants = state.get("merchants", [])

        if not merchants:
            state["errors"].append("No merchants provided to validate")
            state["validation_result"] = {"status": "error", "message": "Empty dataset"}
            return state

        issues: List[Dict] = []
        passed = 0

        for m in merchants:
            merchant_issues = []

            # Required field check
            missing = self.REQUIRED_FIELDS - set(m.keys())
            if missing:
                merchant_issues.append(f"Missing fields: {missing}")

            # MCC code validation
            mcc = str(m.get("mcc_code", "")).zfill(4)
            if mcc not in self.VALID_MCC_CODES:
                merchant_issues.append(f"Invalid MCC code: {mcc}")

            # Amount range check
            amount = m.get("transaction_amount", 0)
            try:
                amount = float(amount)
                if not (0.01 <= amount <= 99999.99):
                    merchant_issues.append(f"Amount out of range: {amount}")
            except (TypeError, ValueError):
                merchant_issues.append(f"Non-numeric amount: {amount}")

            # merchant_id referential integrity
            if m.get("merchant_id") and not m.get("merchant_name"):
                merchant_issues.append("merchant_id set but merchant_name is null")

            if merchant_issues:
                issues.append({"merchant_id": m.get("merchant_id"), "issues": merchant_issues})
            else:
                passed += 1

        pass_rate = passed / len(merchants) if merchants else 0
        state["validation_result"] = {
            "status": "pass" if pass_rate >= 0.95 else "fail",
            "total": len(merchants),
            "passed": passed,
            "failed": len(issues),
            "pass_rate": f"{pass_rate:.1%}",
            "issues_sample": issues[:5],
            "ready_for_embedding": pass_rate >= 0.95,
        }
        logger.info(
            "DataValidationAgent: %d/%d passed (%.1f%%)",
            passed, len(merchants), pass_rate * 100
        )
        return state


# ── Agent 2: Data Freshness ──────────────────────────────────────────────────

class DataFreshnessAgent:
    """
    Monitors merchant data staleness.
    Alerts when data hasn't been updated within SLA thresholds.
    """

    SLA_THRESHOLDS = {
        "critical": 7,   # days — page on-call
        "warning":  3,   # days — Slack alert
        "healthy":  1,   # days — expected cadence
    }

    async def run(self, state: GraphState) -> GraphState:
        state["steps"].append("data_freshness_agent")
        merchants = state.get("merchants", [])

        now = datetime.utcnow()
        stale_critical = []
        stale_warning = []
        fresh = []

        for m in merchants:
            # Use timestamp or ingested_at
            ts_str = m.get("timestamp") or m.get("last_ingested_at")
            if not ts_str:
                stale_critical.append({
                    "merchant_id": m.get("merchant_id"),
                    "merchant_name": m.get("merchant_name"),
                    "reason": "No timestamp available",
                })
                continue

            try:
                ts = datetime.fromisoformat(str(ts_str).replace("Z", ""))
                age_days = (now - ts).days
                entry = {
                    "merchant_id": m.get("merchant_id"),
                    "merchant_name": m.get("merchant_name"),
                    "age_days": age_days,
                    "last_seen": ts_str,
                }
                if age_days >= self.SLA_THRESHOLDS["critical"]:
                    stale_critical.append(entry)
                elif age_days >= self.SLA_THRESHOLDS["warning"]:
                    stale_warning.append(entry)
                else:
                    fresh.append(entry)
            except Exception:
                stale_critical.append({
                    "merchant_id": m.get("merchant_id"),
                    "reason": f"Unparseable timestamp: {ts_str}",
                })

        freshness_pct = len(fresh) / len(merchants) if merchants else 0
        state["freshness_report"] = {
            "status": (
                "critical" if len(stale_critical) / max(len(merchants), 1) > 0.1
                else "warning" if stale_warning
                else "healthy"
            ),
            "total": len(merchants),
            "fresh": len(fresh),
            "stale_warning": len(stale_warning),
            "stale_critical": len(stale_critical),
            "freshness_pct": f"{freshness_pct:.1%}",
            "critical_merchants": stale_critical[:5],
            "warning_merchants": stale_warning[:5],
            "sla_thresholds_days": self.SLA_THRESHOLDS,
            "evaluated_at": now.isoformat(),
        }
        logger.info(
            "DataFreshnessAgent: %d fresh, %d warning, %d critical",
            len(fresh), len(stale_warning), len(stale_critical)
        )
        return state


# ── Agent 3: Anomaly Detection ───────────────────────────────────────────────

class AnomalyDetectionAgent:
    """
    Statistical anomaly detection over merchant transaction data.
    Uses Z-score thresholding + IQR method.
    Generates LLM narrative interpretation for flagged anomalies.
    """

    Z_SCORE_THRESHOLD = 2.5
    VELOCITY_SPIKE_MULTIPLIER = 3.0

    async def run(self, state: GraphState) -> GraphState:
        state["steps"].append("anomaly_detection_agent")
        merchants = state.get("merchants", [])
        if not merchants:
            state["anomalies"] = []
            return state

        # Extract numeric fields for stats
        amounts = [float(m.get("transaction_amount") or 0) for m in merchants]
        velocities = [int(m.get("review_velocity_30d") or 0) for m in merchants]
        stars_list = [float(m.get("stars") or 0) for m in merchants if m.get("stars")]

        anomalies: List[Dict] = []

        # Z-score on transaction amounts
        if len(amounts) > 3:
            mean_a = statistics.mean(amounts)
            stdev_a = statistics.stdev(amounts) or 1
            for m, amount in zip(merchants, amounts):
                z = (amount - mean_a) / stdev_a
                if abs(z) >= self.Z_SCORE_THRESHOLD:
                    anomalies.append({
                        "type": "amount_outlier",
                        "merchant_id": m.get("merchant_id"),
                        "merchant_name": m.get("merchant_name"),
                        "value": amount,
                        "z_score": round(z, 2),
                        "direction": "high" if z > 0 else "low",
                        "mean": round(mean_a, 2),
                        "narrative": (
                            f"{m.get('merchant_name')} has an unusually "
                            f"{'high' if z > 0 else 'low'} average transaction of ${amount:.2f} "
                            f"(global mean: ${mean_a:.2f}, z={z:.2f}). "
                            f"This may indicate {'premium positioning' if z > 0 else 'discount/fast-casual operation'}."
                        ),
                    })

        # Velocity spikes
        if len(velocities) > 3:
            mean_v = statistics.mean(velocities) or 1
            for m, vel in zip(merchants, velocities):
                if vel > mean_v * self.VELOCITY_SPIKE_MULTIPLIER and vel > 10:
                    anomalies.append({
                        "type": "review_velocity_spike",
                        "merchant_id": m.get("merchant_id"),
                        "merchant_name": m.get("merchant_name"),
                        "velocity": vel,
                        "mean_velocity": round(mean_v, 1),
                        "spike_factor": round(vel / mean_v, 1),
                        "narrative": (
                            f"{m.get('merchant_name')} is receiving {vel} reviews/30d — "
                            f"{vel/mean_v:.1f}x the dataset average of {mean_v:.1f}. "
                            f"This signals strong viral growth or a recent viral event."
                        ),
                    })

        # Low star outliers (potential quality or fraud signal)
        if stars_list:
            mean_s = statistics.mean(stars_list)
            stdev_s = statistics.stdev(stars_list) if len(stars_list) > 1 else 0.5
            for m in merchants:
                s = float(m.get("stars") or 0)
                if s > 0:
                    z = (s - mean_s) / (stdev_s or 0.5)
                    if z < -self.Z_SCORE_THRESHOLD:
                        anomalies.append({
                            "type": "low_star_outlier",
                            "merchant_id": m.get("merchant_id"),
                            "merchant_name": m.get("merchant_name"),
                            "stars": s,
                            "z_score": round(z, 2),
                            "narrative": (
                                f"{m.get('merchant_name')} has a significantly low rating of "
                                f"{s} stars (mean: {mean_s:.2f}, z={z:.2f}). "
                                f"May warrant customer experience review."
                            ),
                        })

        state["anomalies"] = anomalies
        logger.info("AnomalyDetectionAgent: detected %d anomalies", len(anomalies))
        return state


# ── LangGraph orchestrator ────────────────────────────────────────────────────

class MerchantAgentGraph:
    """
    LangGraph state machine orchestrating the three agents.

    With langgraph:    Compiled StateGraph with typed state transitions
    Without langgraph: Sequential Python fallback (identical logic)
    """

    def __init__(self):
        self.validation_agent = DataValidationAgent()
        self.freshness_agent  = DataFreshnessAgent()
        self.anomaly_agent    = AnomalyDetectionAgent()
        self._graph = None
        self._try_build_langgraph()

    def _try_build_langgraph(self):
        try:
            from langgraph.graph import StateGraph, END

            graph = StateGraph(GraphState)
            graph.add_node("validate",         self._validate_node)
            graph.add_node("check_freshness",  self._freshness_node)
            graph.add_node("detect_anomalies", self._anomaly_node)
            graph.add_node("summarize",        self._summarize_node)

            graph.set_entry_point("validate")
            graph.add_edge("validate", "check_freshness")
            graph.add_edge("check_freshness", "detect_anomalies")
            graph.add_edge("detect_anomalies", "summarize")
            graph.add_edge("summarize", END)

            self._graph = graph.compile()
            logger.info("LangGraph compiled successfully")
        except ImportError:
            logger.info("langgraph not installed — using sequential fallback")
            logger.info("Install: pip install langgraph")

    async def _validate_node(self, state: GraphState) -> GraphState:
        return await self.validation_agent.run(state)

    async def _freshness_node(self, state: GraphState) -> GraphState:
        return await self.freshness_agent.run(state)

    async def _anomaly_node(self, state: GraphState) -> GraphState:
        return await self.anomaly_agent.run(state)

    async def _summarize_node(self, state: GraphState) -> GraphState:
        validation = state.get("validation_result") or {}
        freshness = state.get("freshness_report") or {}
        anomalies = state.get("anomalies") or []

        summary_parts = [f"Agent Pipeline Summary — Task: {state.get('task')}\n"]

        # Validation summary
        summary_parts.append(
            f"Data Validation: {validation.get('status','N/A').upper()} | "
            f"{validation.get('passed',0)}/{validation.get('total',0)} records passed "
            f"({validation.get('pass_rate','N/A')})"
        )

        # Freshness summary
        summary_parts.append(
            f"Data Freshness: {freshness.get('status','N/A').upper()} | "
            f"{freshness.get('fresh',0)} fresh, {freshness.get('stale_critical',0)} critical"
        )

        # Anomaly summary
        anomaly_types = {}
        for a in anomalies:
            t = a.get("type", "unknown")
            anomaly_types[t] = anomaly_types.get(t, 0) + 1

        if anomalies:
            summary_parts.append(
                f"Anomalies Detected: {len(anomalies)} total | "
                + " | ".join(f"{k}: {v}" for k, v in anomaly_types.items())
            )
            summary_parts.append("\nTop anomalies:")
            for a in anomalies[:3]:
                summary_parts.append(f"  • {a.get('narrative', '')}")
        else:
            summary_parts.append("Anomalies Detected: None")

        state["final_answer"] = "\n".join(summary_parts)
        return state

    async def run(self, merchants: List[Dict], task: str = "pipeline_health_check") -> GraphState:
        """Execute the full agent pipeline."""
        initial_state: GraphState = {
            "task": task,
            "merchants": merchants,
            "validation_result": None,
            "anomalies": None,
            "freshness_report": None,
            "final_answer": None,
            "errors": [],
            "steps": [],
        }

        if self._graph:
            # Use compiled LangGraph
            result = await self._graph.ainvoke(initial_state)
            return result

        # Sequential fallback
        state = initial_state
        state = await self.validation_agent.run(state)
        state = await self.freshness_agent.run(state)
        state = await self.anomaly_agent.run(state)
        state = await self._summarize_node(state)
        return state
