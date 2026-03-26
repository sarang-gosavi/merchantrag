# AGENTS.md — LangGraph Agentic Architecture

> **The most shareable technical content in MerchantRAG.**
> This document explains the three autonomous agents, their state machine,
> decision trees, and the business rationale for each design choice.

---

## Overview

MerchantRAG implements a **LangGraph multi-agent pipeline** that autonomously
validates, monitors, and analyzes merchant data without human intervention.
The agents run on every pipeline reload and are also callable on-demand via `POST /agents/run`.

```
┌────────────────────────────────────────────────────────────────┐
│                   LangGraph State Machine                      │
│                                                                │
│   START                                                        │
│     │                                                          │
│     ▼                                                          │
│  ┌──────────────────────┐                                      │
│  │  DataValidationAgent │  ← Quality gate before embedding     │
│  │  • Required fields   │                                      │
│  │  • MCC code check    │                                      │
│  │  • Amount ranges     │                                      │
│  │  • Referential integ.│                                      │
│  └──────────┬───────────┘                                      │
│             │ GraphState passes through                        │
│             ▼                                                  │
│  ┌──────────────────────┐                                      │
│  │  DataFreshnessAgent  │  ← SLA monitoring                   │
│  │  • Age per merchant  │                                      │
│  │  • Critical > 7d     │                                      │
│  │  • Warning > 3d      │                                      │
│  │  • Alert generation  │                                      │
│  └──────────┬───────────┘                                      │
│             │                                                  │
│             ▼                                                  │
│  ┌──────────────────────┐                                      │
│  │ AnomalyDetectionAgent│  ← Statistical + LLM narrative      │
│  │  • Z-score amounts   │                                      │
│  │  • Velocity spikes   │                                      │
│  │  • Star outliers     │                                      │
│  │  • Business narrative│                                      │
│  └──────────┬───────────┘                                      │
│             │                                                  │
│             ▼                                                  │
│  ┌──────────────────────┐                                      │
│  │    Summarize Node    │  ← Unified report                   │
│  └──────────┬───────────┘                                      │
│             │                                                  │
│            END                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Agent 1: DataValidationAgent

**Purpose:** Quality gate — validates incoming merchant data before it enters
the vector store and influences RAG answers. Bad data in = bad answers out.

**Trigger:** Every pipeline reload, every `/ingest` POST.

### Decision Tree

```
For each merchant record:
│
├─ Is merchant_id present?
│    No  → FAIL: "Missing required field: merchant_id"
│    Yes ↓
│
├─ Is merchant_name present when merchant_id is set?
│    No  → FAIL: "Referential integrity: merchant_id without merchant_name"
│    Yes ↓
│
├─ Is MCC code in VALID_MCC_CODES (ISO 18245)?
│    No  → FAIL: "Invalid MCC code: XXXX"
│    Yes ↓
│
├─ Is transaction_amount between 0.01 and 99,999.99?
│    No  → FAIL: "Amount out of range: X"
│    Yes ↓
│
└─ PASS ✓

If pass_rate >= 95% → status: "pass" → proceed to embedding
If pass_rate <  95% → status: "fail" → alert + partial ingest
```

### Output Schema
```json
{
  "status": "pass | fail",
  "total": 150000,
  "passed": 148500,
  "failed": 1500,
  "pass_rate": "99.0%",
  "ready_for_embedding": true,
  "issues_sample": [
    {"merchant_id": "biz_999", "issues": ["Invalid MCC code: 0000"]}
  ]
}
```

### Design Decision
We use a 95% pass-rate threshold rather than 100% to avoid blocking the entire
pipeline on a single malformed record. The 5% tolerance is common in enterprise
ETL — production Kafka streams always have some noise.

---

## Agent 2: DataFreshnessAgent

**Purpose:** SLA monitoring — alerts when merchant data hasn't been updated
within defined thresholds. A RAG system answering "is this merchant open?"
with 2-week-old data is worse than no answer.

**Trigger:** Every pipeline reload. Nightly via EventBridge.

### SLA Thresholds

| Status | Age | Action |
|---|---|---|
| `healthy` | < 1 day | No action |
| `warning` | 1–7 days | Slack alert |
| `critical` | > 7 days | Page on-call + SNS |

### Decision Tree

```
For each merchant record:
│
├─ Is timestamp or last_ingested_at present?
│    No  → CRITICAL: "No timestamp available"
│    Yes ↓
│
├─ Calculate age_days = now - timestamp
│
├─ age_days >= 7?
│    Yes → Add to stale_critical list
│    No  ↓
│
├─ age_days >= 3?
│    Yes → Add to stale_warning list
│    No  ↓
│
└─ Add to fresh list ✓

Aggregate:
  critical_pct = len(stale_critical) / total
  critical_pct > 10% → overall status: "critical"
  any stale_warning  → overall status: "warning"
  else               → overall status: "healthy"
```

### Output Schema
```json
{
  "status": "healthy | warning | critical",
  "total": 150000,
  "fresh": 148000,
  "stale_warning": 1500,
  "stale_critical": 500,
  "freshness_pct": "98.7%",
  "critical_merchants": [...],
  "sla_thresholds_days": {"critical": 7, "warning": 3, "healthy": 1}
}
```

### Design Decision
The freshness thresholds are configurable. For a payments platform, 7 days is
extremely stale — you'd tighten this to 24–48 hours. For a restaurant directory,
weekly is acceptable. The agent is threshold-agnostic.

---

## Agent 3: AnomalyDetectionAgent

**Purpose:** Statistical anomaly detection with LLM narrative interpretation.
Flags merchants whose transaction patterns deviate significantly from the corpus
— high-value outliers, viral review spikes, unusually low star ratings.

**Trigger:** Every pipeline reload. On-demand via `POST /agents/run?task=anomaly_scan`.

### Detection Methods

#### Method 1: Z-score on Transaction Amounts

```
μ = mean(all transaction_amounts)
σ = stddev(all transaction_amounts)

For each merchant:
  z = (merchant.amount - μ) / σ
  |z| >= 2.5 → flag as amount_outlier

  Direction:
    z > 0 → "high amount" (premium positioning, potential fraud)
    z < 0 → "low amount"  (discount/fast-casual, or data error)
```

#### Method 2: Review Velocity Spikes

```
μ_v = mean(all review_velocity_30d)

For each merchant:
  ratio = merchant.velocity / μ_v
  ratio >= 3.0 AND velocity > 10 → flag as velocity_spike

Narrative: "3.2x the dataset average — signals viral growth or a recent event"
```

#### Method 3: Low Star Outliers

```
μ_s = mean(all stars)
σ_s = stddev(all stars)

For each merchant:
  z_s = (merchant.stars - μ_s) / σ_s
  z_s < -2.5 → flag as low_star_outlier

Narrative: "Significantly below mean — may warrant quality review"
```

### Output Schema
```json
{
  "anomalies": [
    {
      "type": "amount_outlier | velocity_spike | low_star_outlier",
      "merchant_id": "biz_006",
      "merchant_name": "Alinea",
      "value": 350.00,
      "z_score": 4.2,
      "direction": "high",
      "narrative": "Alinea has an unusually high avg transaction of $350 (mean: $45, z=4.2). Indicates luxury/tasting-menu positioning."
    }
  ]
}
```

---

## LangGraph Implementation Notes

### Why LangGraph over a simple function chain?

1. **Typed state** — `GraphState` TypedDict ensures each node's output is validated
2. **Conditional edges** — can branch based on validation result (e.g., skip embedding if fail)
3. **Streaming** — nodes can stream intermediate results
4. **Persistence** — LangGraph supports checkpointing for long-running agent runs
5. **Observability** — every node transition is traced in LangSmith automatically

### Fallback (no langgraph installed)

When `langgraph` is not installed, `MerchantAgentGraph` executes the three agents
sequentially in Python — identical logic, zero LangGraph dependency. This ensures
the project runs in CI/CD without the full LangGraph stack.

```bash
# Install full stack
pip install langgraph langchain langchain-anthropic langsmith

# Verify
python -c "from agents.graph import MerchantAgentGraph; print('LangGraph ready')"
```

### Run the agents

```bash
# Via API
curl -X POST "http://localhost:8000/agents/run?task=anomaly_scan"

# Via Python
from agents.graph import MerchantAgentGraph
graph = MerchantAgentGraph()
result = await graph.run(merchant_dicts, task="pipeline_health_check")
print(result["final_answer"])
```

---

## Adding a New Agent

1. Create a class with an `async def run(self, state: GraphState) -> GraphState` method
2. Add `state["steps"].append("your_agent_name")` at the top
3. Register in `MerchantAgentGraph._try_build_langgraph()`:
   ```python
   graph.add_node("your_agent", self._your_agent_node)
   graph.add_edge("detect_anomalies", "your_agent")
   graph.add_edge("your_agent", "summarize")
   ```
4. Write tests in `tests/test_merchantrag.py`
