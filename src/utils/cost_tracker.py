"""Cost tracking: appends per-call records to cost_log.jsonl and Prometheus."""
from __future__ import annotations
import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

COST_LOG_PATH = Path(os.getenv("COST_LOG_PATH", "cost_log.jsonl"))

COST_PER_TOKEN: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {"input": 0.00000025, "output": 0.00000125},
    "claude-sonnet-4-6": {"input": 0.000003,  "output": 0.000015},
    "gpt-4o-mini":       {"input": 0.00000015, "output": 0.0000006},
    "gpt-4o":            {"input": 0.0000025,  "output": 0.00001},
}

COST_BUDGET_PER_TRANSACTION = float(os.getenv("COST_BUDGET_PER_TRANSACTION_USD", "0.08"))

if PROMETHEUS_AVAILABLE:
    AGENT_TOKEN_COUNTER = Counter(
        "agent_tokens_total",
        "Total tokens consumed per agent",
        ["agent_name", "model", "token_type"],
    )
    TRANSACTION_COST_HISTOGRAM = Histogram(
        "transaction_cost_usd",
        "Cost per transaction pipeline run",
        buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25],
    )
    BUDGET_BREACH_COUNTER = Counter(
        "budget_breach_total",
        "Transactions exceeding cost budget",
        ["agent_name"],
    )
    MODEL_ROUTING_COUNTER = Counter(
        "model_routing_total",
        "LLM routing decisions by tier",
        ["agent_name", "tier"],
    )


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate USD cost from token counts."""
    rates = COST_PER_TOKEN.get(model, {"input": 0.000001, "output": 0.000003})
    return (input_tokens * rates["input"]) + (output_tokens * rates["output"])


def record_agent_call(
    *,
    agent_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    transaction_id: str,
    phase: str,
    duration_seconds: float,
    confidence: Optional[float] = None,
    passed: Optional[bool] = None,
) -> float:
    """Record a single LLM agent call to cost_log.jsonl and Prometheus.

    Returns:
        cost_usd: actual cost calculated from token counts
    """
    cost_usd = calculate_cost(model, input_tokens, output_tokens)
    budget_breached = cost_usd > COST_BUDGET_PER_TRANSACTION
    tier = "strong" if any(s in model for s in ["sonnet", "opus", "gpt-4o:"]) else "cheap"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "transaction_id": transaction_id,
        "phase": phase,
        "agent_name": agent_name,
        "model": model,
        "tier": tier,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 8),
        "duration_seconds": round(duration_seconds, 3),
        "confidence": confidence,
        "passed": passed,
        "budget_breached": budget_breached,
    }

    with open(COST_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    if PROMETHEUS_AVAILABLE:
        AGENT_TOKEN_COUNTER.labels(agent_name, model, "input").inc(input_tokens)
        AGENT_TOKEN_COUNTER.labels(agent_name, model, "output").inc(output_tokens)
        TRANSACTION_COST_HISTOGRAM.observe(cost_usd)
        MODEL_ROUTING_COUNTER.labels(agent_name, tier).inc()
        if budget_breached:
            BUDGET_BREACH_COUNTER.labels(agent_name).inc()
            logger.warning(
                "budget_breach",
                extra={"agent": agent_name, "cost_usd": cost_usd, "tx": transaction_id},
            )

    return cost_usd
