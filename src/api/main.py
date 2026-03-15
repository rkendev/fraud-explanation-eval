"""FastAPI application: SSE streaming, API key auth, rate limiting, Prometheus."""

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Body, Depends, FastAPI, Query, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import StreamingResponse

from src.api.auth import verify_api_key
from src.api.schemas import HealthResponse, PipelineResponse
from src.models.detector import FraudDetector, ModelNotLoadedError
from src.orchestrator.graph import (
    _node_detect,
    _node_evaluate,
    _node_explain,
    _node_sanitize,
    run_pipeline,
)
from src.orchestrator.state import GraphState, PipelineResult
from src.schemas.transactions import FraudTransaction
from src.security.sanitizer import InjectionDetectedError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

API_REQUEST_COUNTER = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status_code"],
)
API_REQUEST_DURATION = Histogram(
    "api_request_duration_seconds",
    "API request duration",
    ["endpoint"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
)

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")
limiter = Limiter(key_func=get_remote_address)


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> Response:
    return Response(
        content=json.dumps({"detail": "Rate limit exceeded"}),
        status_code=429,
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load detector model at startup."""
    try:
        detector = FraudDetector.load()
        app.state.detector = detector
        logger.info("detector_loaded", extra={"version": detector.model_version})
    except (ModelNotLoadedError, Exception) as exc:
        logger.warning("detector_not_loaded", extra={"error": str(exc)})
        app.state.detector = None
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fraud Explanation API",
    version="0.1.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)


# ---------------------------------------------------------------------------
# Middleware: request metrics
# ---------------------------------------------------------------------------


@app.middleware("http")
async def record_request_metrics(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    endpoint = request.url.path
    API_REQUEST_COUNTER.labels(endpoint, request.method, response.status_code).inc()
    API_REQUEST_DURATION.labels(endpoint).observe(duration)
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    detector = getattr(app.state, "detector", None)
    return HealthResponse(
        status="ok",
        version="0.1.0",
        model_loaded=detector is not None,
    )


@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/api/v1/analyze", response_model=PipelineResponse)
@limiter.limit(RATE_LIMIT)
async def analyze(
    request: Request,
    transaction: FraudTransaction = Body(...),
    target_audience: Literal["analyst", "customer"] = Query(default="analyst"),
    _api_key: str = Depends(verify_api_key),
):
    """Run the full fraud explanation pipeline on a transaction."""
    detector = getattr(app.state, "detector", None)
    if detector is None:
        return Response(
            content=json.dumps({"detail": "Model not loaded"}),
            status_code=503,
            media_type="application/json",
        )

    result = await asyncio.to_thread(
        run_pipeline,
        transaction,
        detector=detector,
        target_audience=target_audience,
    )
    return PipelineResponse.from_pipeline_result(result)


@app.post("/api/v1/analyze/stream")
@limiter.limit(RATE_LIMIT)
async def analyze_stream(
    request: Request,
    transaction: FraudTransaction = Body(...),
    target_audience: Literal["analyst", "customer"] = Query(default="analyst"),
    _api_key: str = Depends(verify_api_key),
):
    """Run the pipeline with SSE streaming — emit events per stage."""
    detector = getattr(app.state, "detector", None)
    if detector is None:
        return Response(
            content=json.dumps({"detail": "Model not loaded"}),
            status_code=503,
            media_type="application/json",
        )

    return StreamingResponse(
        _stream_pipeline(transaction, detector, target_audience),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# SSE streaming generator
# ---------------------------------------------------------------------------


async def _stream_pipeline(
    transaction: FraudTransaction,
    detector: FraudDetector,
    target_audience: Literal["analyst", "customer"],
) -> AsyncGenerator[str, None]:
    """Yield SSE events for each pipeline stage."""

    from src.agents.eval_agent import EvalAgent
    from src.agents.explanation_agent import (
        ExplanationAgent,
        ExplanationHallucinationError,
    )

    state: GraphState = {
        "transaction": transaction,
        "error": None,
        "error_stage": None,
        "completed": False,
    }

    def _sse(event: str, data: str) -> str:
        return f"event: {event}\ndata: {data}\n\n"

    # Stage 1: Sanitize
    try:
        updates = await asyncio.to_thread(_node_sanitize, state)
        state.update(updates)
        yield _sse("sanitize", json.dumps({"status": "ok"}))
    except InjectionDetectedError as exc:
        yield _sse("error", json.dumps({"error": str(exc), "error_stage": "sanitize"}))
        return

    # Stage 2: Detect
    try:
        updates = await asyncio.to_thread(_node_detect, state, detector=detector)
        state.update(updates)
        yield _sse("detection", state["detection_result"].model_dump_json())
    except Exception as exc:
        yield _sse("error", json.dumps({"error": str(exc), "error_stage": "detect"}))
        return

    # Stage 3: Explain
    try:
        explanation_agent = ExplanationAgent()
        updates = await asyncio.to_thread(
            _node_explain,
            state,
            explanation_agent=explanation_agent,
            target_audience=target_audience,
        )
        state.update(updates)
        yield _sse("explanation", state["explanation_result"].model_dump_json())
    except ExplanationHallucinationError as exc:
        yield _sse("error", json.dumps({"error": str(exc), "error_stage": "explain"}))
        return

    # Stage 4: Evaluate
    try:
        eval_agent = EvalAgent()
        updates = await asyncio.to_thread(_node_evaluate, state, eval_agent=eval_agent)
        state.update(updates)
        if state.get("eval_result"):
            yield _sse("evaluation", state["eval_result"].model_dump_json())
    except Exception as exc:
        yield _sse("error", json.dumps({"error": str(exc), "error_stage": "evaluate"}))

    # Final
    result = PipelineResult.from_state(state)
    response = PipelineResponse.from_pipeline_result(result)
    yield _sse("complete", response.model_dump_json())
