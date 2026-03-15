"""Unit tests for the FastAPI fraud explanation API.

Tests cover:
- Authentication: unauthenticated requests rejected, valid key accepted
- Health/metrics: no auth required, correct response shape
- Analyze endpoint: happy path, customer mode, invalid input, model not loaded
- Stream endpoint: SSE events emitted per stage
- Rate limiting: 429 on exceeded limit
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from src.api.auth import reset_api_key_cache
from src.models.detector import FraudDetector
from src.orchestrator.state import PipelineResult
from src.schemas.detection import FraudDetectionResult, SHAPFeature
from src.schemas.explanation import ExplanationEvalResult, ExplanationResult
from src.schemas.transactions import FraudTransaction

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_API_KEY = "test-key-12345"


@pytest.fixture(autouse=True)
def _clean_api_key_env(monkeypatch):
    """Ensure API_KEY is always controlled in tests — never leaks from .env."""
    monkeypatch.delenv("API_KEY", raising=False)
    reset_api_key_cache()
    yield
    reset_api_key_cache()


@pytest.fixture()
def _set_api_key(monkeypatch):
    """Set the API_KEY env var for auth-required tests."""
    monkeypatch.setenv("API_KEY", TEST_API_KEY)
    reset_api_key_cache()


def _auth_headers() -> dict[str, str]:
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture()
def sample_transaction_data() -> dict[str, Any]:
    return {
        "TransactionID": "TX_API_001",
        "TransactionAmt": 299.99,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "debit",
        "addr1": 325,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
        "DeviceType": "desktop",
        "DeviceInfo": "Windows 10",
    }


@pytest.fixture()
def mock_detection() -> FraudDetectionResult:
    return FraudDetectionResult(
        transaction_id="TX_API_001",
        fraud_probability=0.87,
        is_fraud_predicted=True,
        top_shap_features=[
            SHAPFeature(
                feature_name="TransactionAmt", shap_value=0.45, feature_value=299.99
            ),
            SHAPFeature(
                feature_name="DeviceInfo", shap_value=0.31, feature_value="Windows 10"
            ),
            SHAPFeature(
                feature_name="P_emaildomain", shap_value=0.22, feature_value="gmail.com"
            ),
            SHAPFeature(feature_name="card6", shap_value=0.18, feature_value="debit"),
            SHAPFeature(feature_name="addr1", shap_value=-0.12, feature_value=325),
        ],
        model_version="1.0.0",
        inference_latency_ms=4.2,
        confidence_tier="high",
    )


@pytest.fixture()
def mock_explanation() -> ExplanationResult:
    return ExplanationResult(
        transaction_id="TX_API_001",
        target_audience="analyst",
        fraud_probability=0.87,
        is_fraud_predicted=True,
        explanation_text=(
            "Transaction TX_API_001 has a fraud probability of 87.0%. "
            "TransactionAmt of $299.99 contributes most strongly."
        ),
        cited_features=["TransactionAmt", "DeviceInfo"],
        uncited_features=["P_emaildomain", "card6", "addr1"],
        hallucinated_features=[],
        uncertainty_flag=False,
        explanation_generated=True,
        token_cost_usd=0.001234,
        generation_latency_seconds=3.2,
    )


@pytest.fixture()
def mock_eval() -> ExplanationEvalResult:
    return ExplanationEvalResult(
        transaction_id="TX_API_001",
        target_audience="analyst",
        grounding_score=0.95,
        clarity_score=0.90,
        completeness_score=0.85,
        audience_appropriateness_score=0.90,
        overall_score=0.90,
        passed=True,
        failure_reasons=[],
        token_cost_usd=0.002345,
    )


def _make_pipeline_result(
    mock_detection, mock_explanation, mock_eval, **overrides
) -> PipelineResult:
    defaults = dict(
        transaction=FraudTransaction(
            TransactionID="TX_API_001",
            TransactionAmt=299.99,
            ProductCD="W",
        ),
        detection_result=mock_detection,
        explanation_result=mock_explanation,
        eval_result=mock_eval,
        completed=True,
    )
    defaults.update(overrides)
    return PipelineResult(**defaults)


@pytest.fixture()
def client_with_model(mock_detection, mock_explanation, mock_eval):
    """TestClient with a mock detector loaded at startup."""
    mock_detector = MagicMock(spec=FraudDetector)
    mock_detector.predict.return_value = mock_detection
    mock_detector.model_version = "1.0.0"

    with patch("src.api.main.FraudDetector.load", return_value=mock_detector):
        from src.api.main import app

        with TestClient(app) as c:
            with patch("src.api.main.run_pipeline") as mock_run:
                mock_run.return_value = _make_pipeline_result(
                    mock_detection, mock_explanation, mock_eval
                )
                c._mock_run_pipeline = mock_run
                yield c


@pytest.fixture()
def client_no_model():
    """TestClient with no detector model loaded."""
    with patch(
        "src.api.main.FraudDetector.load", side_effect=Exception("Model not found")
    ):
        from src.api.main import app

        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Tests: Authentication
# ---------------------------------------------------------------------------


class TestAuthentication:
    """API rejects unauthenticated requests."""

    def test_analyze_without_api_key_returns_401(
        self, _set_api_key, client_with_model, sample_transaction_data
    ):
        resp = client_with_model.post("/api/v1/analyze", json=sample_transaction_data)
        assert resp.status_code == 401

    def test_analyze_with_wrong_api_key_returns_401(
        self, _set_api_key, client_with_model, sample_transaction_data
    ):
        resp = client_with_model.post(
            "/api/v1/analyze",
            json=sample_transaction_data,
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_analyze_with_valid_api_key_succeeds(
        self, _set_api_key, client_with_model, sample_transaction_data
    ):
        resp = client_with_model.post(
            "/api/v1/analyze",
            json=sample_transaction_data,
            headers=_auth_headers(),
        )
        assert resp.status_code == 200

    def test_health_requires_no_auth(self, _set_api_key, client_with_model):
        resp = client_with_model.get("/health")
        assert resp.status_code == 200

    def test_metrics_requires_no_auth(self, _set_api_key, client_with_model):
        resp = client_with_model.get("/metrics")
        assert resp.status_code == 200

    def test_stream_without_api_key_returns_401(
        self, _set_api_key, client_with_model, sample_transaction_data
    ):
        resp = client_with_model.post(
            "/api/v1/analyze/stream", json=sample_transaction_data
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Tests: Health & Metrics
# ---------------------------------------------------------------------------


class TestHealthAndMetrics:
    """Health and metrics endpoints (no auth required in dev mode)."""

    def test_health_returns_status(self, client_with_model):
        resp = client_with_model.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert data["model_loaded"] is True

    def test_health_model_not_loaded(self, client_no_model):
        resp = client_no_model.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is False

    def test_metrics_returns_prometheus_format(self, client_with_model):
        resp = client_with_model.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")
        body = resp.text
        assert "api_requests_total" in body or "python_info" in body


# ---------------------------------------------------------------------------
# Tests: Analyze endpoint (dev mode — no API_KEY, so auth passes)
# ---------------------------------------------------------------------------


class TestAnalyzeEndpoint:
    """POST /api/v1/analyze tests."""

    def test_analyze_happy_path(self, client_with_model, sample_transaction_data):
        resp = client_with_model.post(
            "/api/v1/analyze",
            json=sample_transaction_data,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["completed"] is True
        assert data["transaction_id"] == "TX_API_001"
        assert data["detection_result"] is not None
        assert data["explanation_result"] is not None
        assert data["eval_result"] is not None
        assert data["error"] is None

    def test_analyze_with_customer_audience(
        self, client_with_model, sample_transaction_data
    ):
        resp = client_with_model.post(
            "/api/v1/analyze?target_audience=customer",
            json=sample_transaction_data,
        )
        assert resp.status_code == 200

    def test_analyze_invalid_transaction_returns_422(self, client_with_model):
        resp = client_with_model.post(
            "/api/v1/analyze",
            json={"TransactionID": "TX_BAD"},  # missing required fields
        )
        assert resp.status_code == 422

    def test_analyze_detector_not_loaded_returns_503(
        self, client_no_model, sample_transaction_data
    ):
        resp = client_no_model.post(
            "/api/v1/analyze",
            json=sample_transaction_data,
        )
        assert resp.status_code == 503
        assert "not loaded" in resp.json()["detail"].lower()

    def test_analyze_pipeline_error_returns_200_with_error(
        self, client_with_model, sample_transaction_data
    ):
        """Pipeline errors (e.g., injection) still return 200 with error in body."""
        client_with_model._mock_run_pipeline.return_value = PipelineResult(
            transaction=FraudTransaction(
                TransactionID="TX_API_001",
                TransactionAmt=299.99,
                ProductCD="W",
            ),
            error="Injection detected in DeviceInfo",
            error_stage="sanitize",
            completed=True,
        )
        resp = client_with_model.post(
            "/api/v1/analyze",
            json=sample_transaction_data,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is not None
        assert data["error_stage"] == "sanitize"


# ---------------------------------------------------------------------------
# Tests: Stream endpoint
# ---------------------------------------------------------------------------


class TestStreamEndpoint:
    """POST /api/v1/analyze/stream tests."""

    def test_stream_endpoint_returns_sse(
        self,
        client_with_model,
        sample_transaction_data,
        mock_detection,
        mock_explanation,
        mock_eval,
    ):
        with patch("src.api.main._node_sanitize", return_value={}), patch(
            "src.api.main._node_detect",
            return_value={"detection_result": mock_detection},
        ), patch(
            "src.api.main._node_explain",
            return_value={"explanation_result": mock_explanation},
        ), patch(
            "src.api.main._node_evaluate",
            return_value={"eval_result": mock_eval, "completed": True},
        ):
            resp = client_with_model.post(
                "/api/v1/analyze/stream",
                json=sample_transaction_data,
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_contains_stage_events(
        self,
        client_with_model,
        sample_transaction_data,
        mock_detection,
        mock_explanation,
        mock_eval,
    ):
        with patch("src.api.main._node_sanitize", return_value={}), patch(
            "src.api.main._node_detect",
            return_value={"detection_result": mock_detection},
        ), patch(
            "src.api.main._node_explain",
            return_value={"explanation_result": mock_explanation},
        ), patch(
            "src.api.main._node_evaluate",
            return_value={"eval_result": mock_eval, "completed": True},
        ):
            resp = client_with_model.post(
                "/api/v1/analyze/stream",
                json=sample_transaction_data,
            )
        body = resp.text
        assert "event: sanitize" in body
        assert "event: detection" in body
        assert "event: explanation" in body
        assert "event: evaluation" in body
        assert "event: complete" in body

    def test_stream_detector_not_loaded_returns_503(
        self, client_no_model, sample_transaction_data
    ):
        resp = client_no_model.post(
            "/api/v1/analyze/stream",
            json=sample_transaction_data,
        )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Tests: Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Rate limiting returns 429 when exceeded."""

    def test_rate_limit_exceeded_returns_429(
        self, monkeypatch, sample_transaction_data
    ):
        """Exceed a very tight rate limit and get 429."""
        monkeypatch.setenv("API_KEY", TEST_API_KEY)
        reset_api_key_cache()

        mock_detector = MagicMock(spec=FraudDetector)
        mock_detector.model_version = "1.0.0"

        with patch(
            "src.api.main.FraudDetector.load", return_value=mock_detector
        ), patch("src.api.main.run_pipeline") as mock_run:

            mock_run.return_value = PipelineResult(
                transaction=FraudTransaction(
                    TransactionID="TX_API_001",
                    TransactionAmt=299.99,
                    ProductCD="W",
                ),
                completed=True,
            )

            from src.api.main import app, limiter

            original_limit = limiter._default_limits
            try:
                with TestClient(app) as client:
                    got_429 = False
                    for _ in range(20):
                        resp = client.post(
                            "/api/v1/analyze",
                            json=sample_transaction_data,
                            headers=_auth_headers(),
                        )
                        if resp.status_code == 429:
                            got_429 = True
                            break
                    assert got_429, "Expected 429 but never received one"
            finally:
                limiter._default_limits = original_limit
