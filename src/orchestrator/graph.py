"""LangGraph state machine: FraudTransaction -> Detection -> Explanation -> Eval."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from src.agents.eval_agent import EvalAgent
from src.agents.explanation_agent import ExplanationAgent, ExplanationHallucinationError
from src.models.detector import (
    FraudDetector,
    InferenceTimeoutError,
    ModelNotLoadedError,
    TransactionValidationError,
)
from src.orchestrator.state import GraphState, PipelineResult
from src.security.sanitizer import InjectionDetectedError, sanitize_external_text
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Fields that must be sanitized before entering the pipeline
_SANITIZE_FIELDS = ("DeviceInfo", "P_emaildomain", "R_emaildomain")


def _node_sanitize(state: GraphState) -> dict:
    """Sanitize text fields on the transaction before processing."""
    tx = state["transaction"]

    for field_name in _SANITIZE_FIELDS:
        value = getattr(tx, field_name, None)
        if value and isinstance(value, str):
            sanitize_external_text(value, source=field_name)

    logger.info("sanitize_ok", extra={"transaction_id": tx.TransactionID})
    return {}


def _node_detect(state: GraphState, *, detector: FraudDetector) -> dict:
    """Run XGBoost inference and SHAP extraction."""
    tx = state["transaction"]
    detection_result = detector.predict(tx)
    logger.info(
        "detect_ok",
        extra={
            "transaction_id": tx.TransactionID,
            "fraud_probability": detection_result.fraud_probability,
        },
    )
    return {"detection_result": detection_result}


def _node_explain(
    state: GraphState,
    *,
    explanation_agent: ExplanationAgent,
    target_audience: Literal["analyst", "customer"] = "analyst",
) -> dict:
    """Generate a grounded explanation of the detection result."""
    detection_result = state["detection_result"]
    explanation_result = explanation_agent.explain(detection_result, target_audience)
    logger.info(
        "explain_ok",
        extra={
            "transaction_id": detection_result.transaction_id,
            "explanation_generated": explanation_result.explanation_generated,
        },
    )
    return {"explanation_result": explanation_result}


def _node_evaluate(state: GraphState, *, eval_agent: EvalAgent) -> dict:
    """Score the explanation via LLM-as-judge."""
    explanation_result = state["explanation_result"]
    detection_result = state["detection_result"]

    # Skip eval if explanation was not generated
    if not explanation_result.explanation_generated:
        logger.info(
            "eval_skipped",
            extra={
                "transaction_id": explanation_result.transaction_id,
                "reason": explanation_result.warning or "no_explanation",
            },
        )
        return {"completed": True}

    eval_result = eval_agent.evaluate(explanation_result, detection_result)
    logger.info(
        "eval_ok",
        extra={
            "transaction_id": eval_result.transaction_id,
            "overall_score": eval_result.overall_score,
            "passed": eval_result.passed,
        },
    )
    return {"eval_result": eval_result, "completed": True}


def _node_handle_error(state: GraphState) -> dict:
    """Terminal node — error has already been recorded in state."""
    logger.warning(
        "pipeline_error",
        extra={
            "transaction_id": state["transaction"].TransactionID,
            "error_stage": state.get("error_stage"),
            "error": state.get("error"),
        },
    )
    return {"completed": True}


def _route_after_sanitize(state: GraphState) -> str:
    """Route based on whether sanitization set an error."""
    if state.get("error"):
        return "handle_error"
    return "detect"


def _route_after_detect(state: GraphState) -> str:
    if state.get("error"):
        return "handle_error"
    return "explain"


def _route_after_explain(state: GraphState) -> str:
    if state.get("error"):
        return "handle_error"
    return "evaluate"


def build_graph(
    *,
    detector: FraudDetector,
    explanation_agent: ExplanationAgent | None = None,
    eval_agent: EvalAgent | None = None,
    target_audience: Literal["analyst", "customer"] = "analyst",
) -> StateGraph:
    """Build the LangGraph state machine for the fraud pipeline.

    Args:
        detector: Pre-loaded FraudDetector instance.
        explanation_agent: ExplanationAgent (created if None).
        eval_agent: EvalAgent (created if None).
        target_audience: Audience mode for explanation generation.

    Returns:
        Compiled LangGraph StateGraph.
    """
    explanation_agent = explanation_agent or ExplanationAgent()
    eval_agent = eval_agent or EvalAgent()

    # Wrap nodes with error handling so partial failures produce degraded output
    def sanitize_node(state: GraphState) -> dict:
        try:
            return _node_sanitize(state)
        except InjectionDetectedError as e:
            return {
                "error": str(e),
                "error_stage": "sanitize",
            }

    def detect_node(state: GraphState) -> dict:
        try:
            return _node_detect(state, detector=detector)
        except (
            ModelNotLoadedError,
            TransactionValidationError,
            InferenceTimeoutError,
        ) as e:
            return {
                "error": str(e),
                "error_stage": "detect",
            }

    def explain_node(state: GraphState) -> dict:
        try:
            return _node_explain(
                state,
                explanation_agent=explanation_agent,
                target_audience=target_audience,
            )
        except ExplanationHallucinationError as e:
            return {
                "error": str(e),
                "error_stage": "explain",
            }

    def evaluate_node(state: GraphState) -> dict:
        try:
            return _node_evaluate(state, eval_agent=eval_agent)
        except Exception as e:
            return {
                "error": str(e),
                "error_stage": "evaluate",
                "completed": True,
            }

    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("sanitize", sanitize_node)
    graph.add_node("detect", detect_node)
    graph.add_node("explain", explain_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("handle_error", _node_handle_error)

    # Set entry point
    graph.set_entry_point("sanitize")

    # Conditional edges
    graph.add_conditional_edges("sanitize", _route_after_sanitize)
    graph.add_conditional_edges("detect", _route_after_detect)
    graph.add_conditional_edges("explain", _route_after_explain)

    # Terminal edges
    graph.add_edge("evaluate", END)
    graph.add_edge("handle_error", END)

    return graph.compile()


def run_pipeline(
    transaction,
    *,
    detector: FraudDetector,
    explanation_agent: ExplanationAgent | None = None,
    eval_agent: EvalAgent | None = None,
    target_audience: Literal["analyst", "customer"] = "analyst",
) -> PipelineResult:
    """Run the full fraud explanation pipeline on a single transaction.

    This is the main entry point for the orchestrator. It builds the graph,
    invokes it, and returns a typed PipelineResult.

    Args:
        transaction: Validated FraudTransaction.
        detector: Pre-loaded FraudDetector.
        explanation_agent: Optional ExplanationAgent override.
        eval_agent: Optional EvalAgent override.
        target_audience: "analyst" or "customer".

    Returns:
        PipelineResult with all available outputs and any error info.
    """
    compiled = build_graph(
        detector=detector,
        explanation_agent=explanation_agent,
        eval_agent=eval_agent,
        target_audience=target_audience,
    )

    initial_state: GraphState = {
        "transaction": transaction,
        "error": None,
        "error_stage": None,
        "completed": False,
    }

    final_state = compiled.invoke(initial_state)
    return PipelineResult.from_state(final_state)
