"""GraphState definition for the fraud explanation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from src.schemas.detection import FraudDetectionResult
from src.schemas.explanation import ExplanationEvalResult, ExplanationResult
from src.schemas.transactions import FraudTransaction


class GraphState(TypedDict, total=False):
    """LangGraph state dict for the fraud explanation pipeline.

    Each node reads/writes specific keys. The orchestrator routes
    based on presence of errors or missing data.
    """

    # Input
    transaction: FraudTransaction

    # DetectorModel output
    detection_result: FraudDetectionResult

    # ExplanationAgent output (analyst mode — primary)
    explanation_result: ExplanationResult

    # EvalAgent output
    eval_result: ExplanationEvalResult

    # Error tracking — populated on partial failure
    error: str | None
    error_stage: str | None

    # Pipeline metadata
    completed: bool


@dataclass
class PipelineResult:
    """Convenience wrapper for the final pipeline output.

    Provides typed access to the graph state after execution.
    """

    transaction: FraudTransaction
    detection_result: FraudDetectionResult | None = None
    explanation_result: ExplanationResult | None = None
    eval_result: ExplanationEvalResult | None = None
    error: str | None = None
    error_stage: str | None = None
    completed: bool = False

    @classmethod
    def from_state(cls, state: dict) -> PipelineResult:
        """Construct PipelineResult from a GraphState dict."""
        return cls(
            transaction=state["transaction"],
            detection_result=state.get("detection_result"),
            explanation_result=state.get("explanation_result"),
            eval_result=state.get("eval_result"),
            error=state.get("error"),
            error_stage=state.get("error_stage"),
            completed=state.get("completed", False),
        )
