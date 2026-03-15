"""API request/response schemas for the fraud explanation pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.orchestrator.state import PipelineResult


class PipelineResponse(BaseModel):
    """JSON response wrapping PipelineResult."""

    transaction_id: str
    completed: bool = False
    detection_result: dict[str, Any] | None = None
    explanation_result: dict[str, Any] | None = None
    eval_result: dict[str, Any] | None = None
    error: str | None = None
    error_stage: str | None = None

    @classmethod
    def from_pipeline_result(cls, result: PipelineResult) -> PipelineResponse:
        return cls(
            transaction_id=result.transaction.TransactionID,
            completed=result.completed,
            detection_result=(
                result.detection_result.model_dump()
                if result.detection_result
                else None
            ),
            explanation_result=(
                result.explanation_result.model_dump()
                if result.explanation_result
                else None
            ),
            eval_result=(
                result.eval_result.model_dump() if result.eval_result else None
            ),
            error=result.error,
            error_stage=result.error_stage,
        )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    model_loaded: bool
