"""IEEE-CIS Fraud Detection transaction schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator


class FraudTransaction(BaseModel):
    """Input transaction from IEEE-CIS dataset."""

    TransactionID: str
    TransactionAmt: float
    ProductCD: Literal["W", "H", "C", "S", "R"]
    card1: int | None = None
    card4: Literal["discover", "mastercard", "visa", "american express"] | None = None
    card6: Literal["credit", "debit"] | None = None
    addr1: int | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    DeviceType: Literal["desktop", "mobile"] | None = None
    DeviceInfo: str | None = None

    @field_validator("TransactionAmt")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"TransactionAmt must be positive, got {v}")
        return v

    @field_validator("TransactionID")
    @classmethod
    def id_must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("TransactionID must be non-empty")
        return v

    @field_validator("DeviceInfo", "P_emaildomain", "R_emaildomain", mode="before")
    @classmethod
    def sanitize_text_field(cls, v: str | None) -> str | None:
        """Truncate suspiciously long text fields before they reach validation."""
        if v is not None and len(v) > 256:
            return v[:256]
        return v
