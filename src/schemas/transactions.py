"""IEEE-CIS Fraud Detection transaction schema."""
from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, field_validator, model_validator


class FraudTransaction(BaseModel):
    """Input transaction from IEEE-CIS dataset."""
    TransactionID: str
    TransactionAmt: float
    ProductCD: Literal["W", "H", "C", "S", "R"]
    card1: Optional[int] = None
    card4: Optional[Literal["discover", "mastercard", "visa", "american express"]] = None
    card6: Optional[Literal["credit", "debit"]] = None
    addr1: Optional[int] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    DeviceType: Optional[Literal["desktop", "mobile"]] = None
    DeviceInfo: Optional[str] = None

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
    def sanitize_text_field(cls, v: Optional[str]) -> Optional[str]:
        """Truncate suspiciously long text fields before they reach validation."""
        if v is not None and len(v) > 256:
            return v[:256]
        return v
