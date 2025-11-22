from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, HttpUrl, field_validator


class SolveRequest(BaseModel):
    """
    Incoming payload for quiz-solving requests.

    The fields mirror the contract shared in Question.md, with email/secret/url as
    mandatory inputs. Additional metadata can be accommodated via the `extras`
    field to stay future-proof.
    """

    model_config = ConfigDict(extra="forbid")

    email: EmailStr = Field(..., description="Registered student email address.")
    secret: str = Field(
        ..., min_length=1, max_length=256, description="Shared secret string."
    )
    url: HttpUrl = Field(..., description="Quiz/task entry URL supplied by TDS.")
    extras: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional metadata relayed by the platform (e.g., attempt IDs).",
    )

    @field_validator("secret")
    @classmethod
    def secret_must_not_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("secret must not be blank or whitespace.")
        return value


class SolveResponse(BaseModel):
    """
    Standard API response emitted by the FastAPI endpoint.

    Success/failure is captured via `status`, with additional error context placed in
    `reason`. When a quiz was processed successfully, `accepted` conveys whether the
    payload passed authentication/validation, and `details` may include supporting
    diagnostic information that is safe to expose to the caller.
    """

    model_config = ConfigDict(extra="allow")

    status: Literal["ok", "error"] = Field(
        ..., description="'ok' for HTTP 200 responses, 'error' otherwise."
    )
    accepted: bool = Field(
        ...,
        description="Indicates whether the request passed validation & secret checks.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Optional human-readable explanation when status='error'.",
    )
    details: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional context (e.g., model usage, throttling info).",
    )


class HealthResponse(BaseModel):
    """Lightweight schema for liveness/readiness probes."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["healthy", "degraded", "unhealthy"]
    llm_provider: str = Field(
        ..., description="Current LLM routing target (e.g., AIPipe -> gpt-4.1-nano)."
    )
    message: Optional[str] = Field(
        default=None, description="Optional note explaining degraded/unhealthy states."
    )
