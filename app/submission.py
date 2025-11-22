"""
Submission client responsible for packaging solver outputs and delivering them to
quiz-provided endpoints.

The helper exposes two public functions:

- ``build_submission_payload`` prepares a JSON-serializable dict that satisfies
  the quiz contract (email, secret, url, answer, optional extras).
- ``submit_answer`` performs the HTTP POST with payload size guarding, parses
  the JSON response, and surfaces a structured ``SubmissionResult`` record.

These utilities are designed to be consumed by the future Phase 8 orchestrator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional

import httpx

from app.config import Settings, get_settings
from app.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class SubmissionResult:
    """Structured response returned after attempting a quiz submission."""

    status_code: int
    payload: Dict[str, Any]
    correct: Optional[bool] = None
    next_url: Optional[str] = None
    reason: Optional[str] = None

    @classmethod
    def from_response(
        cls, status_code: int, payload: Dict[str, Any]
    ) -> "SubmissionResult":
        return cls(
            status_code=status_code,
            payload=payload,
            correct=payload.get("correct"),
            next_url=payload.get("url"),
            reason=payload.get("reason"),
        )


class SubmissionError(RuntimeError):
    """Raised when a submission cannot be completed successfully."""


def _json_ready(value: Any) -> Any:
    """Best-effort conversion into a JSON-serializable primitive."""
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        try:
            candidate = value.to_dict()
            json.dumps(candidate)
            return candidate
        except Exception:  # pragma: no cover - defensive path
            return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def build_submission_payload(
    *,
    email: str,
    secret: str,
    quiz_url: str,
    answer: Any,
    extras: Optional[Mapping[str, Any]] = None,
    template: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Construct a JSON-safe payload for quiz submission.

    IMPORTANT: The quiz page may specify the exact JSON structure required.
    If a template is provided (parsed from quiz instructions), use that structure.
    Otherwise, fall back to the default {email, secret, url, answer} format.

    Args:
        email: Registered student email.
        secret: Shared secret configured during onboarding.
        quiz_url: The quiz URL originally provided by TDS.
        answer: Solver-produced answer (number, string, bool, dict, list, etc.).
        extras: Optional mapping of additional keys to merge into the payload.
        template: Optional template from quiz page showing exact required structure.

    Returns:
        Dictionary ready for JSON serialization.
    """
    if template:
        # Use the quiz-specified template structure
        payload: MutableMapping[str, Any] = {}
        for key, value in template.items():
            # Replace placeholder values with actual data
            if key == "email" or key.lower().endswith("email"):
                payload[key] = email
            elif key == "secret" or key.lower() == "secret":
                payload[key] = secret
            elif key == "url" or key.lower() == "quiz_url":
                payload[key] = quiz_url
            elif key == "answer" or key.lower() in ["answer", "result", "solution"]:
                payload[key] = _json_ready(answer)
            else:
                # Keep other fields from template or override from extras
                if extras and key in extras:
                    payload[key] = _json_ready(extras[key])
                else:
                    payload[key] = value
        # Add any extras not already in template
        if extras:
            for key, value in extras.items():
                if key not in payload:
                    payload[key] = _json_ready(value)
    else:
        # Default payload structure
        payload: MutableMapping[str, Any] = {
            "email": email,
            "secret": secret,
            "url": quiz_url,
            "answer": _json_ready(answer),
        }
        if extras:
            for key, value in extras.items():
                if key not in payload:
                    payload[key] = _json_ready(value)

    return dict(payload)


async def submit_answer(
    *,
    submission_url: str,
    email: str,
    secret: str,
    quiz_url: str,
    answer: Any,
    extras: Optional[Mapping[str, Any]] = None,
    payload_template: Optional[Mapping[str, Any]] = None,
    settings: Optional[Settings] = None,
    client: Optional[httpx.AsyncClient] = None,
    timeout: Optional[float] = None,
) -> SubmissionResult:
    """
    Submit the solver's answer to the quiz-provided endpoint.

    IMPORTANT: If the quiz page shows an example JSON payload, pass it as
    payload_template to ensure we match the expected structure exactly.

    Args:
        submission_url: URL extracted from the quiz instructions.
        email: Registered student email.
        secret: Shared secret for authentication.
        quiz_url: Original quiz URL (echoed back per contract).
        answer: Solver-produced answer.
        extras: Optional additional payload fields.
        payload_template: Optional template structure from quiz instructions.
        settings: Optional Settings override (defaults to global settings).
        client: Optional httpx.AsyncClient reuse (caller remains owner).
        timeout: Optional request timeout override in seconds.

    Returns:
        SubmissionResult capturing the parsed response.

    Raises:
        SubmissionError: When the submission fails or the response is invalid.
    """
    if not submission_url:
        raise SubmissionError("submission_url is required for answer submission.")

    resolved_settings = settings or get_settings()
    payload = build_submission_payload(
        email=email,
        secret=secret,
        quiz_url=quiz_url,
        answer=answer,
        extras=extras,
        template=payload_template,
    )

    encoded_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    max_bytes = resolved_settings.max_payload_size_mb * 1024 * 1024
    if len(encoded_body) > max_bytes:
        raise SubmissionError(
            f"Submission payload exceeded {resolved_settings.max_payload_size_mb} MB limit."
        )

    owns_client = client is None
    http_client = client or httpx.AsyncClient(
        timeout=timeout or resolved_settings.request_timeout_seconds
    )

    try:
        logger.info(
            "Submitting quiz answer",
            extra={
                "submission_url": submission_url,
                "quiz_url": quiz_url,
                "payload_bytes": len(encoded_body),
            },
        )
        response = await http_client.post(
            submission_url,
            content=encoded_body,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        try:
            payload_json = response.json()
        except json.JSONDecodeError as exc:
            raise SubmissionError(
                f"Submission endpoint returned non-JSON payload: {response.text}"
            ) from exc

        result = SubmissionResult.from_response(response.status_code, payload_json)
        logger.info(
            "Submission acknowledged",
            extra={
                "submission_url": submission_url,
                "status_code": response.status_code,
                "correct": result.correct,
                "next_url": result.next_url,
            },
        )
        return result
    except httpx.HTTPError as exc:
        raise SubmissionError(f"Failed to submit answer: {exc}") from exc
    finally:
        if owns_client:
            await http_client.aclose()


__all__ = [
    "SubmissionResult",
    "SubmissionError",
    "build_submission_payload",
    "submit_answer",
]
