"""
Phase 9 HTTP API integration helpers.

This module provides small, dependency-light utilities for interacting with
third-party REST APIs from within the quiz solver workflow.  It focuses on
repeatable request construction, consistent logging, JSON parsing safeguards,
and convenient async wrappers around httpx's client.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional

import httpx

from app.config import get_settings
from app.logging_utils import get_logger

logger = get_logger(__name__)


class APIIntegrationError(RuntimeError):
    """Raised when an upstream HTTP API call fails or returns malformed data."""


@dataclass(slots=True)
class APIRequest:
    """Structured HTTP request descriptor."""

    method: str
    url: Optional[str] = None
    path: Optional[str] = None
    headers: MutableMapping[str, str] = field(default_factory=dict)
    params: Optional[Mapping[str, Any]] = None
    json_body: Any = None
    data: Any = None
    auth_token: Optional[str] = None
    timeout: Optional[float] = None


@dataclass(slots=True)
class APIResponse:
    """Normalized response payload returned to callers."""

    status_code: int
    headers: Mapping[str, str]
    text: str
    json_data: Any = None

    def json(self) -> Any:
        """Return the parsed JSON payload, raising if unavailable."""
        if self.json_data is None:
            raise APIIntegrationError("Response did not contain JSON data.")
        return self.json_data


def _merge_headers(
    base: Optional[Mapping[str, str]],
    overrides: Optional[Mapping[str, str]],
) -> MutableMapping[str, str]:
    merged: MutableMapping[str, str] = {}
    if base:
        merged.update(base)
    if overrides:
        merged.update(overrides)
    return merged


class AsyncAPIClient:
    """Thin async wrapper around httpx.AsyncClient with project defaults."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self._base_url = base_url
        self._default_headers = dict(default_headers or {})
        self._timeout = timeout or settings.request_timeout_seconds
        # Only pass base_url to httpx if it's not None
        client_kwargs: Dict[str, Any] = {"timeout": self._timeout}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self._client = httpx.AsyncClient(**client_kwargs)
        self._closed = False

    async def __aenter__(self) -> "AsyncAPIClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    @property
    def base_url(self) -> Optional[str]:
        return self._base_url

    async def close(self) -> None:
        if not self._closed:
            await self._client.aclose()
            self._closed = True

    async def send(self, request: APIRequest) -> APIResponse:
        if self._closed:
            raise APIIntegrationError("API client has already been closed.")

        url = self._resolve_url(request)
        headers = _merge_headers(self._default_headers, request.headers)
        if request.auth_token:
            headers.setdefault("Authorization", f"Bearer {request.auth_token}")

        try:
            logger.debug(
                "Dispatching API request",
                extra={
                    "method": request.method.upper(),
                    "url": url,
                    "params": request.params or {},
                },
            )
            response = await self._client.request(
                request.method.upper(),
                url,
                headers=headers,
                params=request.params,
                json=request.json_body,
                data=request.data,
                timeout=request.timeout or self._timeout,
            )
        except httpx.HTTPError as exc:
            raise APIIntegrationError(f"HTTP error while calling {url}: {exc}") from exc

        json_payload = self._safe_json(response)
        api_response = APIResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            text=response.text,
            json_data=json_payload,
        )

        logger.info(
            "API request complete",
            extra={"url": url, "status_code": response.status_code},
        )

        if response.is_error:
            raise APIIntegrationError(
                f"Upstream API returned {response.status_code}: {response.text[:200]}"
            )

        return api_response

    def _resolve_url(self, request: APIRequest) -> str:
        if request.url:
            return request.url
        if request.path and self._base_url:
            return f"{self._base_url.rstrip('/')}/{request.path.lstrip('/')}"
        if request.path:
            raise APIIntegrationError(
                "Cannot resolve relative path without a configured base_url."
            )
        raise APIIntegrationError("APIRequest must include either url or path.")

    @staticmethod
    def _safe_json(response: httpx.Response) -> Any:
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type.lower():
            return None
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise APIIntegrationError(f"Failed to decode JSON response: {exc}") from exc


def extract_instruction_headers(
    headers: Mapping[str, str], *, prefix: str = "x-quiz-instruction-"
) -> dict[str, str]:
    """
    Parse custom quiz instruction headers into a normalized dictionary.

    Example:
        X-Quiz-Instruction-Answer-Format: integer
    becomes:
        {"answer_format": "integer"}
    """
    if not headers:
        return {}

    prefix_lower = prefix.lower()
    instructions: dict[str, str] = {}
    for raw_key, raw_value in headers.items():
        if not isinstance(raw_key, str):
            continue
        lower_key = raw_key.lower()
        if not lower_key.startswith(prefix_lower):
            continue
        field_name = lower_key[len(prefix_lower) :].replace("-", "_").strip()
        if not field_name:
            continue
        instructions[field_name] = "" if raw_value is None else str(raw_value)
    return instructions


def parse_instruction_text(
    text: str,
    *,
    pair_delimiter: str = ";",
    kv_delimiter: str = ":",
) -> dict[str, str]:
    """
    Split an instruction header/body into normalized key/value pairs.

    Example:
        "answer_format: integer; precision: 2"
    becomes {"answer_format": "integer", "precision": "2"}.
    """
    if not text:
        return {}

    components = [
        segment.strip()
        for segment in text.split(pair_delimiter)
        if segment and segment.strip()
    ]
    parsed: dict[str, str] = {}
    for component in components:
        if kv_delimiter in component:
            key, value = component.split(kv_delimiter, 1)
            parsed[key.strip().lower().replace(" ", "_")] = value.strip()
        else:
            parsed[component.strip().lower().replace(" ", "_")] = ""
    return parsed


async def call_json_api(
    method: str,
    url: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
    json_body: Any = None,
    headers: Optional[Mapping[str, str]] = None,
    auth_token: Optional[str] = None,
    timeout: Optional[float] = None,
) -> APIResponse:
    """
    Fire-and-forget helper for one-off JSON HTTP calls.

    This convenience function instantiates a temporary AsyncAPIClient,
    sends the request, and closes the client automatically.
    """
    client = AsyncAPIClient()
    try:
        request = APIRequest(
            method=method,
            url=url,
            params=params,
            json_body=json_body,
            headers=dict(headers or {}),
            auth_token=auth_token,
            timeout=timeout,
        )
        return await client.send(request)
    finally:
        await client.close()


__all__ = [
    "APIIntegrationError",
    "APIRequest",
    "APIResponse",
    "AsyncAPIClient",
    "extract_instruction_headers",
    "parse_instruction_text",
    "call_json_api",
]
