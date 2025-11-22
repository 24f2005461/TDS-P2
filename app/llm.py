from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings, get_settings
from app.logging_utils import get_logger

logger = get_logger(__name__)


class LLMConfigurationError(RuntimeError):
    """Raised when the application cannot reach the LLM gateway due to misconfiguration."""


class LLMAPIError(RuntimeError):
    """Raised when the upstream LLM provider returns an unexpected payload."""


@dataclass(slots=True)
class TaskParseResult:
    """Structured representation of a quiz task parsed from HTML."""

    question: Optional[str]
    submission_url: Optional[str]
    answer_format: Optional[str]
    resources: List[str] = field(default_factory=list)
    instructions: Optional[str] = None
    reasoning: Optional[str] = None
    payload_template: Optional[Dict[str, Any]] = None
    quiz_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "submission_url": self.submission_url,
            "answer_format": self.answer_format,
            "resources": self.resources,
            "instructions": self.instructions,
            "reasoning": self.reasoning,
            "payload_template": self.payload_template,
            "quiz_url": self.quiz_url,
        }


class AsyncRateLimiter:
    """Simple async rate limiter combining concurrency + minimum spacing."""

    def __init__(
        self, max_concurrent: int = 3, min_interval_seconds: float = 0.2
    ) -> None:
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent))
        self._min_interval = max(0.0, min_interval_seconds)
        self._lock = asyncio.Lock()
        self._last_request_timestamp: float = 0.0

    async def __aenter__(self) -> "AsyncRateLimiter":
        await self._semaphore.acquire()
        if self._min_interval > 0:
            async with self._lock:
                now = asyncio.get_event_loop().time()
                delta = now - self._last_request_timestamp
                if delta < self._min_interval:
                    await asyncio.sleep(self._min_interval - delta)
                self._last_request_timestamp = asyncio.get_event_loop().time()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._semaphore.release()


class AIPipeClient:
    """Async HTTP client targeting the AIPipe OpenRouter + Gemini gateways."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        timeout: Optional[float] = None,
        client: Optional[httpx.AsyncClient] = None,
        rate_limiter: Optional[AsyncRateLimiter] = None,
    ) -> None:
        self.settings = settings or get_settings()
        if not self.settings.aipipe_api_key:
            raise LLMConfigurationError("AIPIPE_API_KEY is required for LLM calls.")
        self._timeout = timeout or self.settings.request_timeout_seconds
        self._client = client or httpx.AsyncClient(timeout=self._timeout)
        self._owns_client = client is None
        self._rate_limiter = rate_limiter or AsyncRateLimiter()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "AIPipeClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.aipipe_api_key}",
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        url: str,
        *,
        json_payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> httpx.Response:
        merged_headers = dict(self._auth_headers())
        if headers:
            merged_headers.update(headers)

        retryer = AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, max=4),
            retry=retry_if_exception_type(httpx.HTTPError),
        )

        async for attempt in retryer:
            with attempt:
                limiter = self._rate_limiter
                if limiter is not None:
                    async with limiter:
                        response = await self._client.request(
                            method,
                            url,
                            json=json_payload,
                            headers=merged_headers,
                        )
                else:
                    response = await self._client.request(
                        method,
                        url,
                        json=json_payload,
                        headers=merged_headers,
                    )
                response.raise_for_status()
                return response

        raise RuntimeError(
            "Retry loop exited unexpectedly without returning a response."
        )

    async def list_models(self) -> List[str]:
        response = await self._request("GET", self.settings.aipipe_models_url)
        payload = response.json()
        models = payload.get("data") or payload.get("models") or []
        return [model.get("id") for model in models if model.get("id")]

    async def chat_completion(
        self,
        *,
        model: str,
        messages: Iterable[Dict[str, Any]],
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens

        response = await self._request(
            "POST",
            self.settings.aipipe_chat_completions_url,
            json_payload=body,
        )
        return response.json()

    async def gemini_generate_content(
        self,
        *,
        model: str,
        contents: List[Dict[str, Any]],
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        url = f"{self.settings.aipipe_gemini_models_url}/{model}:generateContent"
        body = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        response = await self._request("POST", url, json_payload=body)
        return response.json()


_PARSE_SCHEMA_PROMPT = (
    "Extract the quiz instructions from rendered HTML. Respond ONLY with minified JSON "
    "matching this schema:\n"
    '{"question":"string|null","submission_url":"string|null","answer_format":"string|null",'
    '"resources":["..."],"instructions":"string|null","reasoning":"string|null",'
    '"payload_template":{"key":"value"}|null}\n'
    "- Never invent hostnames.\n"
    "- Include every downloadable resource (PDF, CSV, JSON, ZIP, etc.).\n"
    "- Describe the expected answer format succinctly.\n"
    "- CRITICAL: If the page shows an example JSON payload to POST (often in <pre> tags or code blocks), "
    "extract it exactly as payload_template. This is the required submission structure.\n"
    "- The payload_template should include all keys shown in the example (email, secret, url, answer, etc.).\n"
    "- Use null for any missing field."
)


def _select_model(settings: Settings, override: Optional[str]) -> str:
    if override:
        return override
    if settings.llm_default_model:
        return settings.llm_default_model
    if settings.llm_allowed_models:
        return settings.llm_allowed_models[0]
    raise LLMConfigurationError("No default LLM model configured.")


def _extract_message_content(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not choices:
        raise LLMAPIError("LLM completion did not return any choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")

    if isinstance(content, list):
        fragments = [part.get("text") for part in content if isinstance(part, dict)]
        content = "".join(fragment for fragment in fragments if fragment)

    if not isinstance(content, str):
        raise LLMAPIError("LLM response content is not textual.")

    return content.strip()


def _safe_json_loads(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMAPIError(f"LLM response was not valid JSON: {raw}") from exc


_RESOURCE_PATTERN = re.compile(
    r"https?://[^\s\"'<>]+?\.(?:pdf|csv|json|xlsx?|tsv|zip)",
    flags=re.IGNORECASE,
)
_SUBMIT_PATTERN = re.compile(
    r"https?://[^\s\"'<>]*submit[^\s\"'<>]*",
    flags=re.IGNORECASE,
)


def _heuristic_resources(html: str) -> List[str]:
    return sorted({match.group(0) for match in _RESOURCE_PATTERN.finditer(html)})


def _heuristic_submission_url(html: str) -> Optional[str]:
    match = _SUBMIT_PATTERN.search(html)
    return match.group(0) if match else None


def _fallback_parse(html: str, quiz_url: Optional[str] = None) -> TaskParseResult:
    return TaskParseResult(
        question=None,
        submission_url=_heuristic_submission_url(html),
        answer_format=None,
        resources=_heuristic_resources(html),
        instructions="LLM unavailable; fallback heuristics applied.",
        reasoning="LLM request failed; heuristics extracted limited data.",
        payload_template=None,
        quiz_url=quiz_url,
    )


async def parse_task(
    html_content: str,
    *,
    quiz_url: Optional[str] = None,
    model: Optional[str] = None,
    client: Optional[AIPipeClient] = None,
) -> TaskParseResult:
    """Invoke the LLM to extract structured quiz metadata."""
    settings = get_settings()
    selected_model = _select_model(settings, model)
    owns_client = False
    agent = client

    if agent is None:
        agent = AIPipeClient(settings=settings)
        owns_client = True

    try:
        messages = [
            {"role": "system", "content": _PARSE_SCHEMA_PROMPT},
            {
                "role": "user",
                "content": (
                    "Parse the following rendered HTML/text and emit the JSON structure:\n"
                    f"{html_content}"
                ),
            },
        ]
        candidate_models = [selected_model]
        fallback_model = settings.llm_fallback_model
        if fallback_model and fallback_model not in candidate_models:
            candidate_models.append(fallback_model)

        for candidate in candidate_models:
            try:
                completion = await agent.chat_completion(
                    model=candidate,
                    messages=messages,
                    temperature=0.1,
                    max_output_tokens=600,
                )
                content = _extract_message_content(completion)
                data = _safe_json_loads(content)

                return TaskParseResult(
                    question=data.get("question"),
                    submission_url=data.get("submission_url"),
                    answer_format=data.get("answer_format"),
                    resources=[str(item) for item in data.get("resources", []) if item],
                    instructions=data.get("instructions"),
                    reasoning=data.get("reasoning"),
                    payload_template=data.get("payload_template"),
                    quiz_url=quiz_url,
                )
            except (LLMConfigurationError, LLMAPIError, httpx.HTTPError) as exc:
                logger.warning(
                    "LLM parsing failed with model %s: %s",
                    candidate,
                    exc,
                )
                continue

        logger.warning(
            "All configured LLM models (%s) failed; falling back to heuristics.",
            ", ".join(candidate_models),
        )
        return _fallback_parse(html_content, quiz_url=quiz_url)
    finally:
        if owns_client and agent is not None:
            await agent.close()


__all__ = [
    "AIPipeClient",
    "AsyncRateLimiter",
    "TaskParseResult",
    "parse_task",
    "LLMConfigurationError",
    "LLMAPIError",
]
