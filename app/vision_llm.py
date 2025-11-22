"""
Nemotron Nano 2 VL multimodal reasoning helper.

This module exposes a lightweight async client plus convenience helpers for
escalating from the Phase 9 heuristics to NVIDIA's Nemotron Nano 2 VL model
(`nvidia/nemotron-nano-12b-v2-vl:free`). The client wraps the same OpenRouter-style
gateway used elsewhere in the stack, packaging multiple images (or sampled video
frames) alongside succinct instructions so the model can perform OCR-style
extraction, chart reasoning, or document intelligence.

Key goals:
* Reuse the existing AIPipe authentication flow and configuration knobs.
* Respect sampling / frame budgets defined in `Settings`.
* Provide consistent, structured results that the agent/orchestrator can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import httpx

from app import data_tools
from app.config import Settings, get_settings
from app.logging_utils import get_logger

logger = get_logger(__name__)


class VisionLLMError(RuntimeError):
    """Raised when Nemotron reasoning fails."""


@dataclass(slots=True)
class VisionLLMObservation:
    """Single textual observation returned by the multimodal model."""

    role: str
    content: str
    raw: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content, "raw": dict(self.raw)}


@dataclass(slots=True)
class VisionLLMResult:
    """Structured representation of a Nemotron reasoning pass."""

    answer: str
    observations: List[VisionLLMObservation] = field(default_factory=list)
    model: str | None = None
    usage: MutableMapping[str, Any] = field(default_factory=dict)
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "observations": [obs.to_dict() for obs in self.observations],
            "model": self.model,
            "usage": dict(self.usage),
            "metadata": dict(self.metadata),
        }


def _ensure_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        raise VisionLLMError(f"Media file not found: {candidate}")
    if not candidate.is_file():
        raise VisionLLMError(f"Media path is not a file: {candidate}")
    return candidate


def _encode_image_to_data_url(path: Path) -> str:
    """Encode an image file as a data URL suitable for OpenRouter image inputs."""
    b64 = data_tools.image_to_base64(path)
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "image/png" if suffix == "png" else f"image/{suffix}"
    return f"data:{mime};base64,{b64}"


def _truncate_frames(paths: Sequence[Path], *, limit: int) -> List[Path]:
    if len(paths) <= limit:
        return list(paths)
    stride = max(1, len(paths) // limit)
    sampled: List[Path] = []
    for idx, frame in enumerate(paths):
        if len(sampled) >= limit:
            break
        if idx % stride == 0:
            sampled.append(frame)
    return sampled


class NemotronVisionClient:
    """
    Async helper that sends multimodal prompts to Nemotron Nano 2 VL.

    The client mirrors the semantics of the textual LLM gateway but focuses on
    reasoning over batches of images (e.g., sampled video frames, PDF page
    snapshots, or high-resolution charts).
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if not self.settings.aipipe_api_key:
            raise VisionLLMError(
                "AIPIPE_API_KEY must be configured to invoke the Nemotron vision model."
            )
        timeout = self.settings.vision_request_timeout_seconds
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = http_client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> NemotronVisionClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.aipipe_api_key}",
            "Content-Type": "application/json",
        }

    async def reason(
        self,
        *,
        question: str,
        media_paths: Sequence[str | Path],
        instructions: str | None = None,
        hints: Sequence[str] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> VisionLLMResult:
        """
        Execute a Nemotron reasoning request against the provided media assets.

        Args:
            question: The primary question/task for the model.
            media_paths: Ordered collection of image/frame paths.
            instructions: Optional system-style instructions (e.g., answer format).
            hints: Optional list of bullet hints extracted from heuristics.
            temperature: Override for sampling temperature.
            max_output_tokens: Override for output token cap.
            extra_metadata: Additional metadata forwarded to the provider.
        """
        if not media_paths:
            raise VisionLLMError("No media assets were provided for vision reasoning.")

        resolved_paths = [_ensure_path(path) for path in media_paths]
        sampled_paths = _truncate_frames(
            resolved_paths, limit=self.settings.vision_max_frames
        )
        if len(sampled_paths) < len(resolved_paths):
            logger.info(
                "Vision frame budget enforced",
                extra={"requested": len(resolved_paths), "sampled": len(sampled_paths)},
            )

        image_contents = [
            {
                "type": "image_url",
                "image_url": {"url": _encode_image_to_data_url(frame)},
            }
            for frame in sampled_paths
        ]

        user_content: List[Dict[str, Any]] = []

        preamble_parts: List[str] = [question.strip()]
        if instructions:
            preamble_parts.append(f"Constraints: {instructions.strip()}")
        if hints:
            bullets = "\n".join(f"- {hint.strip()}" for hint in hints if hint.strip())
            if bullets:
                preamble_parts.append("Hints:\n" + bullets)
        instruction_block = "\n\n".join(part for part in preamble_parts if part)
        user_content.append({"type": "text", "text": instruction_block})
        user_content.extend(image_contents)

        body: Dict[str, Any] = {
            "model": self.settings.vision_model_name,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": temperature or self.settings.vision_temperature,
        }
        max_tokens = max_output_tokens or self.settings.vision_max_output_tokens
        if max_tokens:
            body["max_output_tokens"] = max_tokens
        if extra_metadata:
            body.setdefault("metadata", {}).update(extra_metadata)

        response = await self._post_json(
            url=f"{str(self.settings.vision_model_endpoint).rstrip('/')}/chat/completions",
            payload=body,
        )
        return self._parse_response(response, media_count=len(sampled_paths))

    async def _post_json(self, *, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = await self._client.post(url, headers=self._headers(), json=payload)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _parse_response(
        payload: Mapping[str, Any], *, media_count: int
    ) -> VisionLLMResult:
        choices = payload.get("choices")
        if not choices:
            raise VisionLLMError("Vision LLM response did not include any choices.")
        primary = choices[0]
        message = primary.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            text = " ".join(
                block.get("text", "") for block in content if isinstance(block, Mapping)
            ).strip()
        else:
            text = str(content).strip() if content else ""

        observations = [
            VisionLLMObservation(
                role=message.get("role", "assistant"),
                content=text,
                raw=message,
            )
        ]
        usage = payload.get("usage") or {}
        metadata = {
            "media_processed": media_count,
            "finish_reason": primary.get("finish_reason"),
        }
        return VisionLLMResult(
            answer=text,
            observations=observations,
            model=payload.get("model"),
            usage=dict(usage),
            metadata=metadata,
        )


async def reason_over_media(
    *,
    question: str,
    media_paths: Sequence[str | Path],
    instructions: str | None = None,
    hints: Sequence[str] | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    client: NemotronVisionClient | None = None,
) -> VisionLLMResult:
    """
    Convenience wrapper that owns the NemotronVisionClient lifecycle.

    This is useful for one-off calls from tools or orchestration layers that do
    not need to retain the client between requests.
    """
    owns_client = False
    agent = client
    if agent is None:
        agent = NemotronVisionClient()
        owns_client = True
    try:
        return await agent.reason(
            question=question,
            media_paths=media_paths,
            instructions=instructions,
            hints=hints,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    finally:
        if owns_client and agent is not None:
            await agent.close()


__all__ = [
    "NemotronVisionClient",
    "VisionLLMError",
    "VisionLLMObservation",
    "VisionLLMResult",
    "reason_over_media",
]
