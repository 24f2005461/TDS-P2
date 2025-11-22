"""
Centralized application configuration powered by Pydantic settings.

This module loads environment variables once at startup, surfaces strongly-typed
settings to the rest of the codebase, and documents the LLM infrastructure
constraints (AIPipe gateway + approved model families).
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, cast

from pydantic import AliasChoices, AnyHttpUrl, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the TDS quiz solver service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(
        default="TDS Quiz Solver API",
        validation_alias=AliasChoices("APP_NAME"),
    )
    environment: str = Field(
        default="development",
        validation_alias=AliasChoices("ENVIRONMENT"),
    )
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("LOG_LEVEL"),
    )

    email: str = Field(..., validation_alias=AliasChoices("EMAIL"))
    secret: str = Field(..., validation_alias=AliasChoices("SECRET"))

    aipipe_endpoint: AnyHttpUrl = Field(
        default=cast(AnyHttpUrl, "https://aipipe.org/openrouter/v1"),
        description="OpenRouter-compatible base URL for routing LLM calls through AIPipe",
        validation_alias=AliasChoices("AIPIPE_ENDPOINT"),
    )
    aipipe_geminiv1beta_endpoint: AnyHttpUrl = Field(
        default=cast(AnyHttpUrl, "https://aipipe.org/geminiv1beta"),
        description="Base URL for Gemini v1beta routes proxied via AIPipe",
        validation_alias=AliasChoices("AIPIPE_GEMINIV1BETA_ENDPOINT"),
    )
    aipipe_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AIPIPE_API_KEY"),
    )

    request_timeout_seconds: float = Field(
        default=10.0,
        validation_alias=AliasChoices("REQUEST_TIMEOUT_SECONDS"),
    )
    max_payload_size_mb: int = Field(
        default=1,
        validation_alias=AliasChoices("MAX_PAYLOAD_SIZE_MB"),
    )
    downloader_request_timeout_seconds: float = Field(
        default=30.0,
        validation_alias=AliasChoices("DOWNLOADER_REQUEST_TIMEOUT_SECONDS"),
        description="Seconds to wait for each resource download before aborting.",
    )
    downloader_max_file_size_mb: int = Field(
        default=50,
        validation_alias=AliasChoices("DOWNLOADER_MAX_FILE_SIZE_MB"),
        description="Maximum size allowed for any single downloaded artifact.",
    )
    downloader_concurrency: int = Field(
        default=3,
        validation_alias=AliasChoices("DOWNLOADER_CONCURRENCY"),
        description="Number of files the downloader may fetch in parallel.",
    )

    playwright_headless: bool = Field(
        default=True,
        validation_alias=AliasChoices("PLAYWRIGHT_HEADLESS"),
        description="Toggle headless mode for Playwright sessions.",
    )
    playwright_wait_selector: str | None = Field(
        default="#result",
        validation_alias=AliasChoices("PLAYWRIGHT_WAIT_SELECTOR"),
        description="CSS selector to await before scraping; set to None to skip explicit waits.",
    )
    playwright_viewport_width: int = Field(
        default=1280,
        validation_alias=AliasChoices("PLAYWRIGHT_VIEWPORT_WIDTH"),
        description="Viewport width used when instantiating Playwright contexts.",
    )
    playwright_viewport_height: int = Field(
        default=720,
        validation_alias=AliasChoices("PLAYWRIGHT_VIEWPORT_HEIGHT"),
        description="Viewport height used when instantiating Playwright contexts.",
    )
    playwright_navigation_timeout_seconds: float = Field(
        default=20.0,
        validation_alias=AliasChoices("PLAYWRIGHT_NAVIGATION_TIMEOUT_SECONDS"),
        description="Seconds to wait for page navigation before aborting.",
    )
    playwright_wait_timeout_seconds: float = Field(
        default=20.0,
        validation_alias=AliasChoices("PLAYWRIGHT_WAIT_TIMEOUT_SECONDS"),
        description="Seconds to wait for readiness selectors before giving up.",
    )

    llm_default_model: str = Field(
        default="x-ai/grok-4.1-fast:free",
        description="Primary lightweight model available today",
        validation_alias=AliasChoices("LLM_DEFAULT_MODEL"),
    )
    llm_fallback_model: str = Field(
        default="moonshotai/kimi-k2:free",
        description="Higher-capability model leveraged when the default response is insufficient",
        validation_alias=AliasChoices("LLM_FALLBACK_MODEL"),
    )
    llm_allowed_models_raw: str = Field(
        default="x-ai/grok-4.1-fast:free,moonshotai/kimi-k2:free",
        description="Comma-separated list of currently approved models via AIPipe",
        validation_alias=AliasChoices("LLM_ALLOWED_MODELS"),
    )
    llm_future_models_raw: str = Field(
        default="gemini-2.5-pro",
        description="Comma-separated list of planned models once google-genai integration lands",
        validation_alias=AliasChoices("LLM_FUTURE_MODELS"),
    )
    google_genai_project_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_GENAI_PROJECT_ID"),
    )

    vision_model_name: str = Field(
        default="nvidia/nemotron-nano-12b-v2-vl:free",
        description="Multimodal reasoning model used for escalated vision analysis.",
        validation_alias=AliasChoices("VISION_MODEL_NAME"),
    )
    vision_model_endpoint: AnyHttpUrl = Field(
        default=cast(AnyHttpUrl, "https://aipipe.org/openrouter/v1"),
        description="Endpoint leveraged for Nemotron Nano 2 VL multimodal invocations.",
        validation_alias=AliasChoices("VISION_MODEL_ENDPOINT"),
    )
    vision_request_timeout_seconds: float = Field(
        default=45.0,
        description="Timeout applied to Nemotron multimodal requests.",
        validation_alias=AliasChoices("VISION_REQUEST_TIMEOUT_SECONDS"),
    )
    vision_max_frames: int = Field(
        default=24,
        description="Maximum number of frames/pages sampled via Efficient Video Sampling before escalation.",
        validation_alias=AliasChoices("VISION_MAX_FRAMES"),
    )
    vision_frame_sampling_strategy: str = Field(
        default="evs-uniform",
        description="Sampling policy for long-form media (e.g., evs-uniform, evs-keyframe).",
        validation_alias=AliasChoices("VISION_FRAME_SAMPLING_STRATEGY"),
    )
    vision_temperature: float = Field(
        default=0.1,
        description="Sampling temperature applied to Nemotron multimodal responses.",
        validation_alias=AliasChoices("VISION_TEMPERATURE"),
    )
    vision_max_output_tokens: int = Field(
        default=1024,
        description="Maximum number of tokens requested from Nemotron multimodal completions.",
        validation_alias=AliasChoices("VISION_MAX_OUTPUT_TOKENS"),
    )
    vision_region_score_threshold: float = Field(
        default=0.18,
        description="Minimum heuristic text-region score that triggers Nemotron escalation.",
        validation_alias=AliasChoices("VISION_REGION_SCORE_THRESHOLD"),
    )
    vision_entropy_trigger: float = Field(
        default=4.2,
        description="Shannon entropy threshold that signals dense textual content requiring VL reasoning.",
        validation_alias=AliasChoices("VISION_ENTROPY_TRIGGER"),
    )
    vision_max_video_seconds: int = Field(
        default=90,
        description="Cap on cumulative video seconds sampled before trimming segments via EVS.",
        validation_alias=AliasChoices("VISION_MAX_VIDEO_SECONDS"),
    )
    vision_evs_stride: int = Field(
        default=4,
        description="Frame stride applied during EVS down-sampling for long videos.",
        validation_alias=AliasChoices("VISION_EVS_STRIDE"),
    )

    @field_validator(
        "request_timeout_seconds",
        "playwright_navigation_timeout_seconds",
        "playwright_wait_timeout_seconds",
        "downloader_request_timeout_seconds",
        "vision_request_timeout_seconds",
    )
    @classmethod
    def _validate_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Timeout values must be positive.")
        return value

    @field_validator("playwright_viewport_width", "playwright_viewport_height")
    @classmethod
    def _validate_viewport_dimension(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(
                "Playwright viewport dimensions must be positive integers."
            )
        return value

    @field_validator("downloader_concurrency")
    @classmethod
    def _validate_downloader_concurrency(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Downloader concurrency must be a positive integer.")
        return value

    @field_validator("downloader_max_file_size_mb")
    @classmethod
    def _validate_downloader_file_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Downloader max file size must be positive.")
        return value

    @field_validator(
        "vision_max_frames",
        "vision_max_video_seconds",
        "vision_evs_stride",
        "vision_max_output_tokens",
    )
    @classmethod
    def _validate_vision_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Vision sampling parameters must be positive integers.")
        return value

    @field_validator("vision_temperature")
    @classmethod
    def _validate_vision_temperature(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Vision temperature must be positive.")
        return value

    @field_validator("vision_region_score_threshold")
    @classmethod
    def _validate_region_threshold(cls, value: float) -> float:
        if not 0 < value <= 1:
            raise ValueError("Vision region score threshold must be within (0, 1].")
        return value

    @field_validator("vision_entropy_trigger")
    @classmethod
    def _validate_entropy_trigger(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Vision entropy trigger must be positive.")
        return value

    @property
    def downloader_max_file_size_bytes(self) -> int:
        """Return the absolute byte ceiling for a single download."""
        return int(self.downloader_max_file_size_mb * 1024 * 1024)

    @property
    def llm_allowed_models(self) -> List[str]:
        """Return the parsed list of currently available models."""
        return self._parse_model_list(self.llm_allowed_models_raw)

    @property
    def llm_future_models(self) -> List[str]:
        """Return the parsed list of planned/experimental models."""
        return self._parse_model_list(self.llm_future_models_raw)

    @staticmethod
    def _parse_model_list(data: str) -> List[str]:
        return [item.strip() for item in data.split(",") if item.strip()]

    @staticmethod
    def _join_url(base: str, *segments: str) -> str:
        normalized = base.rstrip("/")
        tail = "/".join(segment.strip("/") for segment in segments if segment)
        return f"{normalized}/{tail}" if tail else normalized

    @property
    def aipipe_models_url(self) -> str:
        """Full URL for listing available OpenRouter models via AIPipe."""
        return self._join_url(str(self.aipipe_endpoint), "models")

    @property
    def aipipe_chat_completions_url(self) -> str:
        """Full URL for issuing OpenRouter-compatible chat completions."""
        return self._join_url(str(self.aipipe_endpoint), "chat", "completions")

    @property
    def aipipe_gemini_models_url(self) -> str:
        """Base models URL for Gemini v1beta routes."""
        return self._join_url(str(self.aipipe_geminiv1beta_endpoint), "models")

    def describe_llm_capabilities(self) -> str:
        """Human-readable summary of current and upcoming LLM options."""
        current = ", ".join(self.llm_allowed_models) or "none"
        future = ", ".join(self.llm_future_models) or "none"
        return (
            f"AIPipe default model: {self.llm_default_model} "
            f"(fallback: {self.llm_fallback_model}). "
            f"Available now: {current}. "
            f"Planned via google-genai: {future}."
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance to avoid re-reading the environment."""
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        missing_fields = sorted(
            {
                ".".join(str(part) for part in error.get("loc", []))
                for error in exc.errors()
                if error.get("type") == "missing"
            }
        )
        missing_hint = (
            ", ".join(missing_fields) if missing_fields else "required fields"
        )
        raise RuntimeError(
            "Missing required environment variables. "
            f"Set the following keys in your .env or shell environment: {missing_hint}"
        ) from exc
