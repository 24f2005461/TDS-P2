import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.vision import (
    VisionAnalysisResult,
    _build_hints,
    _should_escalate,
    analyze_image,
)
from app.vision_llm import VisionLLMResult


def _create_gradient_image(path: Path, size: int = 64) -> Path:
    """Generate a simple grayscale gradient image for testing."""
    data = np.linspace(0, 255, size * size, dtype=np.uint8).reshape(size, size)
    image = Image.fromarray(data, mode="L").convert("RGB")
    image.save(path)
    return path


def _create_contrast_image(path: Path, size: int = 60) -> Path:
    """Generate an image with alternating black/white stripes to trigger regions."""
    data = np.zeros((size, size, 3), dtype=np.uint8)
    stripe_width = 5
    for i in range(0, size, stripe_width * 2):
        data[:, i : i + stripe_width] = 255
    image = Image.fromarray(data, mode="RGB")
    image.save(path)
    return path


@pytest.mark.anyio
async def test_analyze_image_returns_summary(tmp_path: Path):
    image_path = _create_gradient_image(tmp_path / "gradient.png")
    result = await analyze_image(image_path)

    assert isinstance(result, VisionAnalysisResult)
    assert result.summary.width > 0 and result.summary.height > 0
    assert 0 <= result.summary.brightness <= 1
    assert 0 <= result.summary.contrast <= 1
    assert math.isfinite(result.summary.entropy)
    assert isinstance(result.annotations["path"], str)
    assert result.annotations["region_count"] >= 0


@pytest.mark.anyio
async def test_analyze_image_detects_regions(tmp_path: Path):
    image_path = _create_contrast_image(tmp_path / "stripes.png")
    result = await analyze_image(image_path, grid_shape=(3, 3))

    assert result.annotations["region_count"] > 0
    assert any(region.label == "text_candidate" for region in result.regions)
    assert all(0.0 <= region.score <= 1.0 for region in result.regions)


@pytest.mark.anyio
async def test_analyze_image_honors_grid_shape(tmp_path: Path):
    image_path = _create_gradient_image(tmp_path / "gradient2.png", size=48)
    result = await analyze_image(image_path, grid_shape=(2, 2))

    # Expect at most rows*cols regions from grid sampling
    assert len(result.regions) <= 4
    for region in result.regions:
        x0, y0, x1, y1 = region.bbox
        assert 0 <= x0 < x1 <= result.summary.width
        assert 0 <= y0 < y1 <= result.summary.height


@pytest.mark.anyio
async def test_analyze_image_no_escalation_without_question(tmp_path: Path):
    """Test that no escalation occurs when question is not provided."""
    image_path = _create_contrast_image(tmp_path / "high_entropy.png")
    result = await analyze_image(image_path)

    assert result.llm_result is None
    assert result.annotations.get("escalated") is False


@pytest.mark.anyio
async def test_analyze_image_escalation_with_high_entropy(tmp_path: Path, monkeypatch):
    """Test escalation triggers when entropy exceeds threshold."""
    image_path = _create_contrast_image(tmp_path / "high_entropy.png")

    # Mock settings to lower threshold - use 0.5 to ensure it triggers
    mock_settings = MagicMock()
    mock_settings.vision_entropy_trigger = 0.5  # Very low threshold to ensure trigger
    mock_settings.vision_region_score_threshold = 0.99  # Very high, won't trigger

    import app.vision

    monkeypatch.setattr(app.vision, "_SETTINGS", mock_settings)

    # Mock reason_over_media
    mock_result = VisionLLMResult(
        answer="This is a high contrast striped pattern",
        observations=[],
        model="test-model",
    )

    async def mock_reason(**kwargs):
        return mock_result

    with patch("app.vision_llm.reason_over_media", new=mock_reason):
        result = await analyze_image(image_path, question="What is in this image?")

    assert result.llm_result is not None
    assert result.annotations.get("escalated") is True
    assert (
        result.annotations.get("nemotron_answer")
        == "This is a high contrast striped pattern"
    )


@pytest.mark.anyio
async def test_analyze_image_escalation_with_high_region_score(
    tmp_path: Path, monkeypatch
):
    """Test escalation triggers when region score exceeds threshold."""
    image_path = _create_contrast_image(tmp_path / "regions.png")

    # Mock settings
    mock_settings = MagicMock()
    mock_settings.vision_entropy_trigger = 999.0  # Won't trigger
    mock_settings.vision_region_score_threshold = 0.01  # Very low, will trigger

    import app.vision

    monkeypatch.setattr(app.vision, "_SETTINGS", mock_settings)

    # Mock reason_over_media
    mock_result = VisionLLMResult(
        answer="Detected text regions",
        observations=[],
        model="test-model",
    )

    async def mock_reason(**kwargs):
        return mock_result

    with patch("app.vision_llm.reason_over_media", new=mock_reason):
        result = await analyze_image(image_path, question="Describe this image")

    assert result.llm_result is not None
    assert result.annotations.get("escalated") is True


@pytest.mark.anyio
async def test_analyze_image_no_escalation_below_thresholds(
    tmp_path: Path, monkeypatch
):
    """Test no escalation when both thresholds are not exceeded."""
    image_path = _create_gradient_image(tmp_path / "low_signal.png")

    # Mock settings with high thresholds
    mock_settings = MagicMock()
    mock_settings.vision_entropy_trigger = 999.0
    mock_settings.vision_region_score_threshold = 0.99

    import app.vision

    monkeypatch.setattr(app.vision, "_SETTINGS", mock_settings)

    result = await analyze_image(image_path, question="What is this?")

    assert result.llm_result is None
    assert result.annotations.get("escalated") is False


@pytest.mark.anyio
async def test_analyze_image_escalation_failure_handled(tmp_path: Path, monkeypatch):
    """Test that escalation failures are caught and logged."""
    image_path = _create_contrast_image(tmp_path / "error_test.png")

    # Mock settings to trigger escalation
    mock_settings = MagicMock()
    mock_settings.vision_entropy_trigger = 2.0
    mock_settings.vision_region_score_threshold = 0.01

    import app.vision

    monkeypatch.setattr(app.vision, "_SETTINGS", mock_settings)

    # Mock reason_over_media to raise exception
    async def mock_reason_error(**kwargs):
        raise RuntimeError("API timeout")

    with patch("app.vision_llm.reason_over_media", new=mock_reason_error):
        result = await analyze_image(image_path, question="What is this?")

    # Should not raise, but should record error
    assert result.llm_result is None
    assert "escalation_error" in result.annotations
    assert "API timeout" in result.annotations["escalation_error"]


@pytest.mark.anyio
async def test_analyze_image_includes_hints_in_escalation(tmp_path: Path, monkeypatch):
    """Test that heuristic hints are passed to Nemotron."""
    image_path = _create_contrast_image(tmp_path / "hints_test.png")

    # Mock settings
    mock_settings = MagicMock()
    mock_settings.vision_entropy_trigger = 2.0
    mock_settings.vision_region_score_threshold = 0.01

    import app.vision

    monkeypatch.setattr(app.vision, "_SETTINGS", mock_settings)

    # Capture hints passed to reason_over_media
    captured_kwargs = {}

    async def capture_reason(**kwargs):
        captured_kwargs.update(kwargs)
        return VisionLLMResult(answer="Test", observations=[], model="test")

    with patch("app.vision_llm.reason_over_media", new=capture_reason):
        await analyze_image(
            image_path,
            question="Describe this",
            instructions="Answer concisely",
        )

    assert "hints" in captured_kwargs
    assert isinstance(captured_kwargs["hints"], list)
    assert len(captured_kwargs["hints"]) > 0
    # Should include dimension and entropy info
    assert any("dimensions" in h.lower() for h in captured_kwargs["hints"])
    assert captured_kwargs["instructions"] == "Answer concisely"


def test_should_escalate_no_question():
    """Test _should_escalate returns False when no question provided."""
    from app.vision import VisionSummary

    summary = VisionSummary(
        width=100,
        height=100,
        mode="RGB",
        aspect_ratio=1.0,
        dominant_colors=[],
        brightness=0.5,
        contrast=0.5,
        entropy=10.0,
    )
    regions = []

    result = _should_escalate(summary, regions, question=None)
    assert result is False


def test_should_escalate_high_entropy(monkeypatch):
    """Test _should_escalate triggers on high entropy."""
    from app.vision import VisionSummary

    mock_settings = MagicMock()
    mock_settings.vision_entropy_trigger = 4.0
    mock_settings.vision_region_score_threshold = 0.5

    import app.vision

    monkeypatch.setattr(app.vision, "_SETTINGS", mock_settings)

    summary = VisionSummary(
        width=100,
        height=100,
        mode="RGB",
        aspect_ratio=1.0,
        dominant_colors=[],
        brightness=0.5,
        contrast=0.5,
        entropy=5.0,
    )

    result = _should_escalate(summary, [], question="What is this?")
    assert result is True


def test_should_escalate_high_region_score(monkeypatch):
    """Test _should_escalate triggers on high region score."""
    from app.vision import VisionRegion, VisionSummary

    mock_settings = MagicMock()
    mock_settings.vision_entropy_trigger = 999.0
    mock_settings.vision_region_score_threshold = 0.2

    import app.vision

    monkeypatch.setattr(app.vision, "_SETTINGS", mock_settings)

    summary = VisionSummary(
        width=100,
        height=100,
        mode="RGB",
        aspect_ratio=1.0,
        dominant_colors=[],
        brightness=0.5,
        contrast=0.5,
        entropy=2.0,
    )
    region = VisionRegion(
        label="text_candidate", bbox=(0, 0, 50, 50), score=0.3, metadata={}
    )

    result = _should_escalate(summary, [region], question="Analyze")
    assert result is True


def test_build_hints():
    """Test _build_hints constructs proper hint strings."""
    from app.vision import VisionRegion, VisionSummary

    summary = VisionSummary(
        width=640,
        height=480,
        mode="RGB",
        aspect_ratio=1.33,
        dominant_colors=["#ff0000", "#00ff00", "#0000ff"],
        brightness=0.6,
        contrast=0.4,
        entropy=5.2,
    )
    region1 = VisionRegion("text_candidate", (0, 0, 100, 100), 0.8, {})
    region2 = VisionRegion("text_candidate", (100, 0, 200, 100), 0.6, {})

    hints = _build_hints(summary, [region1, region2])

    assert len(hints) > 0
    assert any("640x480" in h for h in hints)
    assert any("entropy=5.2" in h.lower() for h in hints)
    assert any("#ff0000" in h for h in hints)
    assert any("text_candidate" in h.lower() for h in hints)
