"""Phase 9 vision analysis helpers.

This module introduces lightweight image reasoning primitives that can run in
restricted environments (no GPU, no heavyweight CV dependencies).  It focuses on
extracting descriptive metadata, dominant colors, and heuristic text-like
regions that help the main solver agent decide whether an image probably
contains charts, tables, or textual hints worth OCRing downstream.

Phase 9.5 extends this with escalation to Nemotron Nano 2 VL for deeper
multimodal reasoning when heuristic thresholds are exceeded.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from PIL import Image, ImageFilter

from app import data_tools
from app.config import get_settings
from app.logging_utils import get_logger

logger = get_logger(__name__)
_SETTINGS = get_settings()


@dataclass(slots=True)
class VisionRegion:
    """Represents a detected area of interest inside an image."""

    label: str
    bbox: Tuple[int, int, int, int]
    score: float
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "bbox": self.bbox,
            "score": self.score,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class VisionSummary:
    """High-level statistics for an image."""

    width: int
    height: int
    mode: str
    aspect_ratio: float
    dominant_colors: Sequence[str]
    brightness: float
    contrast: float
    entropy: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "aspect_ratio": self.aspect_ratio,
            "dominant_colors": list(self.dominant_colors),
            "brightness": self.brightness,
            "contrast": self.contrast,
            "entropy": self.entropy,
        }


@dataclass(slots=True)
class VisionAnalysisResult:
    """Aggregate result returned by :func:`analyze_image`."""

    summary: VisionSummary
    regions: List[VisionRegion]
    annotations: MutableMapping[str, Any] = field(default_factory=dict)
    llm_result: Any = None  # Optional VisionLLMResult when escalation occurs

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "summary": self.summary.to_dict(),
            "regions": [region.to_dict() for region in self.regions],
            "annotations": dict(self.annotations),
        }
        if self.llm_result is not None:
            payload["llm_result"] = self.llm_result.to_dict()
        return payload


def _load_image(path: str | Path) -> Image.Image:
    """Delegate to the shared data_tools loader for consistent logging."""
    return data_tools.load_image(path, mode="RGB")


def _prepare_image(image: Image.Image, max_edge: int = 1024) -> Image.Image:
    candidate = image.copy().convert("RGB")
    width, height = candidate.size
    longest = max(width, height)
    if longest > max_edge:
        scale = max_edge / float(longest)
        new_size = (int(width * scale), int(height * scale))
        candidate = candidate.resize(new_size, Image.Resampling.BILINEAR)
    return candidate


def _hex_color(rgb: Tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _dominant_colors(image: Image.Image, top_k: int = 4) -> List[str]:
    downscaled = image.copy()
    downscaled.thumbnail((128, 128))
    palette = downscaled.quantize(colors=top_k * 4, method=Image.Quantize.MEDIANCUT)
    # Convert palette data to list to avoid typing issues with PIL's ImagingCore
    counts = Counter(list(palette.getdata()))  # type: ignore[arg-type]
    most_common = counts.most_common(top_k)
    palette_colors = palette.getpalette()
    colors: List[str] = []
    for value, _ in most_common:
        base = value * 3
        # Ensure palette_colors is not None before subscripting
        if palette_colors is None:
            continue
        rgb = tuple(palette_colors[base : base + 3])
        colors.append(_hex_color(rgb))  # type: ignore[arg-type]
    return colors


def _compute_entropy(gray: Image.Image) -> float:
    histogram = gray.histogram()
    total = float(sum(histogram)) or 1.0
    entropy = 0.0
    for count in histogram:
        if count == 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def _summarize_image(image: Image.Image) -> VisionSummary:
    prepared = _prepare_image(image)
    gray = prepared.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    brightness = float(arr.mean() / 255.0)
    contrast = float(arr.std(ddof=0) / 255.0)
    entropy = _compute_entropy(gray)
    width, height = prepared.size
    dominant = _dominant_colors(prepared)
    summary = VisionSummary(
        width=width,
        height=height,
        mode=image.mode,
        aspect_ratio=round(width / height, 3) if height else 0.0,
        dominant_colors=dominant,
        brightness=round(brightness, 4),
        contrast=round(contrast, 4),
        entropy=round(entropy, 4),
    )
    logger.debug("Vision summary computed", extra=summary.to_dict())
    return summary


def _gradient_magnitude(region: np.ndarray) -> float:
    blurred = region.astype(np.float32)
    grad_x = np.abs(np.diff(blurred, axis=1)).mean() if blurred.shape[1] > 1 else 0.0
    grad_y = np.abs(np.diff(blurred, axis=0)).mean() if blurred.shape[0] > 1 else 0.0
    return float((grad_x + grad_y) / (2 * 255.0))


def _scan_text_like_regions(
    gray: Image.Image,
    *,
    rows: int = 4,
    cols: int = 3,
    min_score: float = 0.12,
) -> List[VisionRegion]:
    arr = np.asarray(gray, dtype=np.float32)
    height, width = arr.shape
    row_height = max(1, height // rows)
    col_width = max(1, width // cols)
    regions: List[VisionRegion] = []

    for row in range(rows):
        for col in range(cols):
            y0 = row * row_height
            y1 = height if row == rows - 1 else (row + 1) * row_height
            x0 = col * col_width
            x1 = width if col == cols - 1 else (col + 1) * col_width
            crop = arr[y0:y1, x0:x1]
            if crop.size < 25:
                continue
            local_std = float(crop.std() / 255.0)
            gradient = _gradient_magnitude(crop)
            combined_score = 0.6 * local_std + 0.4 * gradient
            if combined_score < min_score:
                continue
            metadata = {
                "mean_intensity": float(crop.mean() / 255.0),
                "std_intensity": local_std,
                "gradient": gradient,
                "grid_position": {"row": row, "col": col},
            }
            regions.append(
                VisionRegion(
                    label="text_candidate",
                    bbox=(int(x0), int(y0), int(x1), int(y1)),
                    score=round(combined_score, 4),
                    metadata=metadata,
                )
            )

    logger.debug("Detected %d candidate regions", len(regions))
    return regions


def _should_escalate(
    summary: VisionSummary,
    regions: List[VisionRegion],
    *,
    question: Optional[str] = None,
) -> bool:
    """
    Determine if heuristics warrant escalation to Nemotron VL.

    Escalation triggers when:
    - A question is provided (explicit reasoning request), AND
    - Entropy exceeds the configured threshold, OR
    - Any region score exceeds the configured threshold.
    """
    if not question:
        return False

    entropy_trigger = _SETTINGS.vision_entropy_trigger
    region_threshold = _SETTINGS.vision_region_score_threshold

    if summary.entropy >= entropy_trigger:
        logger.debug(
            "Escalation triggered by entropy",
            extra={"entropy": summary.entropy, "threshold": entropy_trigger},
        )
        return True

    high_score_regions = [r for r in regions if r.score >= region_threshold]
    if high_score_regions:
        logger.debug(
            "Escalation triggered by region scores",
            extra={
                "count": len(high_score_regions),
                "threshold": region_threshold,
            },
        )
        return True

    return False


def _build_hints(summary: VisionSummary, regions: List[VisionRegion]) -> List[str]:
    """Extract structured hints from heuristics to guide Nemotron reasoning."""
    hints: List[str] = []
    hints.append(
        f"Image dimensions: {summary.width}x{summary.height}, "
        f"brightness={summary.brightness:.2f}, contrast={summary.contrast:.2f}, "
        f"entropy={summary.entropy:.2f}"
    )
    if summary.dominant_colors:
        colors = ", ".join(summary.dominant_colors[:3])
        hints.append(f"Dominant colors: {colors}")
    if regions:
        top_regions = sorted(regions, key=lambda r: r.score, reverse=True)[:3]
        region_desc = ", ".join(f"{r.label} (score={r.score:.2f})" for r in top_regions)
        hints.append(f"Top text-like regions: {region_desc}")
    return hints


async def analyze_image(
    path: str | Path,
    *,
    grid_shape: Tuple[int, int] = (4, 3),
    top_colors: int = 4,
    blur_before_scan: bool = True,
    question: Optional[str] = None,
    instructions: Optional[str] = None,
) -> VisionAnalysisResult:
    """
    Compute a structured description of an image for downstream reasoning.

    When a question is provided and heuristic thresholds are exceeded, this
    function will escalate to Nemotron Nano 2 VL for multimodal reasoning.

    Args:
        path: Image file path.
        grid_shape: (rows, cols) grid for text-region scanning.
        top_colors: Number of dominant colors to extract.
        blur_before_scan: Apply median filter before region detection.
        question: Optional question to ask Nemotron VL (enables escalation).
        instructions: Optional constraints/format instructions for Nemotron.
    """
    image = _load_image(path)
    summary = _summarize_image(image)
    prepared = _prepare_image(image)
    gray = prepared.convert("L")
    if blur_before_scan:
        gray = gray.filter(ImageFilter.MedianFilter(size=3))

    regions = _scan_text_like_regions(
        gray,
        rows=grid_shape[0],
        cols=grid_shape[1],
    )

    annotations: Dict[str, Any] = {
        "path": str(Path(path).resolve()),
        "dominant_colors": summary.dominant_colors[:top_colors],
        "region_count": len(regions),
    }

    llm_result = None
    if _should_escalate(summary, regions, question=question):
        logger.info(
            "Escalating to Nemotron VL",
            extra={"path": annotations["path"], "question_provided": bool(question)},
        )
        try:
            from app.vision_llm import reason_over_media

            hints = _build_hints(summary, regions)
            # Ensure question is not None before escalation
            if question is None:
                question = "What is shown in this image?"
            llm_result = await reason_over_media(
                question=question,
                media_paths=[path],
                instructions=instructions,
                hints=hints,
            )
            annotations["escalated"] = True
            annotations["nemotron_answer"] = llm_result.answer
        except Exception as exc:
            logger.warning(
                "Nemotron escalation failed",
                extra={"path": annotations["path"], "error": str(exc)},
            )
            annotations["escalation_error"] = str(exc)
    else:
        annotations["escalated"] = False

    result = VisionAnalysisResult(
        summary=summary,
        regions=regions,
        annotations=annotations,
        llm_result=llm_result,
    )
    logger.info(
        "Vision analysis complete",
        extra={
            "path": annotations["path"],
            "regions": len(regions),
            "escalated": annotations.get("escalated", False),
        },
    )
    return result


__all__ = [
    "VisionAnalysisResult",
    "VisionRegion",
    "VisionSummary",
    "analyze_image",
]
