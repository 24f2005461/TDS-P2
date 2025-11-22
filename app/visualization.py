"""
Phase 9 visualization helpers.

This module centralizes chart-related utilities so both the agent and future
FastAPI endpoints can render lightweight visual artifacts on demand.  Matplotlib
(+Seaborn styles) is used for quick static plots, while Plotly handles richer,
interactive-ready figures that we export to PNG via Kaleido.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from app.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ChartArtifact:
    """Unified payload returned by visualization helpers."""

    image_base64: str
    path: Optional[Path] = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "image_base64": self.image_base64,
            "path": str(self.path) if self.path else None,
            "metadata": dict(self.metadata),
        }
        return payload


def _ensure_directory(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _ensure_dataframe(data: Iterable[Mapping[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(list(data))


def _encode_png(buffer: bytes) -> str:
    return base64.b64encode(buffer).decode("ascii")


def _save_bytes(buffer: bytes, path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    target = _ensure_directory(path)
    target.write_bytes(buffer)
    return target


def render_matplotlib_chart(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    chart_type: str = "line",
    title: Optional[str] = None,
    style: str = "darkgrid",
    save_path: Optional[Path] = None,
) -> ChartArtifact:
    """Render a Matplotlib/Seaborn chart to PNG and return metadata."""
    if x not in df.columns or y not in df.columns:
        missing = sorted({x, y} - set(df.columns))
        raise KeyError(f"Missing columns for chart: {missing}")

    sns.set_style(style)
    fig, ax = plt.subplots(figsize=(6, 4))
    chart = chart_type.lower()

    if chart == "line":
        sns.lineplot(data=df, x=x, y=y, ax=ax)
    elif chart == "bar":
        sns.barplot(data=df, x=x, y=y, ax=ax)
    elif chart == "scatter":
        sns.scatterplot(data=df, x=x, y=y, ax=ax)
    else:
        raise ValueError(
            f"Unsupported chart_type '{chart_type}' for Matplotlib helper."
        )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)

    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)

    png_bytes = buffer.getvalue()
    stored_path = _save_bytes(png_bytes, save_path)

    metadata = {
        "library": "matplotlib",
        "chart_type": chart,
        "rows": len(df),
        "columns": list(df.columns),
    }

    artifact = ChartArtifact(
        image_base64=_encode_png(png_bytes),
        path=stored_path,
        metadata=metadata,
    )
    logger.info("Rendered Matplotlib chart", extra=artifact.metadata)
    return artifact


def render_plotly_chart(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    chart_type: str = "line",
    title: Optional[str] = None,
    template: str = "plotly_dark",
    save_path: Optional[Path] = None,
    scale: float = 2.0,
) -> ChartArtifact:
    """Render a Plotly figure and export to PNG via Kaleido."""
    if x not in df.columns or y not in df.columns:
        missing = sorted({x, y} - set(df.columns))
        raise KeyError(f"Missing columns for chart: {missing}")

    chart = chart_type.lower()
    fig = go.Figure()
    if chart == "line":
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines"))
    elif chart == "bar":
        fig.add_trace(go.Bar(x=df[x], y=df[y]))
    elif chart == "scatter":
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="markers"))
    else:
        raise ValueError(f"Unsupported chart_type '{chart_type}' for Plotly helper.")

    fig.update_layout(template=template, title=title, xaxis_title=x, yaxis_title=y)

    png_bytes = fig.to_image(format="png", scale=scale)
    stored_path = _save_bytes(png_bytes, save_path)

    metadata = {
        "library": "plotly",
        "chart_type": chart,
        "template": template,
        "rows": len(df),
        "columns": list(df.columns),
    }

    artifact = ChartArtifact(
        image_base64=_encode_png(png_bytes),
        path=stored_path,
        metadata=metadata,
    )
    logger.info("Rendered Plotly chart", extra=artifact.metadata)
    return artifact


def create_chart_from_records(
    data: Iterable[Mapping[str, Any]] | pd.DataFrame,
    *,
    x: str,
    y: str,
    engine: str = "matplotlib",
    chart_type: str = "line",
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> ChartArtifact:
    """
    High-level convenience wrapper.

    Args:
        data: Iterable of dict-like records or a DataFrame.
        x, y: Column names to plot.
        engine: 'matplotlib' or 'plotly'.
        chart_type: line|bar|scatter.
        title: Optional chart title.
        save_path: Optional filesystem location for the PNG.
        extra_metadata: Additional key/value pairs merged into the result metadata.
    """
    df = _ensure_dataframe(data)
    engine_normalized = engine.lower()

    if engine_normalized == "matplotlib":
        artifact = render_matplotlib_chart(
            df,
            x=x,
            y=y,
            chart_type=chart_type,
            title=title,
            save_path=save_path,
        )
    elif engine_normalized == "plotly":
        artifact = render_plotly_chart(
            df,
            x=x,
            y=y,
            chart_type=chart_type,
            title=title,
            save_path=save_path,
        )
    else:
        raise ValueError("engine must be either 'matplotlib' or 'plotly'.")

    if extra_metadata:
        artifact.metadata.update(extra_metadata)

    return artifact


__all__ = [
    "ChartArtifact",
    "create_chart_from_records",
    "render_matplotlib_chart",
    "render_plotly_chart",
]
