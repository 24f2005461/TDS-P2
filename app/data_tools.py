"""
Utility helpers for working with quiz resources such as CSV/Excel spreadsheets,
JSON payloads, and PDF documents.

These helpers are intentionally lightweight wrappers around pandas/pdfplumber so
that downstream solver components can focus on reasoning instead of repetitive
I/O glue code.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, List, Optional

import pandas as pd
import pdfplumber
from PIL import Image

from app.logging_utils import get_logger

logger = get_logger(__name__)


class DataFileError(RuntimeError):
    """Raised when an artifact cannot be parsed into the desired structure."""


@dataclass(slots=True)
class DataSummary:
    """Quick summary statistics for a tabular artifact."""

    row_count: int
    column_count: int
    columns: List[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": self.columns,
        }


def _ensure_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        raise DataFileError(f"File not found: {candidate}")
    if not candidate.is_file():
        raise DataFileError(f"Expected file but received: {candidate}")
    return candidate


def _infer_type(path: Path, explicit: Optional[str]) -> str:
    if explicit:
        return explicit.lower()
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"csv", "tsv"}:
        return "csv"
    if suffix in {"xls", "xlsx"}:
        return "excel"
    if suffix in {"json"}:
        return "json"
    if suffix in {"pdf"}:
        return "pdf"
    raise DataFileError(f"Could not infer file type for {path}")


def load_dataframe(
    path: str | Path,
    *,
    file_type: Optional[str] = None,
    **pandas_kwargs: Any,
) -> pd.DataFrame:
    """
    Load a tabular artifact (CSV/TSV/Excel/JSON records) into a pandas DataFrame.

    Args:
        path: File system location for the artifact.
        file_type: Optional override (csv, excel, json). Defaults to suffix inference.
        pandas_kwargs: Extra arguments forwarded to pandas readers.

    Returns:
        pandas.DataFrame
    """
    resolved = _ensure_path(path)
    normalized_type = _infer_type(resolved, file_type)

    if normalized_type == "csv":
        sep = pandas_kwargs.pop(
            "sep", "," if resolved.suffix.lower() == ".csv" else "\t"
        )
        df = pd.read_csv(resolved, sep=sep, **pandas_kwargs)
    elif normalized_type == "excel":
        df = pd.read_excel(resolved, **pandas_kwargs)
    elif normalized_type == "json":
        orient = pandas_kwargs.pop("orient", "records")
        df = pd.read_json(resolved, orient=orient, **pandas_kwargs)
    else:
        raise DataFileError(f"Unsupported tabular file type: {normalized_type}")

    logger.info(
        "Loaded dataframe",
        extra={
            "path": str(resolved),
            "rows": len(df),
            "columns": list(df.columns),
        },
    )
    return df


def summarize_dataframe(df: pd.DataFrame) -> DataSummary:
    """Return lightweight metadata for logging or LLM prompts."""
    summary = DataSummary(
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        columns=[str(col) for col in df.columns],
    )
    logger.debug("Data summary computed", extra=summary.to_dict())
    return summary


def read_json_payload(path: str | Path) -> Any:
    """Load arbitrary JSON content into native Python structures."""
    resolved = _ensure_path(path)
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise DataFileError(f"Invalid JSON in {resolved}: {exc}") from exc
    logger.info("Loaded JSON payload", extra={"path": str(resolved)})
    return payload


def read_text_file(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    dedupe_whitespace: bool = False,
) -> str:
    """Read plaintext resources such as instructions or prompt templates."""
    resolved = _ensure_path(path)
    try:
        content = resolved.read_text(encoding=encoding)
    except UnicodeDecodeError as exc:
        raise DataFileError(f"Failed to decode text file {resolved}: {exc}") from exc
    if dedupe_whitespace:
        content = " ".join(content.split())
    logger.info(
        "Loaded text file",
        extra={"path": str(resolved), "chars": len(content)},
    )
    return content


def load_image(path: str | Path, *, mode: Optional[str] = None) -> Image.Image:
    """
    Load an image asset (charts, diagrams, etc.) for downstream processing.

    Args:
        path: Image file location.
        mode: Optional Pillow mode conversion (e.g., "RGB", "L").
    """
    resolved = _ensure_path(path)
    try:
        image = Image.open(resolved)
    except Exception as exc:  # pragma: no cover - Pillow internals
        raise DataFileError(f"Failed to load image {resolved}: {exc}") from exc
    if mode:
        image = image.convert(mode)
    logger.info(
        "Loaded image",
        extra={
            "path": str(resolved),
            "size": image.size,
            "mode": image.mode,
        },
    )
    return image


def image_to_base64(
    path: str | Path,
    *,
    mode: Optional[str] = None,
    format: Optional[str] = None,
) -> str:
    """Convert an image file to a Base64-encoded string (suitable for JSON payloads)."""
    resolved = _ensure_path(path)
    image = load_image(resolved, mode=mode)
    buffer = BytesIO()
    output_format = format or image.format or "PNG"
    try:
        image.save(buffer, format=output_format)
    except ValueError as exc:
        raise DataFileError(f"Failed to encode image {resolved}: {exc}") from exc
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    logger.info(
        "Encoded image to base64",
        extra={"path": str(resolved), "format": output_format, "length": len(encoded)},
    )
    return encoded


def extract_pdf_text(
    path: str | Path,
    *,
    max_pages: Optional[int] = None,
    dedupe_whitespace: bool = True,
) -> str:
    """
    Extract human-readable text from a PDF using pdfplumber.

    Args:
        path: PDF file location.
        max_pages: Optional cap to avoid heavy processing (e.g., 10).
        dedupe_whitespace: Collapse multiple whitespace runs into single spaces.

    Returns:
        Combined text from the processed pages.
    """
    resolved = _ensure_path(path)
    texts: List[str] = []
    try:
        with pdfplumber.open(resolved) as pdf:
            page_total = len(pdf.pages)
            target_pages = (
                range(page_total)
                if max_pages is None
                else range(min(max_pages, page_total))
            )
            for index in target_pages:
                page = pdf.pages[index]
                text = page.extract_text() or ""
                texts.append(text)
    except Exception as exc:  # pragma: no cover (pdfplumber internal errors)
        raise DataFileError(f"Failed to read PDF {resolved}: {exc}") from exc

    combined = "\n\n".join(texts)
    if dedupe_whitespace:
        combined = " ".join(combined.split())

    logger.info(
        "Extracted PDF text",
        extra={"path": str(resolved), "chars": len(combined), "pages": len(texts)},
    )
    return combined


def extract_pdf_tables(
    path: str | Path,
    *,
    max_pages: Optional[int] = None,
) -> List[pd.DataFrame]:
    """
    Extract tabular data from a PDF.

    Returns:
        List of DataFrames (one per detected table).
    """
    resolved = _ensure_path(path)
    tables: List[pd.DataFrame] = []
    try:
        with pdfplumber.open(resolved) as pdf:
            page_total = len(pdf.pages)
            target_pages = (
                range(page_total)
                if max_pages is None
                else range(min(max_pages, page_total))
            )
            for index in target_pages:
                page = pdf.pages[index]
                raw_tables = page.extract_tables() or []
                for table in raw_tables:
                    df = pd.DataFrame(table)
                    tables.append(df)
    except Exception as exc:  # pragma: no cover
        raise DataFileError(f"Failed to extract tables from {resolved}: {exc}") from exc

    logger.info(
        "Extracted PDF tables",
        extra={"path": str(resolved), "table_count": len(tables)},
    )
    return tables


def dataframe_to_records(df: pd.DataFrame) -> List[dict[str, Any]]:
    """Convert a DataFrame to a list of dictionaries with string keys."""
    records = df.to_dict(orient="records")
    logger.debug("Converted dataframe to records", extra={"records": len(records)})
    return records


def chunk_dataframe(
    df: pd.DataFrame,
    *,
    chunk_size: int,
) -> Iterable[pd.DataFrame]:
    """
    Yield DataFrame chunks to keep LLM/tool prompts within token or memory limits.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    total_rows = len(df)
    for start in range(0, total_rows, chunk_size):
        yield df.iloc[start : start + chunk_size]


__all__ = [
    "DataFileError",
    "DataSummary",
    "chunk_dataframe",
    "dataframe_to_records",
    "extract_pdf_tables",
    "extract_pdf_text",
    "load_dataframe",
    "load_image",
    "image_to_base64",
    "read_json_payload",
    "read_text_file",
    "summarize_dataframe",
]
