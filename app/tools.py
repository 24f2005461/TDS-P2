"""Structured tool registry and Phase 6 helper surface.

This module exposes a centralized registry for every helper the solver agent is
allowed to call.  Tools wrap downloader/data/executor/chart utilities behind
async-friendly callables so the orchestrator can await everything uniformly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Union,
)
from urllib.parse import urljoin, urlparse

import pandas as pd

from app import data_tools
from app.api_integration import (
    call_json_api as api_call_json,
)
from app.api_integration import (
    extract_instruction_headers,
)
from app.api_integration import (
    parse_instruction_text as api_parse_instruction_text,
)
from app.downloader import DownloadResult, download_file, download_files
from app.executor import execute_python, run_trusted_callable
from app.vision import analyze_image as vision_analyze_image
from app.vision_llm import reason_over_media
from app.visualization import create_chart_from_records


class AsyncTool(Protocol):
    async def __call__(self, *args, **kwargs): ...


ToolFn = Union[Callable[..., Awaitable[Any]], AsyncTool]


class ToolRegistry:
    """Simple name → callable registry with async wrappers."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolFn] = {}

    def register(self, name: str, func: ToolFn) -> None:
        self._tools[name] = func

    def get(self, name: str) -> ToolFn:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self._tools[name]

    def names(self) -> List[str]:
        return sorted(self._tools)


registry = ToolRegistry()


async def _maybe_await(value):
    if asyncio.iscoroutine(value) or isinstance(value, Awaitable):
        return await value
    return value


# --------------------------------------------------------------------------- #
# Downloader helpers
# --------------------------------------------------------------------------- #


async def download_resource(url: str, destination_dir: Path) -> DownloadResult:
    return await download_file(url, destination_dir)


async def download_resources(
    urls: Iterable[str], destination_dir: Path
) -> List[DownloadResult]:
    return await download_files(urls, destination_dir)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #


async def load_table(path: Path, file_type: Optional[str] = None):
    return await _maybe_await(data_tools.load_dataframe(path, file_type=file_type))


async def save_table(
    df: pd.DataFrame,
    path: Path,
    *,
    file_type: str = "csv",
    index: bool = False,
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if file_type.lower() == "csv":
        df.to_csv(destination, index=index)
    elif file_type.lower() in {"xlsx", "excel"}:
        df.to_excel(destination, index=index)
    elif file_type.lower() == "json":
        df.to_json(destination, orient="records", force_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported file_type '{file_type}'")

    return str(destination)


async def summarize_table(df: pd.DataFrame) -> Dict[str, Any]:
    return data_tools.summarize_dataframe(df).to_dict()


async def describe_dataframe(
    df: pd.DataFrame,
    *,
    percentiles: Optional[List[float]] = None,
    include_all_columns: bool = False,
) -> Dict[str, Dict[str, Any]]:
    kwargs: Dict[str, Any] = {}
    if percentiles:
        kwargs["percentiles"] = percentiles
    if include_all_columns:
        kwargs["include"] = "all"
    return df.describe(**kwargs).to_dict(orient="index")


async def value_counts(
    df: pd.DataFrame,
    *,
    column: str,
    normalize: bool = False,
    dropna: bool = True,
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe.")
    counts = df[column].value_counts(dropna=dropna, normalize=normalize)
    if top_n is not None:
        counts = counts.head(top_n)

    results: List[Dict[str, Any]] = []
    for value, count in counts.items():
        serialized_value = None if bool(pd.isna(value)) else value
        results.append(
            {
                "value": serialized_value,
                "count": float(count) if normalize else int(count),
            }
        )
    return results


async def chunk_dataframe(df: pd.DataFrame, *, chunk_size: int):
    return [chunk for chunk in data_tools.chunk_dataframe(df, chunk_size=chunk_size)]


async def dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return data_tools.dataframe_to_records(df)


async def read_json(path: Path):
    return data_tools.read_json_payload(path)


async def read_text(path: Path, *, dedupe_whitespace: bool = False) -> str:
    return data_tools.read_text_file(path, dedupe_whitespace=dedupe_whitespace)


async def write_text(path: Path, content: str, *, encoding: str = "utf-8") -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding=encoding)
    return str(destination)


async def extract_pdf_text(path: Path, max_pages: Optional[int] = None) -> str:
    return data_tools.extract_pdf_text(path, max_pages=max_pages)


async def extract_pdf_tables(path: Path, max_pages: Optional[int] = None):
    return data_tools.extract_pdf_tables(path, max_pages=max_pages)


async def read_pdf(path: Path, max_pages: Optional[int] = None) -> str:
    return await extract_pdf_text(path, max_pages=max_pages)


async def read_csv(path: Path):
    return await load_table(path, file_type="csv")


async def read_excel(path: Path):
    return await load_table(path, file_type="excel")


async def load_image(path: Path):
    return data_tools.load_image(path)


async def image_to_base64(path: Path):
    return data_tools.image_to_base64(path)


async def encode_image(path: Path):
    return await image_to_base64(path)


# --------------------------------------------------------------------------- #
# Chart helpers
# --------------------------------------------------------------------------- #


async def create_chart(
    data: Iterable[Dict[str, Any]] | pd.DataFrame,
    *,
    x: str,
    y: str,
    chart_type: str = "line",
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    engine: str = "matplotlib",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    artifact = create_chart_from_records(
        data,
        x=x,
        y=y,
        engine=engine,
        chart_type=chart_type,
        title=title,
        save_path=save_path,
        extra_metadata=extra_metadata,
    )
    payload = artifact.to_dict()
    payload.setdefault(
        "chart_type",
        artifact.metadata.get("chart_type", chart_type.lower()),
    )
    return payload


async def analyze_image_tool(
    path: Path,
    *,
    grid_rows: int = 4,
    grid_cols: int = 3,
    blur_before_scan: bool = True,
    question: Optional[str] = None,
    instructions: Optional[str] = None,
) -> Dict[str, Any]:
    result = await vision_analyze_image(
        path,
        grid_shape=(grid_rows, grid_cols),
        blur_before_scan=blur_before_scan,
        question=question,
        instructions=instructions,
    )
    return result.to_dict()


analyze_image = analyze_image_tool


async def vision_reasoner(
    path: Path,
    *,
    question: str,
    instructions: Optional[str] = None,
    grid_rows: int = 4,
    grid_cols: int = 3,
) -> Dict[str, Any]:
    """
    Analyze an image with Nemotron VL multimodal reasoning.

    This tool performs heuristic analysis first, then escalates to Nemotron
    Nano 2 VL when entropy or region scores exceed configured thresholds.
    The question parameter is required to trigger escalation.

    Args:
        path: Path to the image file.
        question: The question or task to ask the vision model.
        instructions: Optional constraints or format instructions.
        grid_rows: Number of grid rows for heuristic region scanning.
        grid_cols: Number of grid columns for heuristic region scanning.

    Returns:
        A dictionary containing heuristic summary, regions, and Nemotron answer
        (if escalation occurred).
    """
    result = await vision_analyze_image(
        path,
        grid_shape=(grid_rows, grid_cols),
        blur_before_scan=True,
        question=question,
        instructions=instructions,
    )
    return result.to_dict()


async def reason_over_image(
    path: Path,
    *,
    question: str,
    instructions: Optional[str] = None,
    hints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Direct Nemotron VL reasoning without heuristic pre-processing.

    This tool bypasses the heuristic layer and sends the image directly to
    Nemotron Nano 2 VL. Use this when you need guaranteed multimodal reasoning
    regardless of image characteristics.

    Args:
        path: Path to the image file.
        question: The question or task to ask the vision model.
        instructions: Optional constraints or format instructions.
        hints: Optional list of hint strings to guide the model.

    Returns:
        A dictionary containing the Nemotron VL result.
    """
    result = await reason_over_media(
        question=question,
        media_paths=[path],
        instructions=instructions,
        hints=hints,
    )
    return result.to_dict()


async def call_json_api(
    method: str,
    url: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
    json_body: Any = None,
    headers: Optional[Mapping[str, str]] = None,
    auth_token: Optional[str] = None,
    timeout: Optional[float] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call a JSON API endpoint.

    If url is relative (starts with /) and base_url is provided,
    resolves the URL relative to base_url.
    """
    # Resolve relative URLs
    resolved_url = url
    if url.startswith("/") and base_url:
        resolved_url = urljoin(base_url, url)
    elif not urlparse(url).scheme:
        # URL has no scheme (http/https), try to resolve with base
        if base_url:
            resolved_url = urljoin(base_url, url)
        else:
            # No base_url provided, can't resolve
            raise ValueError(
                f"Relative URL '{url}' provided without base_url. "
                "Either provide a full URL (http://...) or set base_url parameter."
            )

    response = await api_call_json(
        method,
        resolved_url,
        params=params,
        json_body=json_body,
        headers=headers,
        auth_token=auth_token,
        timeout=timeout,
    )
    return {
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "json": response.json_data,
        "text": response.text,
    }


async def parse_instruction_headers(
    headers: Mapping[str, str],
    *,
    prefix: str = "x-quiz-instruction-",
) -> Dict[str, str]:
    return extract_instruction_headers(headers, prefix=prefix)


async def parse_instruction_text(
    text: str,
    *,
    pair_delimiter: str = ";",
    kv_delimiter: str = ":",
) -> Dict[str, str]:
    return api_parse_instruction_text(
        text,
        pair_delimiter=pair_delimiter,
        kv_delimiter=kv_delimiter,
    )


# --------------------------------------------------------------------------- #
# Executor helpers
# --------------------------------------------------------------------------- #


async def run_python(code: str, *, timeout_seconds: float = 5.0):
    return await execute_python(code, timeout_seconds=timeout_seconds)


async def run_trusted(func: Callable[..., Any], *args, **kwargs):
    return run_trusted_callable(func, *args, **kwargs).to_dict()


# --------------------------------------------------------------------------- #
# Registry population
# --------------------------------------------------------------------------- #

registry.register("download_resource", download_resource)
registry.register("download_resources", download_resources)
registry.register("download_file", download_resource)
registry.register("load_table", load_table)
registry.register("save_table", save_table)
registry.register("summarize_table", summarize_table)
registry.register("describe_dataframe", describe_dataframe)
registry.register("value_counts", value_counts)
registry.register("chunk_dataframe", chunk_dataframe)
registry.register("dataframe_to_records", dataframe_to_records)
registry.register("read_csv", read_csv)
registry.register("read_excel", read_excel)
registry.register("read_json", read_json)
registry.register("read_text", read_text)
registry.register("write_text", write_text)
registry.register("extract_pdf_text", extract_pdf_text)
registry.register("extract_pdf_tables", extract_pdf_tables)
registry.register("read_pdf", read_pdf)
registry.register("load_image", load_image)
registry.register("image_to_base64", image_to_base64)
registry.register("encode_image", encode_image)
registry.register("create_chart", create_chart)
registry.register("render_chart", create_chart)
registry.register("analyze_image", analyze_image_tool)
registry.register("vision_reasoner", vision_reasoner)
registry.register("reason_over_image", reason_over_image)
registry.register("call_json_api", call_json_api)
registry.register("parse_instruction_headers", parse_instruction_headers)
registry.register("parse_instruction_text", parse_instruction_text)
registry.register("run_python", run_python)
registry.register("execute_python", run_python)
registry.register("run_trusted", run_trusted)

__all__ = [
    "registry",
    "download_resource",
    "download_resources",
    "download_file",
    "load_table",
    "save_table",
    "summarize_table",
    "describe_dataframe",
    "value_counts",
    "chunk_dataframe",
    "dataframe_to_records",
    "read_csv",
    "read_excel",
    "read_json",
    "read_text",
    "write_text",
    "extract_pdf_text",
    "extract_pdf_tables",
    "read_pdf",
    "load_image",
    "image_to_base64",
    "encode_image",
    "create_chart",
    "analyze_image",
    "analyze_image_tool",
    "vision_reasoner",
    "reason_over_image",
    "call_json_api",
    "parse_instruction_headers",
    "parse_instruction_text",
    "run_python",
    "execute_python",
    "run_trusted",
]
