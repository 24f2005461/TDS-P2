"""
Async resource downloader module.

Features:
- Shared httpx.AsyncClient with configurable timeout/headers.
- Temporary-file streaming with atomic move to final destination.
- Bounded concurrency helpers for batch downloads.
- SHA-256 hashing of streamed content for downstream integrity checks.
- High-level convenience wrappers returning structured metadata.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import httpx

from app.config import Settings, get_settings
from app.logging_utils import get_logger

logger = get_logger(__name__)


class DownloadError(RuntimeError):
    """Raised when a download fails or violates safety limits."""


@dataclass(slots=True)
class DownloadResult:
    """Structured response for completed downloads."""

    url: str
    path: Path
    bytes_downloaded: int
    content_type: Optional[str]
    filename: str
    content_hash: Optional[str]


def _sanitize_filename(raw: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw
    )
    return cleaned or "resource"


def _derive_filename(url: str) -> str:
    parsed = urlparse(url)
    candidate = os.path.basename(parsed.path.rstrip("/")) or "resource"
    return _sanitize_filename(candidate)


def _resolve_target_path(destination: Path, filename: str, overwrite: bool) -> Path:
    candidate = destination / filename
    if overwrite or not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        next_candidate = destination / f"{stem}_{counter}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        counter += 1


class AsyncDownloader:
    """Async downloader with streaming + concurrency controls."""

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        client: Optional[httpx.AsyncClient] = None,
        concurrency: Optional[int] = None,
    ) -> None:
        self.settings = settings or get_settings()
        timeout = httpx.Timeout(self.settings.downloader_request_timeout_seconds)
        self._client = client or httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
        )
        self._owns_client = client is None
        max_concurrency = concurrency or self.settings.downloader_concurrency
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def __aenter__(self) -> "AsyncDownloader":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def download(
        self,
        url: str,
        destination_dir: Path,
        *,
        filename: Optional[str] = None,
        overwrite: bool = False,
        chunk_size: int = 65536,
    ) -> DownloadResult:
        destination_root = Path(destination_dir)
        destination_root.mkdir(parents=True, exist_ok=True)
        inferred_name = filename or _derive_filename(url)
        final_path = _resolve_target_path(destination_root, inferred_name, overwrite)
        final_name = final_path.name
        content_hash: Optional[str] = None
        content_type: Optional[str] = None
        bytes_written = 0

        async with self._semaphore:
            logger.info(
                "Downloading resource", extra={"url": url, "path": str(final_path)}
            )
            try:
                async with self._client.stream("GET", url) as response:
                    response.raise_for_status()

                    bytes_limit = self.settings.downloader_max_file_size_bytes
                    content_type = response.headers.get("content-type")
                    hasher = hashlib.sha256()
                    content_length = response.headers.get("content-length")
                    if content_length:
                        try:
                            declared_size = int(content_length)
                        except ValueError:
                            declared_size = None
                        else:
                            if declared_size > bytes_limit:
                                raise DownloadError(
                                    f"Download exceeded size limit ({declared_size} > {bytes_limit})"
                                )

                    with tempfile.NamedTemporaryFile(
                        dir=destination_root, delete=False
                    ) as tmp_handle:
                        tmp_path = Path(tmp_handle.name)
                        try:
                            async for chunk in response.aiter_bytes(
                                chunk_size=chunk_size
                            ):
                                bytes_written += len(chunk)
                                if bytes_written > bytes_limit:
                                    raise DownloadError(
                                        f"Download exceeded size limit ({bytes_written} > {bytes_limit})"
                                    )
                                tmp_handle.write(chunk)
                                hasher.update(chunk)
                        except BaseException:
                            tmp_handle.close()
                            tmp_path.unlink(missing_ok=True)
                            raise

                    tmp_path.replace(final_path)
                    content_hash = hasher.hexdigest()
            except httpx.HTTPError as exc:
                raise DownloadError(f"Failed to download {url}: {exc}") from exc

        logger.info(
            "Download complete",
            extra={
                "url": url,
                "path": str(final_path),
                "size_bytes": bytes_written,
                "content_type": content_type,
                "sha256": content_hash,
            },
        )
        return DownloadResult(
            url=url,
            path=final_path,
            bytes_downloaded=bytes_written,
            content_type=content_type,
            filename=final_name,
            content_hash=content_hash,
        )

    async def download_many(
        self,
        urls: Sequence[str],
        destination_dir: Path,
        *,
        overwrite: bool = False,
    ) -> List[DownloadResult]:
        tasks = [
            self.download(url, destination_dir, overwrite=overwrite)
            for url in urls
            if url
        ]
        results: List[DownloadResult] = []
        failures: List[Tuple[str, BaseException]] = []

        for coro in asyncio.as_completed(tasks):
            try:
                results.append(await coro)
            except BaseException as exc:  # pragma: no cover
                failures.append((getattr(exc, "url", "unknown"), exc))

        if failures:
            first_url, first_exc = failures[0]
            raise DownloadError(
                f"{len(failures)} download(s) failed. First failure ({first_url}): {first_exc}"
            ) from first_exc

        return results


async def download_file(
    url: str,
    destination_dir: Path,
    *,
    filename: Optional[str] = None,
    overwrite: bool = False,
    settings: Optional[Settings] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> DownloadResult:
    destination_root = Path(destination_dir)
    async with AsyncDownloader(settings=settings, client=client) as downloader:
        return await downloader.download(
            url,
            destination_root,
            filename=filename,
            overwrite=overwrite,
        )


async def download_files(
    urls: Iterable[str],
    destination_dir: Path,
    *,
    overwrite: bool = False,
    settings: Optional[Settings] = None,
    client: Optional[httpx.AsyncClient] = None,
    concurrency: Optional[int] = None,
) -> List[DownloadResult]:
    url_list = [url for url in urls if url]
    if not url_list:
        return []

    destination_root = Path(destination_dir)
    async with AsyncDownloader(
        settings=settings,
        client=client,
        concurrency=concurrency,
    ) as downloader:
        return await downloader.download_many(
            url_list,
            destination_root,
            overwrite=overwrite,
        )


__all__ = [
    "AsyncDownloader",
    "DownloadError",
    "DownloadResult",
    "download_file",
    "download_files",
]
