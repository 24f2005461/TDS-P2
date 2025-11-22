import hashlib
from pathlib import Path
from typing import List

import httpx
import pytest

from app.downloader import AsyncDownloader, DownloadError


class DummySettings:
    """Minimal settings stub to drive AsyncDownloader in tests."""

    def __init__(
        self,
        *,
        timeout: float = 5.0,
        concurrency: int = 2,
        max_bytes: int = 1024 * 1024,
    ) -> None:
        self.downloader_request_timeout_seconds = timeout
        self.downloader_concurrency = concurrency
        self._max_bytes = max_bytes

    @property
    def downloader_max_file_size_bytes(self) -> int:
        return self._max_bytes


def _build_transport(responses: List[bytes], content_type: str = "text/plain"):
    """Return a MockTransport that yields the provided byte payloads sequentially."""

    def handler(request: httpx.Request) -> httpx.Response:
        if not responses:
            return httpx.Response(500, text="no payload available")
        payload = responses.pop(0)
        return httpx.Response(
            200,
            content=payload,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(len(payload)),
            },
        )

    return httpx.MockTransport(handler)


@pytest.mark.anyio
async def test_download_file_persists_content_and_hash(tmp_path: Path):
    payload = b"col1\n1\n"
    transport = _build_transport([payload], content_type="text/csv")
    settings = DummySettings(max_bytes=1024)

    async with httpx.AsyncClient(transport=transport) as client:
        async with AsyncDownloader(settings=settings, client=client) as downloader:
            result = await downloader.download("https://example.com/data.csv", tmp_path)

    assert result.bytes_downloaded == len(payload)
    assert result.content_type == "text/csv"
    assert result.path.exists()
    assert result.filename == "data.csv"
    with result.path.open("rb") as handle:
        assert handle.read() == payload
    expected_hash = hashlib.sha256(payload).hexdigest()
    assert result.content_hash == expected_hash


@pytest.mark.anyio
async def test_download_creates_incremented_filename_when_exists(tmp_path: Path):
    first_payload = b"first"
    second_payload = b"second"
    transport = _build_transport([first_payload, second_payload])
    settings = DummySettings(max_bytes=1024)

    async with httpx.AsyncClient(transport=transport) as client:
        async with AsyncDownloader(settings=settings, client=client) as downloader:
            first = await downloader.download(
                "https://example.com/files/report.pdf", tmp_path
            )
            second = await downloader.download(
                "https://example.com/files/report.pdf", tmp_path
            )

    assert first.path.name == "report.pdf"
    assert second.path.name == "report_1.pdf"
    with first.path.open("rb") as handle:
        assert handle.read() == first_payload
    with second.path.open("rb") as handle:
        assert handle.read() == second_payload


@pytest.mark.anyio
async def test_download_rejects_when_payload_exceeds_limit(tmp_path: Path):
    payload = b"0123456789"  # 10 bytes
    transport = _build_transport([payload])
    settings = DummySettings(max_bytes=5)  # Force tiny limit

    async with httpx.AsyncClient(transport=transport) as client:
        async with AsyncDownloader(settings=settings, client=client) as downloader:
            with pytest.raises(DownloadError):
                await downloader.download("https://example.com/too-big.bin", tmp_path)

    # Ensure no partial file was left behind
    assert not any(tmp_path.iterdir())
