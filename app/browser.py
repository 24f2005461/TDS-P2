from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncGenerator, Iterable, List, Optional, Sequence, cast

from playwright.async_api import BrowserContext, Page, ViewportSize, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from app.config import Settings, get_settings
from app.logging_utils import get_logger

logger = get_logger(__name__)
settings: Settings = get_settings()

_WAIT_SELECTOR_RAW = settings.playwright_wait_selector
DEFAULT_WAIT_SELECTOR = (
    None
    if _WAIT_SELECTOR_RAW is not None and not _WAIT_SELECTOR_RAW.strip()
    else (_WAIT_SELECTOR_RAW or "#result")
)
DEFAULT_VIEWPORT = {
    "width": settings.playwright_viewport_width,
    "height": settings.playwright_viewport_height,
}
RESOURCE_SUFFIXES = (
    ".pdf",
    ".csv",
    ".json",
    ".xlsx",
    ".xls",
    ".txt",
    ".tsv",
    ".png",
    ".jpg",
    ".jpeg",
    ".zip",
    ".xml",
    ".parquet",
)


def _sanitize_timeout(value: Optional[float], fallback: float) -> float:
    candidate = value if value and value > 0 else fallback
    return max(candidate, 1e-2)


@dataclass(slots=True)
class BrowserRuntimeConfig:
    """Runtime tuning knobs for the Playwright session."""

    headless: bool = settings.playwright_headless
    navigation_timeout: float = settings.playwright_navigation_timeout_seconds
    wait_timeout: float = settings.playwright_wait_timeout_seconds
    user_agent: Optional[str] = None
    viewport: Optional[dict[str, int]] = field(
        default_factory=lambda: dict(DEFAULT_VIEWPORT)
    )
    wait_selector: Optional[str] = DEFAULT_WAIT_SELECTOR


@dataclass(slots=True)
class ScrapeResult:
    """Structured snapshot of a rendered quiz page."""

    requested_url: str
    final_url: str
    html: str
    text: str
    links: List[str]
    resource_links: List[str]
    submit_urls: List[str]
    submission_url: Optional[str]
    question_snippet: Optional[str]
    instructions_snippet: Optional[str]


@asynccontextmanager
async def browser_session(
    config: Optional[BrowserRuntimeConfig] = None,
) -> AsyncGenerator[BrowserContext, None]:
    """
    Yield a Playwright browser context configured with environment-aware defaults.
    """
    cfg = config or BrowserRuntimeConfig()
    nav_timeout_ms = int(
        _sanitize_timeout(
            cfg.navigation_timeout, settings.playwright_navigation_timeout_seconds
        )
        * 1000
    )

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=cfg.headless)
    viewport_dict = cfg.viewport or DEFAULT_VIEWPORT
    context = await browser.new_context(
        user_agent=cfg.user_agent, viewport=cast(ViewportSize, viewport_dict)
    )
    context.set_default_navigation_timeout(nav_timeout_ms)
    context.set_default_timeout(nav_timeout_ms)

    try:
        yield context
    finally:
        await context.close()
        await browser.close()
        await playwright.stop()


async def _extract_links(page: Page) -> List[str]:
    raw_links: Sequence[str] = await page.eval_on_selector_all(
        "a[href]", "els => els.map(e => e.href)"
    )
    return sorted({link.strip() for link in raw_links if link.strip()})


def _filter_resources(links: Iterable[str]) -> List[str]:
    resources: List[str] = []
    for link in links:
        normalized = link.lower().split("?", 1)[0]
        if any(normalized.endswith(ext) for ext in RESOURCE_SUFFIXES):
            resources.append(link)
    return resources


def _guess_submit_urls(links: Iterable[str], html: str) -> List[str]:
    submit_candidates = [link for link in links if "submit" in link.lower()]
    for token in html.split():
        cleaned = token.strip("\"'()[]{}<>;,")
        if cleaned.startswith(("http://", "https://")) and "submit" in cleaned.lower():
            submit_candidates.append(cleaned)
    return sorted({candidate for candidate in submit_candidates})


def _extract_question_and_instructions(
    text: str,
) -> tuple[Optional[str], Optional[str]]:
    """
    Attempt to isolate the primary question/instruction block.

    Heuristics:
    - Look for lines starting with 'Q' or 'Question'
    - Extract paragraphs mentioning 'Post your answer' or 'Submit'
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    question = None
    instructions = None

    for line in lines:
        lower = line.lower()
        if question is None and (lower.startswith("q") or "question" in lower):
            question = line
        if instructions is None and (
            "post your answer" in lower
            or "submit" in lower
            or "payload" in lower
            or "answer" in lower
            and "http" in lower
        ):
            instructions = line
        if question and instructions:
            break

    return question, instructions


async def scrape_quiz_page(
    url: str,
    *,
    wait_for_selector: Optional[str] = None,
    config: Optional[BrowserRuntimeConfig] = None,
) -> ScrapeResult:
    """
    Render a quiz page, wait for JavaScript content, and collect useful artifacts.
    """
    cfg = config or BrowserRuntimeConfig()
    wait_timeout_ms = int(
        _sanitize_timeout(cfg.wait_timeout, settings.playwright_wait_timeout_seconds)
        * 1000
    )
    selector_to_wait_for = (
        wait_for_selector if wait_for_selector is not None else cfg.wait_selector
    )

    async with browser_session(cfg) as context:
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=wait_timeout_ms)
        except PlaywrightTimeoutError as exc:
            logger.warning("Navigation timeout for %s: %s", url, exc)

        # Give JavaScript a moment to execute after page load
        import asyncio

        await asyncio.sleep(0.5)

        if selector_to_wait_for:
            try:
                await page.wait_for_selector(
                    selector_to_wait_for, timeout=wait_timeout_ms
                )
            except PlaywrightTimeoutError:
                logger.info(
                    "Selector %s did not appear within %.1fs for %s",
                    selector_to_wait_for,
                    cfg.wait_timeout,
                    url,
                )

        html = await page.content()
        text = await page.inner_text("body")
        links = await _extract_links(page)
        resource_links = _filter_resources(links)
        submit_urls = _guess_submit_urls(links, html)
        question_snippet, instructions_snippet = _extract_question_and_instructions(
            text
        )
        primary_submit_url = submit_urls[0] if submit_urls else None

        return ScrapeResult(
            requested_url=url,
            final_url=page.url,
            html=html,
            text=text,
            links=links,
            resource_links=resource_links,
            submit_urls=submit_urls,
            submission_url=primary_submit_url,
            question_snippet=question_snippet,
            instructions_snippet=instructions_snippet,
        )


def scrape_quiz_page_sync(
    url: str,
    *,
    wait_for_selector: Optional[str] = None,
    config: Optional[BrowserRuntimeConfig] = None,
    loop: Optional[AbstractEventLoop] = None,
) -> ScrapeResult:
    """
    Convenience wrapper for synchronous callers (e.g., CLI smoke tests).
    """
    if loop and loop.is_running():
        raise RuntimeError(
            "scrape_quiz_page_sync cannot run inside an active event loop. "
            "Use the async variant instead."
        )

    return asyncio.run(
        scrape_quiz_page(
            url,
            wait_for_selector=wait_for_selector,
            config=config,
        )
    )
