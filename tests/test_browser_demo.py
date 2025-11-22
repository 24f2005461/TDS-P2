import pytest

from app.browser import scrape_quiz_page


@pytest.mark.anyio
async def test_scrape_example_com_contains_expected_text():
    """Smoke-test the Playwright scraper end-to-end against example.com."""
    url = "https://example.com"
    result = await scrape_quiz_page(url)

    assert "Example Domain" in result.text
    assert "Example Domain" in result.html
    assert result.submission_url is None
    assert result.requested_url == url
