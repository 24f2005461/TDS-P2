import sys
from pathlib import Path
from typing import List

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.llm import TaskParseResult, parse_task


@pytest.fixture(autouse=True)
def configure_settings(monkeypatch):
    """Ensure Settings() reads the monkeypatched environment for every test."""
    monkeypatch.setenv("EMAIL", "test@example.com")
    monkeypatch.setenv("SECRET", "super-secret")
    monkeypatch.setenv("AIPIPE_API_KEY", "dummy-token")
    monkeypatch.setenv("AIPIPE_ENDPOINT", "https://aipipe.org/openrouter/v1")
    monkeypatch.setenv(
        "AIPIPE_GEMINIV1BETA_ENDPOINT", "https://aipipe.org/geminiv1beta"
    )
    monkeypatch.setenv("LLM_DEFAULT_MODEL", "x-ai/grok-4.1-fast:free")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "moonshotai/kimi-k2:free")

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def anyio_backend():
    """Restrict pytest-anyio to asyncio to avoid trio dependency."""
    return "asyncio"


class FakeSuccessClient:
    """In-memory stand-in for the AIPipe client returning predetermined JSON output."""

    async def chat_completion(self, *, model, messages, temperature, max_output_tokens):
        _ = (model, messages, temperature, max_output_tokens)
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"question":"Q123. Summarize the data?",'
                            '"submission_url":"https://quiz.example.com/submit",'
                            '"answer_format":"integer",'
                            '"resources":["https://quiz.example.com/data.csv"],'
                            '"instructions":"Download the CSV and compute the sum.",'
                            '"reasoning":"Identified explicit instructions in DOM."}'
                        )
                    }
                }
            ]
        }


class FakeFallbackSuccessClient:
    """Client that fails on the default model but succeeds with the fallback model."""

    def __init__(self):
        self.seen_models: List[str] = []

    async def chat_completion(self, *, model, messages, temperature, max_output_tokens):
        _ = (messages, temperature, max_output_tokens)
        self.seen_models.append(model)
        if model == "x-ai/grok-4.1-fast:free":
            raise httpx.HTTPError("default model failed")
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"question":"Fallback question",'
                            '"submission_url":"https://quiz.example.com/fallback-submit",'
                            '"answer_format":"string",'
                            '"resources":[],'
                            '"instructions":"Use fallback instructions.",'
                            '"reasoning":"Parsed with fallback model."}'
                        )
                    }
                }
            ]
        }


class FakeFailureClient:
    """Client that simulates transport issues to trigger the fallback parser."""

    async def chat_completion(self, **_kwargs):
        raise httpx.HTTPError("synthetic failure")


@pytest.mark.anyio
async def test_parse_task_with_fake_llm_client():
    """Verify that parse_task returns structured data from a fake LLM response."""
    html = """
    <div class="question">
        <p>Q123. Summarize the data?</p>
        <a href="https://quiz.example.com/data.csv">Download CSV</a>
        <p>Submit your answer via https://quiz.example.com/submit</p>
    </div>
    """
    result = await parse_task(
        html_content=html,
        model="fake/test-model",
        client=FakeSuccessClient(),
    )

    assert isinstance(result, TaskParseResult)
    assert result.question == "Q123. Summarize the data?"
    assert result.submission_url == "https://quiz.example.com/submit"
    assert result.answer_format == "integer"
    assert result.resources == ["https://quiz.example.com/data.csv"]
    assert "Download the CSV" in (result.instructions or "")
    assert "Identified explicit instructions" in (result.reasoning or "")


@pytest.mark.anyio
async def test_parse_task_fallback_on_llm_failure():
    """Ensure the heuristic fallback kicks in when the LLM client errors out."""
    html = """
    <section>
        <p>Please download <a href="https://quiz.example.com/report.pdf">the PDF</a>.</p>
        <p>Submit results at https://quiz.example.com/submit-final</p>
    </section>
    """
    result = await parse_task(
        html_content=html,
        model="fake/test-model",
        client=FakeFailureClient(),
    )

    assert result.question is None
    assert result.submission_url == "https://quiz.example.com/submit-final"
    assert result.resources == ["https://quiz.example.com/report.pdf"]
    assert result.answer_format is None
    assert result.instructions == "LLM unavailable; fallback heuristics applied."
    assert "LLM request failed" in (result.reasoning or "")


@pytest.mark.anyio
async def test_parse_task_switches_to_fallback_model():
    """Fallback model should be attempted when the primary model raises errors."""
    html = """
    <div>
        <p>This content forces the default LLM to fail so the fallback is used.</p>
    </div>
    """
    client = FakeFallbackSuccessClient()
    result = await parse_task(
        html_content=html,
        model="x-ai/grok-4.1-fast:free",
        client=client,
    )

    assert result.question == "Fallback question"
    assert result.submission_url == "https://quiz.example.com/fallback-submit"
    assert result.instructions == "Use fallback instructions."
    assert client.seen_models == ["x-ai/grok-4.1-fast:free", "moonshotai/kimi-k2:free"]


@pytest.mark.anyio
async def test_heuristic_parser_detects_multiple_resources_and_submission_links():
    """Fallback heuristics should deduplicate resources and capture the first submit link."""
    html = """
    <article>
        <p>Resources: <a href="https://quiz.example.com/data.csv">csv</a>,
        <a href="https://quiz.example.com/data.xlsx?token=abc">excel</a>,
        <a href="https://quiz.example.com/archive.zip">zip</a></p>
        <p>Submit at https://quiz.example.com/submit-one and also mention
        https://quiz.example.com/submit-two (but first should win).</p>
    </article>
    """
    result = await parse_task(
        html_content=html,
        model="fake/test-model",
        client=FakeFailureClient(),
    )

    assert result.submission_url == "https://quiz.example.com/submit-one"
    assert result.resources == [
        "https://quiz.example.com/archive.zip",
        "https://quiz.example.com/data.csv",
        "https://quiz.example.com/data.xlsx",
    ]
    assert result.instructions == "LLM unavailable; fallback heuristics applied."
