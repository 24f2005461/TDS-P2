import json

import httpx
import pytest

from app.config import get_settings
from app.submission import SubmissionError, build_submission_payload, submit_answer


def _make_response(
    status_code: int,
    *,
    url: str,
    json_data: dict | None = None,
    text_data: str | None = None,
) -> httpx.Response:
    request = httpx.Request("POST", url)
    if json_data is not None:
        return httpx.Response(status_code, json=json_data, request=request)
    return httpx.Response(status_code, text=text_data or "", request=request)


class StubClient:
    def __init__(self, response: httpx.Response):
        self._response = response
        self.requests: list[dict] = []

    async def post(self, url: str, content: bytes, headers: dict[str, str]):
        self.requests.append({"url": url, "content": content, "headers": headers})
        return self._response


def test_build_submission_payload_merges_extras_without_overwriting():
    payload = build_submission_payload(
        email="user@example.com",
        secret="super-secret",
        quiz_url="https://quiz.local/q1",
        answer=123,
        extras={
            "notes": "keep",
            "answer": "ignored",
            "metadata": {"difficulty": "easy"},
        },
    )

    assert payload["email"] == "user@example.com"
    assert payload["url"] == "https://quiz.local/q1"
    assert payload["answer"] == 123
    assert payload["notes"] == "keep"
    assert payload["metadata"] == {"difficulty": "easy"}
    assert payload["secret"] == "super-secret"


@pytest.mark.anyio
async def test_submit_answer_success_parses_response():
    submission_url = "https://example.com/submit"
    quiz_url = "https://example.com/quiz"
    client = StubClient(
        _make_response(
            200,
            url=submission_url,
            json_data={
                "correct": True,
                "url": "https://example.com/next",
                "reason": None,
            },
        )
    )

    result = await submit_answer(
        submission_url=submission_url,
        email="student@example.com",
        secret="s3cr3t",
        quiz_url=quiz_url,
        answer={"value": 42},
        client=client,
    )

    assert result.correct is True
    assert result.next_url == "https://example.com/next"
    assert result.status_code == 200
    assert result.reason is None
    assert len(client.requests) == 1

    posted_payload = json.loads(client.requests[0]["content"].decode("utf-8"))
    assert posted_payload["answer"] == {"value": 42}
    assert posted_payload["email"] == "student@example.com"
    assert posted_payload["url"] == quiz_url


@pytest.mark.anyio
async def test_submit_answer_raises_for_non_json_response():
    submission_url = "https://example.com/submit"
    client = StubClient(
        _make_response(200, url=submission_url, text_data="<html>not json</html>")
    )

    with pytest.raises(SubmissionError, match="non-JSON"):
        await submit_answer(
            submission_url=submission_url,
            email="student@example.com",
            secret="s3cr3t",
            quiz_url="https://example.com/quiz",
            answer="abc",
            client=client,
        )


@pytest.mark.anyio
async def test_submit_answer_wraps_http_errors():
    submission_url = "https://example.com/submit"
    client = StubClient(
        _make_response(500, url=submission_url, json_data={"error": "boom"})
    )

    with pytest.raises(SubmissionError, match="Failed to submit answer"):
        await submit_answer(
            submission_url=submission_url,
            email="student@example.com",
            secret="s3cr3t",
            quiz_url="https://example.com/quiz",
            answer=0,
            client=client,
        )


@pytest.mark.anyio
async def test_submit_answer_respects_payload_size_limit(monkeypatch):
    monkeypatch.setenv("MAX_PAYLOAD_SIZE_MB", "0")
    get_settings.cache_clear()

    with pytest.raises(SubmissionError, match="payload exceeded"):
        await submit_answer(
            submission_url="https://example.com/submit",
            email="student@example.com",
            secret="s3cr3t",
            quiz_url="https://example.com/quiz",
            answer="will exceed zero byte limit",
        )

    get_settings.cache_clear()
