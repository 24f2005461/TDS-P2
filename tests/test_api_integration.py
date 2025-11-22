import pytest

from app import api_integration
from app.api_integration import (
    APIIntegrationError,
    APIRequest,
    APIResponse,
    extract_instruction_headers,
    parse_instruction_text,
)


def test_extract_instruction_headers_parses_prefixed_keys():
    headers = {
        "X-Quiz-Instruction-Answer-Format": "integer",
        "x-quiz-instruction-extra_hint": "use csv",
        "Content-Type": "application/json",
    }

    result = extract_instruction_headers(headers)

    assert result == {
        "answer_format": "integer",
        "extra_hint": "use csv",
    }


def test_parse_instruction_text_handles_missing_values():
    text = "answer_format: json; precision: 2; needs_auth"
    parsed = parse_instruction_text(text)

    assert parsed["answer_format"] == "json"
    assert parsed["precision"] == "2"
    assert parsed["needs_auth"] == ""


@pytest.mark.anyio
async def test_call_json_api_returns_response_payload(monkeypatch):
    recorded_request = {}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            self._closed = False

        async def send(self, request: APIRequest) -> APIResponse:
            recorded_request["method"] = request.method
            recorded_request["url"] = request.url
            recorded_request["json_body"] = request.json_body
            return APIResponse(
                status_code=200,
                headers={"content-type": "application/json"},
                text='{"ok": true}',
                json_data={"ok": True},
            )

        async def close(self):
            self._closed = True

    monkeypatch.setattr(api_integration, "AsyncAPIClient", DummyClient)

    payload = await api_integration.call_json_api(
        method="POST",
        url="https://example.com/api",
        json_body={"value": 42},
    )

    assert recorded_request["method"] == "POST"
    assert recorded_request["url"] == "https://example.com/api"
    assert recorded_request["json_body"] == {"value": 42}
    assert payload.status_code == 200
    assert payload.json_data == {"ok": True}
    assert payload.text == '{"ok": true}'


@pytest.mark.anyio
async def test_call_json_api_propagates_errors(monkeypatch):
    class FailingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def send(self, request):
            raise APIIntegrationError("boom")

        async def close(self):
            pass

    monkeypatch.setattr(api_integration, "AsyncAPIClient", FailingClient)

    with pytest.raises(APIIntegrationError):
        await api_integration.call_json_api("GET", "https://example.com")
